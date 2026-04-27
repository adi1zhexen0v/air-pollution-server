import cron from "node-cron";
import "dotenv/config";
import { Measurement } from "../models/Measurement.js";
import { Alert } from "../models/Alert.js";
import { sendAlertEmail } from "./emailAlert.js";

const KNOWN_DEVICES = ["Sensor-1", "Sensor-2", "Sensor-3"];
const OFFLINE_THRESHOLD_MIN = 30;
const COMPLETENESS_THRESHOLD = 4;
const ANOMALY_WINDOW = 6;

async function hasUnresolvedAlert(query) {
  const existing = await Alert.findOne({ ...query, resolved: false });
  return !!existing;
}

async function resolveAlerts(query) {
  const result = await Alert.updateMany(
    { ...query, resolved: false },
    { $set: { resolved: true, resolvedAt: new Date() } }
  );
  if (result.modifiedCount > 0) {
    console.log(`[INFO] Resolved ${result.modifiedCount} alert(s): ${query.type} ${query.deviceId}${query.field ? ` (${query.field})` : ""}`);
  }
}

function buildEmailAlert(alertDoc) {
  const { type, deviceId, field, severity, message, details } = alertDoc;

  if (type === "SENSOR_OFFLINE") {
    return {
      severity,
      title: `${deviceId} — Offline`,
      description: message,
      action: `Check physical power supply and network connectivity on ${deviceId}. Verify the ESP32 is running and connected to Wi-Fi. If the device was intentionally powered off, this alert can be ignored.`,
      details: [
        { label: "Device", value: deviceId },
        { label: "Status", value: "No data received", color: "#e24b4a", bold: true },
        { label: "Threshold", value: `${OFFLINE_THRESHOLD_MIN} minutes` },
        ...(details?.lastSeen
          ? [{ label: "Last seen", value: formatGMT5Short(details.lastSeen) }]
          : []),
      ],
    };
  }

  if (type === "FIELD_ANOMALY") {
    const fieldLabels = {
      pressure: { sensor: "BMP280", title: "BMP280 pressure failure", unit: "hPa" },
      temperature: { sensor: "Sensor", title: "Temperature out of range", unit: "°C" },
      humidity: { sensor: "AHT10", title: "AHT10 humidity failure", unit: "%" },
      pm_all: { sensor: "PMS5003", title: "PMS5003 all channels zero", unit: "µg/m³" },
    };
    const info = fieldLabels[field] || { sensor: field, title: `${field} anomaly`, unit: "" };

    const actionMap = {
      pressure: `Check BMP280 I²C wiring on ${deviceId} (GPIO 21/22). If connections are intact, the sensor may need replacement. Temperature and humidity readings continue via AHT10, so PM2.5 data collection is not affected.`,
      temperature: `Verify ${deviceId} is not exposed to extreme heat sources. Check AHT10 sensor connections. If readings persist, the sensor may need recalibration or replacement.`,
      humidity: `Check AHT10 I²C wiring on ${deviceId}. If humidity reads 0 consistently, the sensor has likely failed and needs replacement.`,
      pm_all: `Check PMS5003 serial connection on ${deviceId}. Ensure the fan is spinning and the air intake is not blocked. Try power-cycling the sensor. If all PM channels remain at 0, the PMS5003 may need replacement.`,
    };

    return {
      severity,
      title: `${deviceId} — ${info.title}`,
      description: message,
      action: actionMap[field] || `Investigate ${field} readings on ${deviceId}.`,
      details: [
        { label: "Affected field", value: field },
        {
          label: "Current value",
          value: details?.currentValue ?? "anomalous",
          color: "#e24b4a",
          bold: true,
        },
        ...(details?.lastValid
          ? [{ label: "Last valid reading", value: details.lastValid }]
          : []),
        ...(details?.failureStarted
          ? [{ label: "Failure started", value: formatGMT5Short(details.failureStarted) }]
          : []),
      ],
    };
  }

  if (type === "LOW_COMPLETENESS") {
    return {
      severity,
      title: `${deviceId} — Low data completeness`,
      description: message,
      details: [
        { label: "Device", value: deviceId },
        { label: "Records received", value: `${details?.count ?? "?"} of ${COMPLETENESS_THRESHOLD}+ expected`, color: "#854f0b", bold: true },
        { label: "Time window", value: "Last 60 minutes" },
      ],
    };
  }

  return { severity, title: `${deviceId} — ${type}`, description: message, details: [] };
}

function formatGMT5Short(date) {
  return new Date(date).toLocaleString("en-US", {
    timeZone: "Asia/Almaty",
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
    hour12: true,
  });
}

export async function runHealthCheck() {
  const now = new Date();
  const newAlerts = [];
  const summaries = [];

  for (const deviceId of KNOWN_DEVICES) {
    // --- Check 1: SENSOR_OFFLINE ---
    const cutoff30 = new Date(now.getTime() - OFFLINE_THRESHOLD_MIN * 60 * 1000);
    const recentCount = await Measurement.countDocuments({
      deviceId,
      createdAt: { $gte: cutoff30 },
    });

    if (recentCount === 0) {
      const lastRecord = await Measurement.findOne({ deviceId })
        .sort({ createdAt: -1 })
        .lean();

      const dedupQuery = { type: "SENSOR_OFFLINE", deviceId };
      if (!(await hasUnresolvedAlert(dedupQuery))) {
        const alert = await new Alert({
          type: "SENSOR_OFFLINE",
          deviceId,
          severity: "critical",
          message: `No data received from ${deviceId} in the last ${OFFLINE_THRESHOLD_MIN} minutes.`,
          details: { lastSeen: lastRecord?.createdAt || null },
        }).save();
        newAlerts.push(alert);
        summaries.push(`${deviceId} SENSOR_OFFLINE`);
      }

      // Escalate: resolve any LOW_COMPLETENESS since SENSOR_OFFLINE is more severe
      await resolveAlerts({ type: "LOW_COMPLETENESS", deviceId });
      continue; // skip remaining checks for offline device
    } else {
      // Sensor is reporting: auto-resolve SENSOR_OFFLINE
      await resolveAlerts({ type: "SENSOR_OFFLINE", deviceId });
    }

    // --- Check 2: FIELD_ANOMALY (last 6 records) ---
    const lastRecords = await Measurement.find({ deviceId })
      .sort({ createdAt: -1 })
      .limit(ANOMALY_WINDOW)
      .lean();

    if (lastRecords.length >= ANOMALY_WINDOW) {
      // 2a: pressure === 0 (BMP280 failure)
      if (lastRecords.every((r) => r.pressure === 0)) {
        const dedupQuery = { type: "FIELD_ANOMALY", deviceId, field: "pressure" };
        if (!(await hasUnresolvedAlert(dedupQuery))) {
          const firstBad = lastRecords[lastRecords.length - 1];
          const alert = await new Alert({
            type: "FIELD_ANOMALY",
            deviceId,
            field: "pressure",
            severity: "critical",
            message: `Pressure readings dropped to 0 hPa for the last ${ANOMALY_WINDOW} consecutive readings. The BMP280 sensor has stopped responding.`,
            details: {
              currentValue: "0 hPa",
              failureStarted: firstBad.createdAt,
            },
          }).save();
          newAlerts.push(alert);
          summaries.push(`${deviceId} FIELD_ANOMALY (pressure)`);
        }
      } else {
        await resolveAlerts({ type: "FIELD_ANOMALY", deviceId, field: "pressure" });
      }

      // 2b: temperature out of range
      if (lastRecords.every((r) => r.temperature > 50 || r.temperature < -50)) {
        const dedupQuery = { type: "FIELD_ANOMALY", deviceId, field: "temperature" };
        if (!(await hasUnresolvedAlert(dedupQuery))) {
          const avg = (lastRecords.reduce((s, r) => s + r.temperature, 0) / lastRecords.length).toFixed(1);
          const alert = await new Alert({
            type: "FIELD_ANOMALY",
            deviceId,
            field: "temperature",
            severity: "warning",
            message: `Temperature readings are outside plausible range (avg: ${avg}°C over last ${ANOMALY_WINDOW} readings).`,
            details: { currentValue: `${avg} °C` },
          }).save();
          newAlerts.push(alert);
          summaries.push(`${deviceId} FIELD_ANOMALY (temperature)`);
        }
      } else {
        await resolveAlerts({ type: "FIELD_ANOMALY", deviceId, field: "temperature" });
      }

      // 2c: humidity === 0 or > 100 (AHT10 failure)
      if (lastRecords.every((r) => r.humidity === 0 || r.humidity > 100)) {
        const dedupQuery = { type: "FIELD_ANOMALY", deviceId, field: "humidity" };
        if (!(await hasUnresolvedAlert(dedupQuery))) {
          const sample = lastRecords[0].humidity;
          const alert = await new Alert({
            type: "FIELD_ANOMALY",
            deviceId,
            field: "humidity",
            severity: "critical",
            message: `Humidity readings are invalid (${sample}%) for the last ${ANOMALY_WINDOW} consecutive readings. The AHT10 sensor may have failed.`,
            details: { currentValue: `${sample} %` },
          }).save();
          newAlerts.push(alert);
          summaries.push(`${deviceId} FIELD_ANOMALY (humidity)`);
        }
      } else {
        await resolveAlerts({ type: "FIELD_ANOMALY", deviceId, field: "humidity" });
      }

      // 2d: all PM channels === 0 (PMS5003 stuck)
      if (lastRecords.every((r) => r.pm1_raw === 0 && r.pm25_raw === 0 && r.pm10_raw === 0)) {
        const dedupQuery = { type: "FIELD_ANOMALY", deviceId, field: "pm_all" };
        if (!(await hasUnresolvedAlert(dedupQuery))) {
          const alert = await new Alert({
            type: "FIELD_ANOMALY",
            deviceId,
            field: "pm_all",
            severity: "warning",
            message: `All PM channels (PM1, PM2.5, PM10) read 0 µg/m³ for the last ${ANOMALY_WINDOW} consecutive readings. The PMS5003 may be stuck or disconnected.`,
            details: { currentValue: "0 µg/m³ (all channels)" },
          }).save();
          newAlerts.push(alert);
          summaries.push(`${deviceId} FIELD_ANOMALY (pm_all)`);
        }
      } else {
        await resolveAlerts({ type: "FIELD_ANOMALY", deviceId, field: "pm_all" });
      }
    }

    // --- Check 3: LOW_COMPLETENESS ---
    const cutoff60 = new Date(now.getTime() - 60 * 60 * 1000);
    const hourCount = await Measurement.countDocuments({
      deviceId,
      createdAt: { $gte: cutoff60 },
    });

    if (hourCount > 0 && hourCount < COMPLETENESS_THRESHOLD) {
      const dedupQuery = { type: "LOW_COMPLETENESS", deviceId };
      if (!(await hasUnresolvedAlert(dedupQuery))) {
        const alert = await new Alert({
          type: "LOW_COMPLETENESS",
          deviceId,
          severity: "warning",
          message: `Only ${hourCount} of expected ${COMPLETENESS_THRESHOLD}+ records received in the last hour.`,
          details: { count: hourCount },
        }).save();
        newAlerts.push(alert);
        summaries.push(`${deviceId} LOW_COMPLETENESS (${hourCount} records)`);
      }
    } else if (hourCount >= COMPLETENESS_THRESHOLD) {
      await resolveAlerts({ type: "LOW_COMPLETENESS", deviceId });
    }
  }

  // --- Send email if new alerts found ---
  if (newAlerts.length > 0) {
    const emailAlerts = newAlerts.map(buildEmailAlert);
    await sendAlertEmail(emailAlerts, now);

    // Update notifiedAt on all new alert docs
    await Alert.updateMany(
      { _id: { $in: newAlerts.map((a) => a._id) } },
      { $set: { notifiedAt: now } }
    );

    console.log(
      `[${now.toISOString()}] Health check: ${newAlerts.length} alert(s) — ${summaries.join(", ")}`
    );
  } else {
    console.log(`[${now.toISOString()}] Health check: all ${KNOWN_DEVICES.length} sensors OK`);
  }

  return { timestamp: now, newAlerts: newAlerts.length, alerts: summaries };
}

export function startHealthCheckCron() {
  cron.schedule("30 * * * *", async () => {
    console.log(`[${new Date().toISOString()}] Running sensor health check...`);
    try {
      await runHealthCheck();
    } catch (error) {
      console.error("[ERR] Health check failed:", error.message);
    }
  });

  console.log("[INFO] Sensor health check cron job scheduled (hourly, :30)");
}
