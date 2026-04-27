import { ReferenceMeasurement } from "../models/ReferenceMeasurement.js";

// EPA PM2.5 AQI breakpoint table: [AQI_lo, AQI_hi, conc_lo, conc_hi]
const PM25_BREAKPOINTS = [
  [0, 50, 0.0, 12.0],
  [51, 100, 12.1, 35.4],
  [101, 150, 35.5, 55.4],
  [151, 200, 55.5, 150.4],
  [201, 300, 150.5, 250.4],
  [301, 400, 250.5, 350.4],
  [401, 500, 350.5, 500.4],
];

function aqiToUgm3Pm25(aqi) {
  if (aqi == null || isNaN(aqi)) return null;
  if (aqi < 0) return 0;
  aqi = Math.round(aqi);
  for (const [aqiLo, aqiHi, concLo, concHi] of PM25_BREAKPOINTS) {
    if (aqi >= aqiLo && aqi <= aqiHi) {
      return (concHi - concLo) / (aqiHi - aqiLo) * (aqi - aqiLo) + concLo;
    }
  }
  const [aqiLo, aqiHi, concLo, concHi] = PM25_BREAKPOINTS[PM25_BREAKPOINTS.length - 1];
  return (concHi - concLo) / (aqiHi - aqiLo) * (aqi - aqiLo) + concLo;
}

function ensureUgm3(doc) {
  const obj = doc.toObject ? doc.toObject() : { ...doc };
  if (obj.pm25_ugm3 == null && obj.pm25_raw != null) {
    obj.pm25_ugm3 = aqiToUgm3Pm25(obj.pm25_raw);
  }
  return obj;
}

class ReferenceMeasurementController {
  async create(req, res) {
    try {
      const {
        pm1_raw,
        pm25_raw,
        pm10_raw,
        temperature,
        pressure,
        heat_index,
        humidity,
        latitude,
        longitude,
        satellites,
        deviceId,
      } = req.body;

      const newReferenceMeasurement = new ReferenceMeasurement({
        pm1_raw,
        pm25_raw,
        pm10_raw,
        heat_index,
        temperature,
        pressure,
        humidity,
        latitude,
        longitude,
        satellites,
        deviceId,
      });

      await newReferenceMeasurement.save();

      res.status(201).json({
        message: "New reference measurement stored successfully",
        measurement: newReferenceMeasurement,
      });
    } catch (error) {
      console.error("[ERR] Failed to save reference measurement:", error);
      res.status(500).json({ error: error.message });
    }
  }

  async getAll(req, res) {
    try {
      const measurements = await ReferenceMeasurement.find().sort({ createdAt: -1 }).limit(1000);
      if (!measurements.length) {
        return res.status(404).json({ error: "Reference measurements not found" });
      }

      res.json(measurements.map(ensureUgm3));
    } catch (error) {
      console.error("[ERR] Failed to fetch reference measurements:", error);
      res.status(500).json({ error: error.message });
    }
  }

  async getHistory(req, res) {
    try {
      const { period = "7d", from, to } = req.query;

      let dateFrom, dateTo;
      dateTo = new Date();
      if (from && to) {
        dateFrom = new Date(from);
        dateTo = new Date(to);
      } else if (period === "30d" || period === "all") {
        dateFrom = new Date(0);
      } else {
        const periodMap = { "24h": 1, "3d": 3, "7d": 7 };
        const days = periodMap[period] || 7;
        dateFrom = new Date(dateTo.getTime() - days * 24 * 60 * 60 * 1000);
      }

      const measurements = await ReferenceMeasurement.find({
        createdAt: { $gte: dateFrom, $lte: dateTo },
      }).sort({ createdAt: 1 });

      res.json(measurements.map(ensureUgm3));
    } catch (error) {
      console.error("[ERR] Failed to fetch reference measurement history:", error);
      res.status(500).json({ error: error.message });
    }
  }
}

export default new ReferenceMeasurementController();
