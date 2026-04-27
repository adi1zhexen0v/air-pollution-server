import { Measurement } from "../models/Measurement.js";

class MeasurementController {
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

      const newMeasurement = new Measurement({
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

      await newMeasurement.save();

      res.status(201).json({
        message: "New measurement stored successfully",
        measurement: newMeasurement,
      });
    } catch (error) {
      console.error("[ERR] Failed to save measurement:", error);
      res.status(500).json({ error: error.message });
    }
  }

  async getAll(req, res) {
    try {
      const measurements = await Measurement.find().sort({ createdAt: -1 }).limit(1000);
      if (!measurements.length) {
        return res.status(404).json({ error: "Measurements not found" });
      }

      res.json(measurements);
    } catch (error) {
      console.error("[ERR] Failed to fetch measurements:", error);
      res.status(500).json({ error: error.message });
    }
  }

  async getLatest(req, res) {
    try {
      const latest = await Measurement.aggregate([
        { $sort: { createdAt: -1 } },
        {
          $group: {
            _id: "$deviceId",
            doc: { $first: "$$ROOT" },
          },
        },
        { $replaceRoot: { newRoot: "$doc" } },
      ]);

      res.json(latest);
    } catch (error) {
      console.error("[ERR] Failed to fetch latest measurements:", error);
      res.status(500).json({ error: error.message });
    }
  }

  async getHistory(req, res) {
    try {
      const { deviceId, period = "7d", from, to } = req.query;
      if (!deviceId) {
        return res.status(400).json({ error: "deviceId is required" });
      }

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

      const spanDays = (dateTo - dateFrom) / (24 * 60 * 60 * 1000);
      const needsDownsample = spanDays > 7;

      if (needsDownsample) {
        const results = await Measurement.aggregate([
          {
            $match: {
              deviceId,
              createdAt: { $gte: dateFrom, $lte: dateTo },
            },
          },
          {
            $group: {
              _id: {
                $dateTrunc: { date: "$createdAt", unit: "hour" },
              },
              pm25_raw: { $avg: "$pm25_raw" },
              pm10_raw: { $avg: "$pm10_raw" },
              pm1_raw: { $avg: "$pm1_raw" },
              pm25_calibrated: { $avg: "$pm25_calibrated" },
              pm10_calibrated: { $avg: "$pm10_calibrated" },
              pm1_calibrated: { $avg: "$pm1_calibrated" },
              temperature: { $avg: "$temperature" },
              humidity: { $avg: "$humidity" },
              count: { $sum: 1 },
            },
          },
          { $sort: { _id: 1 } },
          {
            $project: {
              _id: 0,
              createdAt: "$_id",
              pm25_raw: { $round: ["$pm25_raw", 2] },
              pm10_raw: { $round: ["$pm10_raw", 2] },
              pm1_raw: { $round: ["$pm1_raw", 2] },
              pm25_calibrated: { $round: ["$pm25_calibrated", 2] },
              pm10_calibrated: { $round: ["$pm10_calibrated", 2] },
              pm1_calibrated: { $round: ["$pm1_calibrated", 2] },
              temperature: { $round: ["$temperature", 2] },
              humidity: { $round: ["$humidity", 2] },
              count: 1,
            },
          },
        ]);

        return res.json(results);
      }

      const measurements = await Measurement.find({
        deviceId,
        createdAt: { $gte: dateFrom, $lte: dateTo },
      }).sort({ createdAt: 1 });

      res.json(measurements);
    } catch (error) {
      console.error("[ERR] Failed to fetch measurement history:", error);
      res.status(500).json({ error: error.message });
    }
  }

  async getHourly(req, res) {
    try {
      const { deviceId, from, to } = req.query;
      if (!from || !to) {
        return res.status(400).json({ error: "from and to are required" });
      }

      const dateFrom = new Date(from);
      const dateTo = new Date(to);

      const matchStage = {
        createdAt: { $gte: dateFrom, $lte: dateTo },
      };
      if (deviceId) {
        matchStage.deviceId = deviceId;
      }

      const groupId = deviceId
        ? { hour: { $dateTrunc: { date: "$createdAt", unit: "hour" } } }
        : {
            hour: { $dateTrunc: { date: "$createdAt", unit: "hour" } },
            deviceId: "$deviceId",
          };

      const results = await Measurement.aggregate([
        { $match: matchStage },
        {
          $group: {
            _id: groupId,
            pm25_raw: { $avg: "$pm25_raw" },
            pm10_raw: { $avg: "$pm10_raw" },
            pm1_raw: { $avg: "$pm1_raw" },
            pm25_calibrated: { $avg: "$pm25_calibrated" },
            pm10_calibrated: { $avg: "$pm10_calibrated" },
            pm1_calibrated: { $avg: "$pm1_calibrated" },
            temperature: { $avg: "$temperature" },
            humidity: { $avg: "$humidity" },
            count: { $sum: 1 },
          },
        },
        { $sort: { "_id.hour": 1 } },
        {
          $project: {
            _id: 0,
            hour: "$_id.hour",
            deviceId: "$_id.deviceId",
            pm25_raw: { $round: ["$pm25_raw", 2] },
            pm10_raw: { $round: ["$pm10_raw", 2] },
            pm1_raw: { $round: ["$pm1_raw", 2] },
            pm25_calibrated: { $round: ["$pm25_calibrated", 2] },
            pm10_calibrated: { $round: ["$pm10_calibrated", 2] },
            pm1_calibrated: { $round: ["$pm1_calibrated", 2] },
            temperature: { $round: ["$temperature", 2] },
            humidity: { $round: ["$humidity", 2] },
            count: 1,
          },
        },
      ]);

      res.json(results);
    } catch (error) {
      console.error("[ERR] Failed to fetch hourly measurements:", error);
      res.status(500).json({ error: error.message });
    }
  }

  async getDevices(req, res) {
    try {
      const devices = await Measurement.aggregate([
        {
          $group: {
            _id: "$deviceId",
            latitude: { $first: "$latitude" },
            longitude: { $first: "$longitude" },
            last_seen: { $max: "$createdAt" },
            total_measurements: { $sum: 1 },
          },
        },
        {
          $project: {
            _id: 0,
            deviceId: "$_id",
            latitude: 1,
            longitude: 1,
            last_seen: 1,
            total_measurements: 1,
            is_online: {
              $gt: [
                "$last_seen",
                new Date(Date.now() - 30 * 60 * 1000),
              ],
            },
          },
        },
      ]);

      res.json(devices);
    } catch (error) {
      console.error("[ERR] Failed to fetch devices:", error);
      res.status(500).json({ error: error.message });
    }
  }
}

export default new MeasurementController();
