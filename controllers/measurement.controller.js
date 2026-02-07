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
      const measurements = await Measurement.find().sort({ createdAt: -1 });
      if (!measurements.length) {
        return res.status(404).json({ error: "Measurements not found" });
      }

      res.json(measurements);
    } catch (error) {
      console.error("[ERR] Failed to fetch measurements:", error);
      res.status(500).json({ error: error.message });
    }
  }
}

export default new MeasurementController();
