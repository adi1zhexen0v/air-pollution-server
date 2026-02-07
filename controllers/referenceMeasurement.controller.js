import { ReferenceMeasurement } from "../models/ReferenceMeasurement.js";

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
      const measurements = await ReferenceMeasurement.find().sort({ createdAt: -1 });
      if (!measurements.length) {
        return res.status(404).json({ error: "Reference measurements not found" });
      }

      res.json(measurements);
    } catch (error) {
      console.error("[ERR] Failed to fetch reference measurements:", error);
      res.status(500).json({ error: error.message });
    }
  }
}

export default new ReferenceMeasurementController();
