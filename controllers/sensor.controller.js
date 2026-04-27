import { Sensor } from "../models/Sensor.js";
import { CalibrationModel } from "../models/CalibrationModel.js";

class SensorController {
  async getAll(req, res) {
    try {
      const sensors = await Sensor.find().sort({ createdAt: -1 });
      if (!sensors.length) {
        return res.status(404).json({ error: "No sensors found" });
      }

      res.json(sensors);
    } catch (error) {
      console.error("[ERR] Failed to fetch sensors:", error);
      res.status(500).json({ error: error.message });
    }
  }

  async getActiveModel(req, res) {
    try {
      const { sensor_id } = req.params;
      const model = await CalibrationModel.findOne({
        sensor_id,
        is_active: true,
      }).sort({ trained_at: -1 });

      if (!model) {
        return res.status(404).json({ error: "No active model for this sensor" });
      }

      res.json(model);
    } catch (error) {
      console.error("[ERR] Failed to fetch calibration model:", error);
      res.status(500).json({ error: error.message });
    }
  }
}

export default new SensorController();
