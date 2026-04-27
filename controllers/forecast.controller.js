import { Forecast } from "../models/Forecast.js";

class ForecastController {
  async getLatest(req, res) {
    try {
      const { station_id } = req.query;

      if (station_id && station_id !== "all") {
        const forecast = await Forecast.findOne({ station_id }).sort({ created_at: -1 });
        if (!forecast) return res.status(404).json({ error: "No forecast available" });
        return res.json(forecast);
      }

      const forecasts = await Forecast.aggregate([
        { $sort: { created_at: -1 } },
        { $group: { _id: "$station_id", doc: { $first: "$$ROOT" } } },
        { $replaceRoot: { newRoot: "$doc" } },
      ]);

      if (!forecasts.length) return res.status(404).json({ error: "No forecasts available" });
      res.json(forecasts);
    } catch (error) {
      console.error("[ERR] Failed to fetch forecast:", error);
      res.status(500).json({ error: error.message });
    }
  }

  async create(req, res) {
    try {
      const { station_id, model_type, predictions, input_days_used, padded, horizon_days, metadata } = req.body;

      if (!station_id || !predictions || !Array.isArray(predictions)) {
        return res.status(400).json({ error: "station_id and predictions array are required" });
      }

      const forecast = new Forecast({
        station_id,
        model_type,
        predictions,
        input_days_used,
        padded,
        horizon_days,
        metadata,
      });

      await forecast.save();

      res.status(201).json({
        message: "Forecast created successfully",
        forecast,
      });
    } catch (error) {
      console.error("[ERR] Failed to create forecast:", error);
      res.status(500).json({ error: error.message });
    }
  }
}

export default new ForecastController();
