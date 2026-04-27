import mongoose from "mongoose";

const forecastSchema = new mongoose.Schema(
  {
    created_at: { type: Date, default: Date.now },
    station_id: { type: String, required: true },
    model_type: { type: String },
    predictions: [
      {
        day: { type: Number },
        date: { type: Date },
        pm25: { type: Number },
        rmse: { type: Number },
      },
    ],
    input_days_used: { type: Number },
    padded: { type: Boolean, default: false },
    horizon_days: { type: Number },
    metadata: {
      training_samples: { type: Number },
      lookback_window: { type: Number },
      model_version: { type: String },
    },
  },
  { timestamps: true }
);

forecastSchema.index({ station_id: 1, created_at: -1 });

export const Forecast = mongoose.model("Forecast", forecastSchema);
