import mongoose from "mongoose";

const calibrationModelSchema = new mongoose.Schema(
  {
    sensor_id: { type: String, required: true },
    trained_at: { type: Date, default: Date.now },
    model_type: { type: String, default: "MLR" },
    coefficients: { type: mongoose.Schema.Types.Mixed },
    metrics: {
      r2: { type: Number },
      rmse: { type: Number },
      mae: { type: Number },
    },
    sensor_age_days: { type: Number },
    training_samples: { type: Number },
    is_active: { type: Boolean, default: true },
    model_path: { type: String },
  },
  { timestamps: true }
);

export const CalibrationModel = mongoose.model("CalibrationModel", calibrationModelSchema);
