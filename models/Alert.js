import mongoose from "mongoose";

const alertSchema = new mongoose.Schema(
  {
    type: {
      type: String,
      required: true,
      enum: ["SENSOR_OFFLINE", "FIELD_ANOMALY", "LOW_COMPLETENESS"],
    },
    deviceId: { type: String, required: true },
    field: { type: String },
    severity: {
      type: String,
      required: true,
      enum: ["warning", "critical"],
    },
    message: { type: String },
    details: { type: mongoose.Schema.Types.Mixed },
    resolved: { type: Boolean, default: false },
    resolvedAt: { type: Date },
    notifiedAt: { type: Date, default: Date.now },
  },
  { timestamps: true }
);

alertSchema.index({ type: 1, deviceId: 1, resolved: 1 });

export const Alert = mongoose.model("Alert", alertSchema);
