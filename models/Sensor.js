import mongoose from "mongoose";

const sensorSchema = new mongoose.Schema(
  {
    sensor_id: { type: String, unique: true, required: true },
    name: { type: String },
    is_calibration_sensor: { type: Boolean, default: false },
    nearest_reference_station: { type: String },
    reference_station_distance_m: { type: Number },
    install_date: { type: Date },
    location: {
      latitude: { type: Number },
      longitude: { type: Number },
    },
    status: {
      type: String,
      enum: ["active", "degraded", "offline"],
      default: "active",
    },
  },
  { timestamps: true }
);

export const Sensor = mongoose.model("Sensor", sensorSchema);
