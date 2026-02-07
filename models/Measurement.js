import mongoose from "mongoose";

const measurementSchema = new mongoose.Schema(
  {
    pm1_raw: { type: Number, required: true },
    pm25_raw: { type: Number, required: true },
    pm10_raw: { type: Number, required: true },

    pm1_calibrated: { type: Number },
    pm25_calibrated: { type: Number },
    pm10_calibrated: { type: Number },

    temperature: { type: Number },
    pressure: { type: Number },
    humidity: { type: Number },
    heat_index: { type: Number },

    latitude: { type: Number },
    longitude: { type: Number },
    satellites: { type: Number },

    deviceId: { type: String, default: "ESP32-Unit1" },
  },
  { timestamps: true }
);

export const Measurement = mongoose.model("Measurement", measurementSchema);
