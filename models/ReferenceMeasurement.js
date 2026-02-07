import mongoose from "mongoose";

const referenceMeasurementSchema = new mongoose.Schema(
  {
    pm1_raw: { type: Number },
    pm25_raw: { type: Number },
    pm10_raw: { type: Number },

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

    deviceId: { type: String, default: "Reference-Station" },
  },
  { timestamps: true }
);

export const ReferenceMeasurement = mongoose.model("ReferenceMeasurement", referenceMeasurementSchema);
