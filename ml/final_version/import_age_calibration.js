/**
 * Import age-based calibration models into MongoDB.
 *
 * Reads outputs/age_calibration_models.json and inserts 4 weekly
 * CalibrationModel documents into the calibrationmodels collection.
 *
 * Usage: node import_age_calibration.js
 */

import mongoose from "mongoose";
import "dotenv/config";
import { readFileSync } from "fs";
import { fileURLToPath } from "url";
import path from "path";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const JSON_PATH = path.join(__dirname, "outputs", "age_calibration_models.json");

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
    week: { type: Number },
    period_start: { type: String },
    period_end: { type: String },
  },
  { timestamps: true }
);

async function main() {
  const uri = process.env.DB_URL;
  if (!uri) {
    console.error("[ERR] Set DB_URL in .env");
    process.exit(1);
  }

  let docs;
  try {
    docs = JSON.parse(readFileSync(JSON_PATH, "utf-8"));
    console.log(`[INFO] Read ${docs.length} age-based models from JSON`);
  } catch (err) {
    console.error("[ERR] Failed to read JSON:", err.message);
    process.exit(1);
  }

  await mongoose.connect(uri);
  console.log("[INFO] Connected to MongoDB");

  const CalibrationModel = mongoose.model("CalibrationModel", calibrationModelSchema);

  // Deactivate existing age-based models for Sensor-4
  const deactivated = await CalibrationModel.updateMany(
    { sensor_id: "Sensor-4", is_active: true },
    { $set: { is_active: false } }
  );
  console.log(`[INFO] Deactivated ${deactivated.modifiedCount} existing Sensor-4 models`);

  for (const doc of docs) {
    await CalibrationModel.create({
      ...doc,
      trained_at: new Date(doc.trained_at),
    });
    console.log(`[INFO] Inserted Week ${doc.week} model (R2=${doc.metrics.r2.toFixed(4)})`);
  }

  console.log(`[INFO] Done. Inserted ${docs.length} age-based calibration models.`);
  await mongoose.disconnect();
}

main().catch((err) => {
  console.error("[ERR]", err);
  process.exit(1);
});
