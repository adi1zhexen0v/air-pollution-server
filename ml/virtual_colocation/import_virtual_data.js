/**
 * Import virtual co-location data into MongoDB.
 *
 * Reads the Python pipeline output and inserts virtual sensor records
 * into the referencemeasurements collection with deviceId "Sensor-4".
 * Also saves calibration results into a calibrationresults collection.
 *
 * Usage: node import_virtual_data.js
 */

import mongoose from "mongoose";
import "dotenv/config";
import { readFileSync } from "fs";
import { fileURLToPath } from "url";
import path from "path";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// ── Schemas ─────────────────────────────────────────────────────────────────

const referenceMeasurementSchema = new mongoose.Schema(
  {
    pm1_raw: { type: Number },
    pm25_raw: { type: Number },
    pm10_raw: { type: Number },
    pm1_calibrated: { type: Number },
    pm25_calibrated: { type: Number },
    pm10_calibrated: { type: Number },
    pm25_aqi: { type: Number },
    pm25_ugm3: { type: Number },
    temperature: { type: Number },
    pressure: { type: Number },
    humidity: { type: Number },
    heat_index: { type: Number },
    latitude: { type: Number },
    longitude: { type: Number },
    satellites: { type: Number },
    deviceId: { type: String, default: "Sensor-4" },
  },
  { timestamps: true }
);

referenceMeasurementSchema.index({ createdAt: -1 });

const calibrationResultSchema = new mongoose.Schema({
  run_date: { type: Date, default: Date.now },
  method: { type: String, default: "virtual_colocation_IDW" },
  dataset_size: Number,
  models: [
    {
      name: String,
      cv_r2_mean: Number,
      cv_r2_std: Number,
      cv_rmse_mean: Number,
      cv_mae_mean: Number,
      final_r2: Number,
      final_rmse: Number,
      final_mae: Number,
      training_time_sec: Number,
    },
  ],
  best_model: String,
  dataset_info: mongoose.Schema.Types.Mixed,
  metadata: mongoose.Schema.Types.Mixed,
});

// ── Main ────────────────────────────────────────────────────────────────────

async function main() {
  const uri = process.env.MONGODB_URI || process.env.DB_URL;
  if (!uri) {
    console.error("[ERR] Set MONGODB_URI or DB_URL in .env");
    process.exit(1);
  }

  // Read output files
  const outputDir = path.join(__dirname, "output");

  let virtualData, calibrationResults;
  try {
    virtualData = JSON.parse(
      readFileSync(path.join(outputDir, "virtual_colocation_data.json"), "utf-8")
    );
    console.log(
      `[INFO] Read virtual_colocation_data.json: ${virtualData.data.length} records`
    );
  } catch (err) {
    console.error("[ERR] Failed to read virtual_colocation_data.json:", err.message);
    process.exit(1);
  }

  try {
    calibrationResults = JSON.parse(
      readFileSync(path.join(outputDir, "calibration_results.json"), "utf-8")
    );
    console.log(
      `[INFO] Read calibration_results.json: ${calibrationResults.models.length} models`
    );
  } catch (err) {
    console.error("[ERR] Failed to read calibration_results.json:", err.message);
    process.exit(1);
  }

  // Connect to MongoDB
  try {
    await mongoose.connect(uri);
    console.log("[INFO] Connected to MongoDB");
  } catch (err) {
    console.error("[ERR] MongoDB connection failed:", err.message);
    process.exit(1);
  }

  const ReferenceMeasurement = mongoose.model(
    "ReferenceMeasurement",
    referenceMeasurementSchema
  );
  const CalibrationResult = mongoose.model(
    "CalibrationResult",
    calibrationResultSchema
  );

  try {
    // Delete existing Sensor-4 records
    const deleteResult = await ReferenceMeasurement.deleteMany({
      deviceId: "Sensor-4",
    });
    console.log(
      `[INFO] Deleted ${deleteResult.deletedCount} existing Sensor-4 records`
    );

    // Prepare documents with explicit createdAt timestamps
    const docs = virtualData.data.map((record) => ({
      ...record,
      createdAt: new Date(record.createdAt),
      updatedAt: new Date(),
    }));

    // Insert virtual sensor records
    // Use insertMany with timestamps:false so Mongoose doesn't overwrite createdAt
    const insertResult = await ReferenceMeasurement.insertMany(docs, {
      timestamps: false,
    });
    console.log(
      `[INFO] Inserted ${insertResult.length} Sensor-4 records into referencemeasurements`
    );

    // Save calibration results
    const calDoc = new CalibrationResult({
      run_date: new Date(calibrationResults.run_date),
      method: calibrationResults.method,
      dataset_size: calibrationResults.dataset_size,
      models: calibrationResults.models,
      best_model: calibrationResults.best_model,
      dataset_info: calibrationResults.dataset_info,
      metadata: virtualData.metadata,
    });
    await calDoc.save();
    console.log("[INFO] Saved calibration results to calibrationresults collection");

    // Summary
    const totalSensor4 = await ReferenceMeasurement.countDocuments({
      deviceId: "Sensor-4",
    });
    const totalRef = await ReferenceMeasurement.countDocuments({
      deviceId: "Reference-Station",
    });
    console.log("\n" + "=".repeat(60));
    console.log("IMPORT SUMMARY");
    console.log("=".repeat(60));
    console.log(`  Collection: referencemeasurements`);
    console.log(`  Sensor-4 records:          ${totalSensor4}`);
    console.log(`  Reference-Station records: ${totalRef}`);
    console.log(`  Best model:                ${calibrationResults.best_model}`);
    console.log(`  Observation period:        ${virtualData.metadata.observation_period.start} → ${virtualData.metadata.observation_period.end}`);
    console.log("=".repeat(60));
  } catch (err) {
    console.error("[ERR] Import failed:", err.message);
  } finally {
    await mongoose.disconnect();
    console.log("[INFO] Disconnected from MongoDB");
  }
}

main();
