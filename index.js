import express from "express";
import cors from "cors";
import { connectDatabase } from "./config/db.js";
import MeasurementController from "./controllers/measurement.controller.js";
import ReferenceMeasurementController from "./controllers/referenceMeasurement.controller.js";
import ForecastController from "./controllers/forecast.controller.js";
import SensorController from "./controllers/sensor.controller.js";
import { startReferenceMeasurementCron } from "./services/referenceMeasurement.service.js";
import { startRetrainCron } from "./services/retrain.service.js";
import { startForecastCron } from "./services/forecast.service.js";
import { startHealthCheckCron, runHealthCheck } from "./services/sensorHealthCheck.js";
import { Alert } from "./models/Alert.js";

const app = express();
const PORT = 4000;

app.use(express.json());
app.use(cors());

app.get("/test", (req, res) => {
  res.send("GET Request is working");
});
app.get("/measurement", MeasurementController.getAll);
app.get("/measurement/latest", MeasurementController.getLatest);
app.get("/measurement/history", MeasurementController.getHistory);
app.get("/measurement/hourly", MeasurementController.getHourly);
app.get("/measurement/devices", MeasurementController.getDevices);
app.post("/measurement/create", MeasurementController.create);
app.get("/reference-measurement", ReferenceMeasurementController.getAll);
app.get("/reference-measurement/history", ReferenceMeasurementController.getHistory);
app.post("/reference-measurement/create", ReferenceMeasurementController.create);
app.get("/forecast", ForecastController.getLatest);
app.post("/forecast/create", ForecastController.create);
app.get("/sensor", SensorController.getAll);
app.get("/calibration-model/:sensor_id", SensorController.getActiveModel);

app.get("/api/health-check/run", async (req, res) => {
  try {
    const result = await runHealthCheck();
    res.json(result);
  } catch (error) {
    console.error("[ERR] Manual health check failed:", error);
    res.status(500).json({ error: error.message });
  }
});

app.get("/api/health-check/history", async (req, res) => {
  try {
    const alerts = await Alert.find().sort({ createdAt: -1 }).limit(50);
    res.json({ alerts });
  } catch (error) {
    console.error("[ERR] Failed to fetch alert history:", error);
    res.status(500).json({ error: error.message });
  }
});

app.listen(PORT, async () => {
  await connectDatabase();
  startReferenceMeasurementCron();
  startRetrainCron();
  startForecastCron();
  startHealthCheckCron();
  console.log("Air quality collection server is running...");
});
