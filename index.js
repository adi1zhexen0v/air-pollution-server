import express from "express";
import cors from "cors";
import { connectDatabase } from "./config/db.js";
import MeasurementController from "./controllers/measurement.controller.js";
import ReferenceMeasurementController from "./controllers/referenceMeasurement.controller.js";
import { startReferenceMeasurementCron } from "./services/referenceMeasurement.service.js";

const app = express();
const PORT = 4000;

app.use(express.json());
app.use(cors());

app.get("/test", (req, res) => {
  res.send("GET Request is working");
});
app.get("/measurement", MeasurementController.getAll);
app.post("/measurement/create", MeasurementController.create);
app.get("/reference-measurement", ReferenceMeasurementController.getAll);
app.post("/reference-measurement/create", ReferenceMeasurementController.create);

app.listen(PORT, async () => {
  await connectDatabase();
  startReferenceMeasurementCron();
  console.log("Air quality collection server is running...");
});
