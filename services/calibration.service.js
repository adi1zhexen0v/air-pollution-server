import { spawn } from "child_process";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const SCRIPT_PATH = path.join(__dirname, "..", "ml", "production", "calibrate.py");

export function calibrateMeasurement(measurementId) {
  return new Promise((resolve, reject) => {
    const proc = spawn("python", [SCRIPT_PATH, "--measurement_id", measurementId], {
      env: { ...process.env },
    });

    let stdout = "";
    let stderr = "";

    proc.stdout.on("data", (data) => {
      stdout += data.toString();
    });

    proc.stderr.on("data", (data) => {
      stderr += data.toString();
    });

    proc.on("close", (code) => {
      if (stderr) {
        console.log("[CALIBRATION]", stderr.trim());
      }

      if (code !== 0) {
        console.error("[ERR] Calibration process exited with code", code);
        return reject(new Error(`Calibration failed with code ${code}`));
      }

      try {
        const result = JSON.parse(stdout.trim().split("\n").pop());
        console.log("[INFO] Calibration result:", result);
        resolve(result);
      } catch (e) {
        console.error("[ERR] Failed to parse calibration output:", stdout);
        reject(e);
      }
    });
  });
}
