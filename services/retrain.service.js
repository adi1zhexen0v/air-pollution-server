import cron from "node-cron";
import { spawn } from "child_process";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const SCRIPT_PATH = path.join(__dirname, "..", "ml", "production", "retrain_forecast.py");

function runRetrain() {
  return new Promise((resolve, reject) => {
    console.log("[INFO] Starting forecast model retraining...");

    const proc = spawn("python", [SCRIPT_PATH], {
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
        console.log("[RETRAIN]", stderr.trim());
      }

      if (code !== 0) {
        console.error("[ERR] Retrain process exited with code", code);
        return reject(new Error(`Retrain failed with code ${code}`));
      }

      try {
        const result = JSON.parse(stdout.trim().split("\n").pop());
        console.log("[INFO] Retrain result:", result);
        resolve(result);
      } catch (e) {
        console.log("[INFO] Retrain output:", stdout.trim());
        resolve({ status: "done", raw: stdout.trim() });
      }
    });
  });
}

export function startRetrainCron() {
  // Weekly: Monday at 3am
  cron.schedule("0 3 * * 1", async () => {
    console.log("[INFO] Running weekly model retraining...");
    try {
      await runRetrain();
    } catch (err) {
      console.error("[ERR] Weekly retrain failed:", err.message);
    }
  });

  console.log("[INFO] Retrain cron job scheduled (weekly, Monday 3am)");
}
