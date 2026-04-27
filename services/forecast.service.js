import cron from "node-cron";
import { spawn } from "child_process";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const SCRIPT_PATH = path.join(__dirname, "..", "ml", "production", "forecast.py");

function runForecast() {
  return new Promise((resolve, reject) => {
    console.log("[INFO] Starting forecast generation...");

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
        console.log("[FORECAST]", stderr.trim());
      }

      if (code !== 0) {
        console.error("[ERR] Forecast process exited with code", code);
        return reject(new Error(`Forecast failed with code ${code}`));
      }

      try {
        const result = JSON.parse(stdout.trim().split("\n").pop());
        console.log("[INFO] Forecast result:", result);
        resolve(result);
      } catch (e) {
        console.log("[INFO] Forecast output:", stdout.trim());
        resolve({ status: "done", raw: stdout.trim() });
      }
    });
  });
}

export function startForecastCron() {
  // Daily at 4am
  cron.schedule("0 4 * * *", async () => {
    console.log("[INFO] Running daily forecast generation...");
    try {
      await runForecast();
    } catch (err) {
      console.error("[ERR] Daily forecast failed:", err.message);
    }
  });

  console.log("[INFO] Forecast cron job scheduled (daily, 4am)");
}
