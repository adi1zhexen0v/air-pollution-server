import cron from "node-cron";
import "dotenv/config";
import { ReferenceMeasurement } from "../models/ReferenceMeasurement.js";

const AQICN_URL = "https://api.waqi.info/feed/A531799/";
const OPENWEATHER_URL = "https://api.openweathermap.org/data/2.5/weather";
const LATITUDE = 51.158041944444;
const LONGITUDE = 71.415435;

async function fetchAQICNData() {
  try {
    const response = await fetch(`${AQICN_URL}?token=${process.env.AQICN_TOKEN}`);
    const data = await response.json();

    if (data.status !== "ok") {
      throw new Error(`AQICN API error: ${data.data || "Unknown error"}`);
    }

    const iaqi = data.data.iaqi || {};
    return {
      pm1_raw: iaqi.pm1?.v || null,
      pm25_raw: iaqi.pm25?.v || null,
      pm10_raw: iaqi.pm10?.v || null,
    };
  } catch (error) {
    console.error("[ERR] Failed to fetch AQICN data:", error.message);
    throw error;
  }
}

async function fetchOpenWeatherData() {
  try {
    const url = `${OPENWEATHER_URL}?lat=${LATITUDE}&lon=${LONGITUDE}&appid=${process.env.OPENWEATHER_TOKEN}&units=metric`;
    const response = await fetch(url);
    const data = await response.json();

    if (data.cod !== 200) {
      throw new Error(`OpenWeather API error: ${data.message || "Unknown error"}`);
    }

    return {
      temperature: data.main?.temp || null,
      pressure: data.main?.pressure || null,
      humidity: data.main?.humidity || null,
      heat_index: null,
    };
  } catch (error) {
    console.error("[ERR] Failed to fetch OpenWeather data:", error.message);
    throw error;
  }
}

async function collectReferenceMeasurement() {
  try {
    console.log("[INFO] Starting reference measurement collection...");

    const [aqicnData, weatherData] = await Promise.all([
      fetchAQICNData(),
      fetchOpenWeatherData(),
    ]);

    const newReferenceMeasurement = new ReferenceMeasurement({
      ...aqicnData,
      ...weatherData,
      latitude: LATITUDE,
      longitude: LONGITUDE,
      deviceId: "Reference-Station",
    });

    await newReferenceMeasurement.save();
    console.log("[INFO] Reference measurement saved successfully");
  } catch (error) {
    console.error("[ERR] Failed to collect reference measurement:", error.message);
  }
}

export function startReferenceMeasurementCron() {
  cron.schedule("0 * * * *", async () => {
    console.log("[INFO] Running hourly reference measurement collection...");
    await collectReferenceMeasurement();
  });

  console.log("[INFO] Reference measurement cron job scheduled (hourly)");
}
