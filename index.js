import express from "express";
import fs from "fs";
import path, { dirname } from "path";
import { fileURLToPath } from "url";
import cors from "cors";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const app = express();
const PORT = process.env.PORT || 3000;
app.use(cors());

app.get("/api/predictions", (req, res) => {
  const csvFile = path.join(__dirname, "predicted_pm25.csv");
  fs.readFile(csvFile, "utf8", (err, data) => {
    if (err) return res.status(500).json({ error: "Failed to read file" });

    const lines = data.split("\n");
    const headers = lines[0].replace(/\r/g, "").split(",");
    const rows = lines
      .slice(1)
      .filter(Boolean)
      .map((line) => {
        const values = line.replace(/\r/g, "").split(",");
        const obj = {};
        headers.forEach((h, i) => {
          obj[h] = values[i];
        });
        return obj;
      });

    res.json(rows);
  });
});

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
