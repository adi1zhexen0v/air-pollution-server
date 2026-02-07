import mongoose from "mongoose";
import "dotenv/config";

export async function connectDatabase() {
  try {
    await mongoose.connect(process.env.DB_URL);
    console.log("Database connected");
  } catch (error) {
    console.log("An error occurred while connecting to the database: " + error.message);
  }
}
