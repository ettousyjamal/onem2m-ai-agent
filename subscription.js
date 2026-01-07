const express = require("express");
const bodyParser = require("body-parser");
const axios = require("axios");

const app = express();
const PORT = 4000;

// Your Mobius config
const MOBIUS_BASE_URL = "http://54.242.158.10:7579"; // Updated to match Python config
const AE_ID = "admin:admin"; // default origin (username:password)
const AE_NAME = "PlantDataListener";
const NOTI_URI = `http://54.242.158.10:${PORT}/notification`; // must be accessible by Mobius

// For parsing JSON
app.use(bodyParser.json());

// 1. Start an Express server to receive notifications
app.post("/notification", (req, res) => {
  try {
    const sgn = req.body["m2m:sgn"];
    const cin = sgn?.nev?.rep["m2m:cin"];
    const subscriptionUri = sgn?.sur || "unknown";

    if (cin) {
      // subscriptionUri example: /Mobius/Greenhouse1/PlantData1/sub_123456-20250518095013311300
      // Extract full path after /Mobius/
      const parts = subscriptionUri.split('/');
      const ae = parts[1] || "unknown AE";  // Should be "Greenhouse1", "Greenhouse2", etc.
      // full path after Mobius, excluding last segment (resource instance id)
      const fullPath = parts.slice(2, -1).join('/') || "unknown container";

      // Extract greenhouse and data container from the path
      const pathSegments = parts.slice(2);
      const greenhouse = parts[1] || "unknown greenhouse"; // The AE is the greenhouse
      const dataContainer = pathSegments[0] || "unknown container"; // PlantData1, PlantData2, etc.

      const notificationData = {
        AE: ae,
        Greenhouse: greenhouse,
        DataContainer: dataContainer,
        FullContainerPath: fullPath,
        Value: cin.con,
        CreatedTime: cin.ct,
        ResourceId: cin.ri,
        Timestamp: new Date().toISOString()
      };

      console.log("🌿 Plant Data Notification received:");
      console.log(notificationData);
      
      // You can add additional processing here, such as:
      // - Parse the complete data row from cin.con
      // - Store to database
      // - Send to analytics service
      // - Trigger alerts based on plant health data
      
    } else {
      console.log("Unknown payload:", JSON.stringify(req.body, null, 2));
    }
  } catch (error) {
    console.error("Error processing notification:", error);
  }
  res.sendStatus(200);
});

// 2. Create subscription with X-M2M-RI header fixed
async function createSubscription(container) {
  try {
    const subName = `sub_${Date.now()}`;
    const response = await axios.post(
      `${MOBIUS_BASE_URL}/Mobius/${container}`,
      {
        "m2m:sub": {
          rn: subName,
          nu: [NOTI_URI],
          nct: 2, // notification content type = whole resource
          enc: { net: [3] }, // event type: update of resource (like createContentInstance)
        },
      },
      {
        headers: {
          "X-M2M-Origin": AE_ID,
          "Content-Type": "application/json;ty=23", // type=23 -> subscription
          "X-M2M-RI": Date.now().toString(), // unique request ID (required!)
        },
      }
    );
    console.log(`✅ Subscribed to /Mobius/${container} with name ${subName}`);
  } catch (err) {
    console.error("❌ Failed to create subscription:", err.response?.data || err.message);
  }
}

// 3. Start the server
app.listen(PORT, () => {
  console.log(`🚀 Plant Data Listener running at http://localhost:${PORT}`);
  
  // Define greenhouses and data containers (matching Python structure)
  const GREENHOUSES = ["Greenhouse1", "Greenhouse2", "Greenhouse3", "Greenhouse4"];
  const DATA_CONTAINERS = Array.from({length: 10}, (_, i) => `PlantData${i + 1}`);

  console.log("🏠 Greenhouses being monitored:", GREENHOUSES);
  console.log("📊 Data containers being monitored:", DATA_CONTAINERS);

  // Create subscriptions for all greenhouses and data containers
  GREENHOUSES.forEach(greenhouse => {
    console.log(`Setting up subscriptions for ${greenhouse}...`);
    DATA_CONTAINERS.forEach(container => {
      createSubscription(`${greenhouse}/${container}`);
    });
  });
  
  console.log("Notification subscriptions setup complete!");
  console.log("Ready to receive plant data notifications...");
});