const predictForm = document.getElementById("predictForm");
const resultNode  = document.getElementById("predictionResult");
const predictBtn  = document.getElementById("predictBtn");

function setResult(text, state) {
  resultNode.textContent = text;
  resultNode.className = "state-" + state;
}

async function runPrediction(event) {
  event.preventDefault();
  predictBtn.disabled = true;
  setResult("Predicting…", "loading");

  const payload = {
    data: {
      CGPA: Number(document.getElementById("cgpa").value),
      IQ: Number(document.getElementById("iq").value),
      "Profile Score": Number(document.getElementById("profileScore").value),
    },
  };

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const body = await response.json();

    if (!response.ok) throw new Error(body.detail || "Prediction failed");

    const val = String(body.prediction);
    const isPlaced = val === "1" || val.toLowerCase().includes("placed");
    setResult(isPlaced ? "✅ Placed" : val, isPlaced ? "placed" : "placed");
  } catch (error) {
    setResult("Error: " + error.message, "error");
  } finally {
    predictBtn.disabled = false;
  }
}

predictForm.addEventListener("submit", runPrediction);