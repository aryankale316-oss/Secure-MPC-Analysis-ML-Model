// ===============================
// ELEMENT REFERENCES
// ===============================
const form = document.getElementById("predictionForm");

const idleView = document.getElementById("idleView");
const loadingView = document.getElementById("loadingView");
const resultView = document.getElementById("resultView");

const predictionText = document.getElementById("predictionText");
const confValue = document.getElementById("confValue");
const meterFill = document.getElementById("meterFill");
const riskAdvice = document.getElementById("riskAdvice");

// ===============================
// FORM SUBMIT → SEND TO FLASK
// ===============================
form.addEventListener("submit", async (e) => {
    e.preventDefault();

    // Switch views
    idleView.classList.add("hidden");
    resultView.classList.add("hidden");
    loadingView.classList.remove("hidden");

    // Collect 13 features in correct order
    const features = [
        age.value,
        sex.value,
        cp.value,
        trestbps.value,
        chol.value,
        fbs.value,
        restecg.value,
        thalach.value,
        exang.value,
        oldpeak.value,
        slope.value,
        ca.value,
        thal.value
    ].map(v => Number(v) || 0); // prevent NaN

    console.log("📤 Sending features:", features);

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ features })
        });

        const data = await response.json();
        console.log("📥 Response:", data);

        // Validate response
        const prediction = data.prediction || "Unknown";
        const confidence = Number(data.confidence) || 0;

        // Show results after delay for UX
        setTimeout(() => {
            loadingView.classList.add("hidden");
            resultView.classList.remove("hidden");

            const isDisease = prediction === "Disease Detected";
            const color = isDisease ? "#fb7185" : "#34d399";

            // ================= UI UPDATE =================
            predictionText.innerText = prediction;
            confValue.innerText = confidence.toFixed(2);

            // Reset glow classes
            predictionText.classList.remove("success-glow", "danger-glow");
            meterFill.classList.remove("glow");

            // Apply color + glow
            predictionText.style.color = color;
            meterFill.style.backgroundColor = color;
            meterFill.style.width = confidence + "%";

            if (isDisease) {
                predictionText.classList.add("danger-glow");
            } else {
                predictionText.classList.add("success-glow");
            }

            meterFill.classList.add("glow");

            // Advice text
            riskAdvice.innerText = isDisease
                ? "⚠ High cardiovascular risk. Please consult a cardiologist."
                : "✓ Low risk. Maintain a healthy lifestyle.";

        }, 700);

    } catch (error) {
        console.error("❌ Error:", error);
        alert("Server connection failed. Make sure Flask is running.");
        loadingView.classList.add("hidden");
        idleView.classList.remove("hidden");
    }
});

// ===============================
// FILL HEALTHY DATA
// ===============================
function fillHealthyData() {
    age.value = 30;
    sex.value = 0;
    cp.value = 0;
    trestbps.value = 110;
    chol.value = 175;
    fbs.value = 0;
    restecg.value = 0;
    thalach.value = 180;
    exang.value = 0;
    oldpeak.value = 0;
    slope.value = 1; // correct healthy slope
    ca.value = 0;
    thal.value = 1;
}

// ===============================
// RESET FORM
// ===============================
function resetForm() {
    form.reset();

    // Reset UI
    meterFill.style.width = "0%";
    predictionText.innerText = "--";
    confValue.innerText = "0";

    predictionText.classList.remove("success-glow", "danger-glow");
    meterFill.classList.remove("glow");

    resultView.classList.add("hidden");
    loadingView.classList.add("hidden");
    idleView.classList.remove("hidden");
}