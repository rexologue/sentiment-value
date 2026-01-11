const classChips = document.getElementById("class-chips");
const classesStatus = document.getElementById("classes-status");
const predictButton = document.getElementById("predict-btn");
const inputText = document.getElementById("input-text");
const errorMessage = document.getElementById("error-message");
const predictionLabel = document.getElementById("prediction-label");
const probabilityList = document.getElementById("probability-list");

let classes = [];

function setError(message) {
  errorMessage.textContent = message;
}

function clearResults() {
  predictionLabel.textContent = "—";
  probabilityList.innerHTML = "";
}

function renderClasses() {
  classChips.innerHTML = "";
  classes.forEach((item) => {
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.textContent = item;
    classChips.appendChild(chip);
  });
}

function renderProbabilities(result) {
  probabilityList.innerHTML = "";
  classes.forEach((label, index) => {
    const valueKey = `class_${index}`;
    const value = result[valueKey];
    const row = document.createElement("div");
    row.className = "probability-row";
    if (label === result.prediction) {
      row.classList.add("highlight");
    }

    const labelWrap = document.createElement("div");
    labelWrap.className = "probability-label";
    const title = document.createElement("strong");
    title.textContent = label;
    const subtitle = document.createElement("span");
    subtitle.textContent = valueKey;
    labelWrap.appendChild(title);
    labelWrap.appendChild(subtitle);

    const valueWrap = document.createElement("div");
    valueWrap.className = "probability-value";
    const numeric = typeof value === "number" ? value : Number(value);
    if (Number.isFinite(numeric)) {
      const percent = (numeric * 100).toFixed(2);
      valueWrap.textContent = `${numeric.toFixed(4)} (${percent}%)`;
    } else {
      valueWrap.textContent = "—";
    }

    row.appendChild(labelWrap);
    row.appendChild(valueWrap);
    probabilityList.appendChild(row);
  });
}

async function loadClasses() {
  classesStatus.textContent = "Loading...";
  try {
    const response = await fetch("/classes");
    if (!response.ok) {
      throw new Error(`Failed to load classes (${response.status})`);
    }
    const data = await response.json();
    if (!Array.isArray(data.classes)) {
      throw new Error("Invalid classes response");
    }
    classes = data.classes;
    renderClasses();
    classesStatus.textContent = `${classes.length} classes`;
  } catch (error) {
    classesStatus.textContent = "Unavailable";
    setError("Unable to load classes. Please check the API connection.");
  }
}

async function handlePredict() {
  setError("");
  const text = inputText.value.trim();
  if (!text) {
    setError("Please enter some text to classify.");
    return;
  }
  if (classes.length === 0) {
    setError("Classes not loaded yet. Please wait.");
    return;
  }
  predictButton.disabled = true;
  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ text }),
    });
    if (!response.ok) {
      const detail = await response.json().catch(() => ({}));
      const message = detail.detail || "Prediction failed.";
      throw new Error(message);
    }
    const result = await response.json();
    if (!result.prediction) {
      throw new Error("Invalid prediction response.");
    }
    predictionLabel.textContent = result.prediction;
    renderProbabilities(result);
  } catch (error) {
    clearResults();
    setError(error.message || "Prediction failed.");
  } finally {
    predictButton.disabled = false;
  }
}

predictButton.addEventListener("click", handlePredict);

loadClasses();
