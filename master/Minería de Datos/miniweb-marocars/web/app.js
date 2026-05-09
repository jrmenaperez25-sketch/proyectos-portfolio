const API_BASE_URL =
  window.APP_CONFIG?.API_BASE_URL ||
  "http://127.0.0.1:8000";

const state = {
  metadata: null,
  lastPrediction: null
};

const form = document.getElementById("car-form");
const brandSelect = document.getElementById("brand");
const modelSelect = document.getElementById("model");
const ageSelect = document.getElementById("age");
const mileageInput = document.getElementById("mileage");
const fiscalPowerSelect = document.getElementById("fiscal-power");
const doorsSelect = document.getElementById("doors");
const gearboxSelect = document.getElementById("gearbox");
const fuelSelect = document.getElementById("fuel");
const conditionSelect = document.getElementById("condition");
const extrasContainer = document.getElementById("extras-container");
const resultsSection = document.getElementById("results");
const estimatedPriceNode = document.getElementById("estimated-price");
const firstOwnerLabelNode = document.getElementById("first-owner-label");
const firstOwnerProbabilityNode = document.getElementById("first-owner-probability");
const warningsNode = document.getElementById("result-warnings");
const inlineWarningNode = document.getElementById("inline-warning");
const downloadPdfButton = document.getElementById("download-pdf");

function buildOption(value, label = value) {
  const option = document.createElement("option");
  option.value = value;
  option.textContent = label;
  return option;
}

function resetSelect(selectNode, placeholder) {
  selectNode.innerHTML = "";
  selectNode.appendChild(buildOption("", placeholder));
}

function fillSelect(selectNode, values, placeholder, formatter) {
  resetSelect(selectNode, placeholder);
  values.forEach((value) => {
    const label = formatter ? formatter(value) : value;
    selectNode.appendChild(buildOption(value, label));
  });
}

function renderExtras(extras) {
  extrasContainer.innerHTML = "";
  extras.forEach((extra) => {
    const wrapper = document.createElement("label");
    wrapper.className = "extra-item";

    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.name = "extras";
    checkbox.value = extra;

    const text = document.createElement("span");
    text.textContent = extra;

    wrapper.appendChild(checkbox);
    wrapper.appendChild(text);
    extrasContainer.appendChild(wrapper);
  });
}

function populateBrandModelMap() {
  const brands = Object.keys(state.metadata.brand_model_map);
  fillSelect(brandSelect, brands, "Selecciona una marca");
  resetSelect(modelSelect, "Selecciona primero una marca");
  modelSelect.disabled = true;
}

function updateModelsForBrand() {
  const selectedBrand = brandSelect.value;
  const models = state.metadata.brand_model_map[selectedBrand] || [];
  fillSelect(modelSelect, models, "Selecciona un modelo");
  modelSelect.disabled = models.length === 0;
}

function readSelectedExtras() {
  return [...document.querySelectorAll('input[name="extras"]:checked')].map(
    (checkbox) => checkbox.value
  );
}

function renderInlineWarnings() {
  const mileage = Number(mileageInput.value);
  if (Number.isFinite(mileage) && mileage > 700000) {
    inlineWarningNode.textContent =
      "Kilometraje anomalo: superar 700000 km puede afectar a la precision de la prediccion.";
    inlineWarningNode.classList.remove("hidden");
    return;
  }

  inlineWarningNode.textContent = "";
  inlineWarningNode.classList.add("hidden");
}

function buildPayload() {
  return {
    brand: brandSelect.value,
    model: modelSelect.value,
    age: Number(ageSelect.value),
    mileage_mid: Number(mileageInput.value),
    fiscal_power_num: Number(fiscalPowerSelect.value),
    number_of_doors: doorsSelect.value,
    gearbox: gearboxSelect.value,
    fuel: fuelSelect.value,
    condition: conditionSelect.value,
    extras: readSelectedExtras()
  };
}

function formatMad(value) {
  return new Intl.NumberFormat("es-ES", {
    maximumFractionDigits: 0
  }).format(value);
}

function renderResults(prediction) {
  estimatedPriceNode.textContent = `${formatMad(prediction.estimated_price_mad)} MAD`;
  firstOwnerLabelNode.textContent = prediction.predicted_first_owner;
  firstOwnerProbabilityNode.textContent =
    `Probabilidad de primer propietario: ${Math.round(
      prediction.first_owner_probability * 100
    )}%`;

  warningsNode.innerHTML = "";
  prediction.warnings.forEach((warning) => {
    const item = document.createElement("div");
    item.className = "notice";
    item.textContent = warning;
    warningsNode.appendChild(item);
  });

  resultsSection.classList.remove("hidden");
}

async function loadMetadata() {
  const response = await fetch(`${API_BASE_URL}/metadata`);
  if (!response.ok) {
    throw new Error("No se pudieron cargar los metadatos del formulario.");
  }

  state.metadata = await response.json();

  populateBrandModelMap();
  fillSelect(
    ageSelect,
    state.metadata.age_values,
    "Selecciona antiguedad",
    (value) => `${value} anos`
  );
  fillSelect(
    fiscalPowerSelect,
    state.metadata.fiscal_power_values,
    "Selecciona potencia fiscal",
    (value) => `${value} CV`
  );
  fillSelect(doorsSelect, state.metadata.door_values, "Selecciona puertas");
  fillSelect(gearboxSelect, state.metadata.gearbox_values, "Selecciona cambio");
  fillSelect(fuelSelect, state.metadata.fuel_values, "Selecciona combustible");
  fillSelect(conditionSelect, state.metadata.condition_values, "Selecciona estado");
  renderExtras(state.metadata.extras);
}

async function submitPrediction(event) {
  event.preventDefault();
  renderInlineWarnings();

  const payload = buildPayload();
  const response = await fetch(`${API_BASE_URL}/predict`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(payload)
  });

  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || "No se pudo calcular la prediccion.");
  }

  state.lastPrediction = {
    ...payload,
    ...data
  };

  renderResults(state.lastPrediction);
}

function downloadPdf() {
  if (!state.lastPrediction) {
    return;
  }

  const { jsPDF } = window.jspdf;
  const pdf = new jsPDF();
  const p = state.lastPrediction;
  const extrasText = p.extras.length > 0 ? p.extras.join(", ") : "Sin extras marcados";
  const warningLines =
    p.warnings.length > 0 ? p.warnings : ["Sin avisos adicionales del modelo."];

  let y = 18;
  pdf.setFont("helvetica", "bold");
  pdf.setFontSize(18);
  pdf.text("SouqAuto Maroc - Informe de estimacion", 14, y);

  y += 10;
  pdf.setFont("helvetica", "normal");
  pdf.setFontSize(11);
  pdf.text(`Fecha: ${new Date().toLocaleString("es-ES")}`, 14, y);
  y += 10;
  pdf.text(`Marca: ${p.brand}`, 14, y);
  y += 8;
  pdf.text(`Modelo: ${p.model}`, 14, y);
  y += 8;
  pdf.text(`Antiguedad: ${p.age} anos`, 14, y);
  y += 8;
  pdf.text(`Kilometraje: ${formatMad(p.mileage_mid)} km`, 14, y);
  y += 8;
  pdf.text(`Potencia fiscal: ${p.fiscal_power_num} CV`, 14, y);
  y += 8;
  pdf.text(`Puertas: ${p.number_of_doors}`, 14, y);
  y += 8;
  pdf.text(`Cambio: ${p.gearbox}`, 14, y);
  y += 8;
  pdf.text(`Combustible: ${p.fuel}`, 14, y);
  y += 8;
  pdf.text(`Estado: ${p.condition}`, 14, y);
  y += 8;

  const extrasLines = pdf.splitTextToSize(`Extras: ${extrasText}`, 180);
  pdf.text(extrasLines, 14, y);
  y += extrasLines.length * 7 + 4;

  pdf.setFont("helvetica", "bold");
  pdf.text(`Precio estimado: ${formatMad(p.estimated_price_mad)} MAD`, 14, y);
  y += 8;
  pdf.text(
    `Probabilidad de first owner: ${Math.round(p.first_owner_probability * 100)}%`,
    14,
    y
  );
  y += 8;
  pdf.text(`Prediccion final: ${p.predicted_first_owner}`, 14, y);
  y += 12;

  pdf.setFont("helvetica", "normal");
  pdf.text("Avisos del modelo:", 14, y);
  y += 8;

  warningLines.forEach((line) => {
    const lines = pdf.splitTextToSize(`- ${line}`, 180);
    pdf.text(lines, 16, y);
    y += lines.length * 7;
  });

  pdf.save(`souqauto-${p.brand}-${p.model}.pdf`.replace(/\s+/g, "-").toLowerCase());
}

brandSelect.addEventListener("change", updateModelsForBrand);
mileageInput.addEventListener("input", renderInlineWarnings);
form.addEventListener("submit", (event) => {
  submitPrediction(event).catch((error) => {
    alert(error.message);
  });
});
downloadPdfButton.addEventListener("click", downloadPdf);

loadMetadata().catch((error) => {
  alert(error.message);
});
