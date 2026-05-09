library(plumber)
library(readxl)

options(stringsAsFactors = FALSE)

bundle_path <- "models/model_bundle.rds"
data_path <- if (file.exists("../data/datos_imputados_knn.xlsx")) {
  "../data/datos_imputados_knn.xlsx"
} else {
  "/data/datos_imputados_knn.xlsx"
}

if (!file.exists(bundle_path)) {
  stop("No se encontro models/model_bundle.rds. Ejecuta antes train_models.R.")
}

if (!file.exists(data_path)) {
  stop("No se encontro data/datos_imputados_knn.xlsx.")
}

bundle <- readRDS(bundle_path)
raw_data <- as.data.frame(read_excel(data_path))

raw_data$BrandModel <- as.character(raw_data$BrandModel)
raw_data$Gearbox <- as.character(raw_data$Gearbox)
raw_data$Fuel <- as.character(raw_data$Fuel)
raw_data$Condition <- as.character(raw_data$Condition)
raw_data$Number.of.Doors <- as.character(raw_data$Number.of.Doors)

split_brand_model <- function(values) {
  out <- list()
  for (value in sort(unique(na.omit(values)))) {
    parts <- strsplit(value, " ", fixed = TRUE)[[1]]
    brand <- parts[1]
    model <- if (length(parts) > 1) paste(parts[-1], collapse = " ") else parts[1]
    current <- out[[brand]]
    if (is.null(current)) {
      current <- character(0)
    }
    out[[brand]] <- sort(unique(c(current, model)))
  }
  out[sort(names(out))]
}

brand_model_map <- split_brand_model(raw_data$BrandModel)

metadata <- list(
  age_values = as.list(bundle$age_values),
  fiscal_power_values = as.list(bundle$fiscal_power_values),
  gearbox_values = as.list(bundle$levels$Gearbox),
  fuel_values = as.list(bundle$levels$Fuel),
  condition_values = as.list(bundle$levels$Condition),
  door_values = as.list(bundle$levels$Number.of.Doors),
  first_owner_labels = list(yes = "Si", no = "No"),
  extras = bundle$extras_catalog,
  brand_model_map = brand_model_map,
  luxury_brands = bundle$luxury_brands
)

coerce_factor <- function(value, levels) {
  factor(as.character(value), levels = levels)
}

build_input_row <- function(payload) {
  extras_total <- length(intersect(payload$extras, bundle$extras_catalog))
  data.frame(
    Age = as.numeric(payload$age),
    Mileage_mid = as.numeric(payload$mileage_mid),
    Fiscal_Power_num = as.numeric(payload$fiscal_power_num),
    Number.of.Doors = coerce_factor(payload$number_of_doors, bundle$levels$Number.of.Doors),
    n_extras_total = as.numeric(extras_total),
    Gearbox = coerce_factor(payload$gearbox, bundle$levels$Gearbox),
    Fuel = coerce_factor(payload$fuel, bundle$levels$Fuel),
    Condition = coerce_factor(payload$condition, bundle$levels$Condition)
  )
}

detect_warnings <- function(payload) {
  warnings <- character(0)

  if (!is.null(payload$mileage_mid) && isTRUE(as.numeric(payload$mileage_mid) > 700000)) {
    warnings <- c(
      warnings,
      "Kilometraje anomalo: superar 700000 km puede afectar a la precision de la prediccion."
    )
  }

  if (!is.null(payload$brand) && payload$brand %in% bundle$luxury_brands) {
    warnings <- c(
      warnings,
      "Auto de lujo: en este segmento la prediccion puede fallar o ser menos precisa."
    )
  }

  warnings
}

#* @filter cors
function(req, res) {
  res$setHeader("Access-Control-Allow-Origin", "*")
  res$setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
  res$setHeader("Access-Control-Allow-Headers", "Content-Type")

  if (req$REQUEST_METHOD == "OPTIONS") {
    res$status <- 200
    return(list())
  }

  plumber::forward()
}

#* Estado del servicio
#* @get /health
#* @serializer json list(auto_unbox = TRUE)
function() {
  list(status = "ok", model_bundle_loaded = TRUE)
}

#* Metadatos para la web
#* @get /metadata
#* @serializer json list(auto_unbox = TRUE)
function() {
  metadata
}

#* Prediccion de precio y first owner
#* @post /predict
#* @serializer json list(auto_unbox = TRUE)
function(req, res) {
  payload <- jsonlite::fromJSON(req$postBody, simplifyVector = TRUE)

  required_fields <- c(
    "brand", "model", "age", "mileage_mid", "fiscal_power_num", "number_of_doors",
    "gearbox", "fuel", "condition"
  )

  missing_fields <- required_fields[!nzchar(as.character(payload[required_fields]))]
  if (length(missing_fields) > 0) {
    res$status <- 400
    return(list(error = paste("Faltan campos:", paste(missing_fields, collapse = ", "))))
  }

  input_row <- build_input_row(payload)

  prob_first_owner <- as.numeric(
    predict(bundle$mod.logit, newdata = input_row, type = "response")
  )
  predicted_first_owner <- if (prob_first_owner >= 0.5) "Si" else "No"
  estimated_price <- as.numeric(predict(bundle$mod.rf, newdata = input_row))

  list(
    brand = payload$brand,
    model = payload$model,
    estimated_price_mad = round(estimated_price, 0),
    first_owner_probability = round(prob_first_owner, 4),
    predicted_first_owner = predicted_first_owner,
    extras_selected = as.list(intersect(payload$extras, bundle$extras_catalog)),
    extras_total = input_row$n_extras_total[[1]],
    warnings = as.list(detect_warnings(payload))
  )
}
