# ============================================================
# XGBOOST — MODELO ÚNICO
# Versión corregida: orden de operaciones y sin data leakage
# ============================================================

library(dplyr)
library(tidyr)
library(xgboost)

# ── 1. CARGA ─────────────────────────────────────────────────
d  = read.csv("datos_input.csv")
dp = read.csv("datos_output.csv")
regresores_externos = names(d)[26:317]

d_long = d %>%
  pivot_longer(
    cols      = starts_with("dem_h"),
    names_to  = "hora_str",
    values_to = "demanda"
  ) %>%
  mutate(hora = as.integer(sub("dem_h", "", hora_str))) %>%
  select(-hora_str)

# ── 3. LAGS POR HORA ─────────────────────────────────────────

d_long = d_long %>%
  arrange(hora, fecha) %>%
  group_by(hora) %>%
  mutate(
    demanda_lag1  = lag(demanda, 1),
    demanda_lag2  = lag(demanda, 2),
    demanda_lag3  = lag(demanda, 3),
    demanda_lag4  = lag(demanda, 4),
    demanda_lag5  = lag(demanda, 5),
    demanda_lag6  = lag(demanda, 6),
    demanda_lag7  = lag(demanda, 7),
    demanda_lag14 = lag(demanda, 14),
    demanda_lag21 = lag(demanda, 21),
    demanda_lag28 = lag(demanda, 28),
    demanda_lag35 = lag(demanda, 35),
    demanda_lag42 = lag(demanda, 42)
  ) %>%
  ungroup()

d_long = d_long %>% drop_na()


# ── 6. FEATURES ──────────────────────────────────────────────
lags_names <- c(
  "demanda_lag1",  "demanda_lag2",  "demanda_lag3",
  "demanda_lag4",  "demanda_lag5",  "demanda_lag6",
  "demanda_lag7",  "demanda_lag14", "demanda_lag21",
  "demanda_lag28", "demanda_lag35", "demanda_lag42"
)

# Total: 292 regresores + 1 hora + 17 lags = 310
features <- c(regresores_externos, "hora", lags_names)
cat("Total features:", length(features), "\n")
stopifnot(all(features %in% names(d_long)))

# ── 7. MÉTRICAS ──────────────────────────────────────────────
MSE  = function(real, pred) mean((real - pred)^2)
RMSE = function(real, pred) sqrt(MSE(real, pred))
MAPE = function(real_log, pred_log) {
  mean(abs((exp(real_log) - exp(pred_log)) / exp(real_log)), na.rm = TRUE) * 100
}

# ── 8. PARTICIÓN TEMPORAL ────────────────────────────────────
# 2010-2021: entrenamiento
# 2022:      test final — nunca visto durante entrenamiento

año <- substr(d_long$fecha, 1, 4)

datos_train <- d_long[año >= "2010" & año <= "2021", ]
datos_test  <- d_long[año == "2022", ]

cat("Train:", nrow(datos_train),
    "| Test:", nrow(datos_test), "\n")

stopifnot(
  nrow(datos_train) > 0,
  nrow(datos_test)  > 0
)

# ── 9. MATRICES XGBOOST ──────────────────────────────────────
X_train <- as.matrix(datos_train[, features])
y_train <- datos_train$demanda


X_test  <- as.matrix(datos_test[, features])
y_test  <- datos_test$demanda

dtrain <- xgb.DMatrix(data = X_train, label = y_train)
dtest  <- xgb.DMatrix(data = X_test,  label = y_test)

gc()

# ── 10. PARÁMETROS ───────────────────────────────────────────
params <- list(
  booster          = "gbtree",
  objective        = "reg:squarederror",
  eta              = 0.05,
  max_depth        = 10,
  subsample        = 0.8,
  colsample_bytree = 0.8,
  min_child_weight = 5,
  gamma            = 0,
  tree_method      = "hist"
)
# ── 11. ENTRENAMIENTOS MÚLTIPLES ─────────────────────────────

n_runs = 10
seeds = 42:51

resultados <- data.frame(
  seed = integer(), MSE = numeric(),
  RMSE = numeric(), MAPE = numeric()
)

modelo_referencia = NULL   # guardará el modelo con seed=42

for (i in seq_along(seeds)) {
  
  cat(sprintf("Ejecución %d/%d (seed=%d)\n", i, n_runs, seeds[i]))
  set.seed(seeds[i])
  
  modelo_tmp <- xgb.train(
    params    = params,
    data      = dtrain,
    nrounds   = 1000,
    watchlist = list(train = dtrain),
    verbose   = 0
  )
  
  pred_tmp <- predict(modelo_tmp, newdata = dtest)
  
  resultados <- rbind(resultados, data.frame(
    seed = seeds[i],
    MSE  = MSE(y_test, pred_tmp),
    RMSE = RMSE(y_test, pred_tmp),
    MAPE = MAPE(y_test, pred_tmp)
  ))
  
  # Guardar el modelo de la semilla de referencia
  if (seeds[i] == 42) {
    modelo_referencia <- modelo_tmp
    pred_referencia   <- pred_tmp
  }
}

# ── MÉTRICAS (media ± sd) ─────────────────────────
cat("\n===== MÉTRICAS: MEDIA DE 10 RÉPLICAS =====\n")
cat(sprintf("MSE  : %.7f ± %.7f\n", mean(resultados$MSE),  sd(resultados$MSE)))
cat(sprintf("RMSE : %.5f ± %.5f\n", mean(resultados$RMSE), sd(resultados$RMSE)))
cat(sprintf("MAPE : %.4f%% ± %.4f%%\n", mean(resultados$MAPE), sd(resultados$MAPE)))

# ── IMPORTANCIA DE VARIABLES (modelo seed=42) ────────────────
importancia <- xgb.importance(
  feature_names = features,
  model         = modelo_referencia
)

cat("\nTop 20 variables (seed=42):\n")
print(head(importancia, 20))

xgb.plot.importance(
  head(importancia, 20),
  main = "Top 20 variables — XGBoost (seed = 42)"
)
