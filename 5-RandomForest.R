# ============================================================
# RANDOM FOREST
# Features: regresores externos + 24 dummies hora + lags
# ============================================================

library(dplyr)
library(tidyr)
library(ranger)

# ── 1. CARGA ─────────────────────────────────────────────────
d <- read.csv("datos_input.csv")
regresores_externos <- names(d)[26:317]

# ── 2. RESHAPE ───────────────────────────────────────────────
d_long <- d %>%
  pivot_longer(
    cols      = starts_with("dem_h"),
    names_to  = "hora_str",
    values_to = "demanda"
  ) %>%
  mutate(
    hora  = factor(as.integer(sub("dem_h", "", hora_str)), levels = 1:24),
  ) %>%
  select(-hora_str)

# ── 3. LAGS DIARIOS ──────────────────────────────────────────
d_long <- d_long %>%
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
  ungroup() %>%
  drop_na()

# ── 4. DUMMIES DE HORA ───────────────────────────────────────
for (h in 1:24) {
  d_long[[paste0("hora_", h)]] <- as.integer(d_long$hora == h)
}
dummies_hora <- paste0("hora_", 1:24)

# ── 5. FEATURES ──────────────────────────────────────────────
lags_names <- c("demanda_lag1", "demanda_lag2", "demanda_lag3",
                "demanda_lag4", "demanda_lag5", "demanda_lag6",
                "demanda_lag7", "demanda_lag14", "demanda_lag21",
                "demanda_lag28", "demanda_lag35", "demanda_lag42")

features <- c(regresores_externos, dummies_hora, lags_names)
cat("Total features:", length(features), "\n")

# ── 6. TRAIN / TEST ──────────────────────────────────────────
año = substr(d_long$fecha, 1, 4)
datos_train <- d_long[año >= 2010 & año <= 2021, c("demanda", features)]
datos_test  <- d_long[año == 2022,  c("demanda", features)]

cat("Train:", nrow(datos_train), "| Test:", nrow(datos_test), "\n")

# ── 7. MÉTRICAS ──────────────────────────────────────────────
MSE  <- function(real, pred) mean((real - pred)^2)
RMSE <- function(real, pred) sqrt(MSE(real, pred))
MAPE <- function(real_log, pred_log) {
  mean(abs((exp(real_log) - exp(pred_log)) / exp(real_log)), na.rm = TRUE) * 100
}

# ── 9. RANDOM FOREST: 10 RÉPLICAS ────────────────────────────

seeds  <- 42:51
n_runs <- length(seeds)

resultados_rf <- data.frame(
  seed = integer(),
  MSE  = numeric(),
  RMSE = numeric(),
  MAPE = numeric()
)

modelo_ref  <- NULL
pred_ref    <- NULL

for (i in seq_along(seeds)) {
  
  cat(sprintf("\nRéplica %d/%d (seed=%d)\n", i, n_runs, seeds[i]))
  
  rf_tmp <- ranger(
    dependent.variable.name = "demanda",
    data                    = datos_train,
    num.trees               = 500,
    mtry                    = 65,
    min.node.size           = 5,
    max.depth               = 20,
    replace                 = FALSE,
    sample.fraction         = 0.632,
    splitrule               = "variance",
    importance              = if (seeds[i] == 42) "impurity" else "none",
    num.threads             = parallel::detectCores() - 1,
    seed                    = seeds[i],
    verbose                 = FALSE
  )
  
  pred_tmp <- predict(rf_tmp, data = datos_test)$predictions
  
  mse_i  <- MSE(datos_test$demanda, pred_tmp)
  rmse_i <- RMSE(datos_test$demanda, pred_tmp)
  mape_i <- MAPE(datos_test$demanda, pred_tmp)
  
  resultados_rf <- rbind(resultados_rf, data.frame(
    seed = seeds[i],
    MSE  = mse_i,
    RMSE = rmse_i,
    MAPE = mape_i
  ))
  
  cat(sprintf("   MSE=%.7f  RMSE=%.5f  MAPE=%.4f%%\n", mse_i, rmse_i, mape_i))
  
  # Guardar modelo de referencia (seed=42)
  if (seeds[i] == 42) {
    modelo_ref <- rf_tmp
    pred_ref   <- pred_tmp
  }
}

# ── 10. RESUMEN DE RÉPLICAS ──────────────────────────────────

cat("\n===== RANDOM FOREST: 10 RÉPLICAS (seeds 42–51) =====\n")
print(resultados_rf)
cat(sprintf("\nMSE  : %.7f ± %.7f\n", mean(resultados_rf$MSE),  sd(resultados_rf$MSE)))
cat(sprintf("RMSE : %.5f ± %.5f\n",   mean(resultados_rf$RMSE), sd(resultados_rf$RMSE)))
cat(sprintf("MAPE : %.4f%% ± %.4f%%\n", mean(resultados_rf$MAPE), sd(resultados_rf$MAPE)))

# ── 11. IMPORTANCIA DE VARIABLES (modelo seed=42) ────────────

importancia <- sort(modelo_ref$variable.importance, decreasing = TRUE)
top20 <- head(importancia, 20)

cat("\nTop 20 variables (seed=42):\n")
print(top20)

par(mar = c(5, 12, 4, 2))
barplot(rev(top20), horiz = TRUE, las = 1,
        border = "white",
        main = "Top 20 variables importantes — RF (seed = 42)",
        xlab = "Reducción de varianza")
par(mar = c(5, 4, 4, 2))

# ── 12. DIAGNÓSTICO DE RESIDUOS (modelo seed=42) ─────────────

et_rf <- datos_test$demanda - pred_ref

par(mfrow = c(1, 2))
acf(et_rf,  lag.max = 48, main = "ACF residuos — RF (seed = 42)")
pacf(et_rf, lag.max = 48, main = "PACF residuos — RF (seed = 42)")
par(mfrow = c(1, 1))

lb <- Box.test(et_rf, lag = 24, type = "Ljung-Box")
cat("\nLjung-Box(24): p =", round(lb$p.value, 4), "\n")