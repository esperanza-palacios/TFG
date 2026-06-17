# ============================================================
# MODELO DE REGRESIÓN LINEAL SIN RETARDOS — 24 HORAS
# ============================================================

d <- read.csv("datos_input.csv")
dp <- read.csv("datos_output.csv")

library(dplyr)
library(lubridate)

source("funciones/MSE.R")

# Partición temporal
año <- substr(d$fecha, 1, 4)
datos_train <- d[año >= "2010" & año <= "2021", ]
datos_test  <- d[año == "2022", ]

# Regresores (columnas 26 a 317, igual que tu código)
regresores <- names(d)[26:317]

# ============================================================
# BUCLE SOBRE LAS 24 HORAS
# ============================================================

resultados <- data.frame(
  hora   = 1:24,
  MSE_0  = NA,   # MSE modelo propio
  MSE_1  = NA    # MSE modelo referencia REE
)

modelos      <- list()
predicciones <- list()

for (h in 1:24) {
  
  hora_str <- sprintf("h%02d", h)           # "h01", "h02", ..., "h24"
  var_dep  <- paste0("dem_", hora_str)       # "dem_h01", ..., "dem_h24"
  var_ref  <- paste0("pred_", hora_str)      # columna en datos_output.csv
  
  # Fórmula
  formula_h <- as.formula(paste(var_dep, "~", paste(regresores, collapse = " + ")))
  
  # Ajuste sobre train
  modelo_h <- lm(formula_h, data = datos_train)
  modelos[[hora_str]] <- modelo_h
  
  # Predicción sobre test
  pred_h <- predict(modelo_h, newdata = datos_test)
  
  # Predicción del modelo de referencia REE (en log)
  pred_ref_h <- log(dp[[var_ref]])
  # Si ya viene en log en tu archivo, usa directamente: dp[[var_ref]]
  
  # MSE modelo propio
  resultados$MSE_0[h] <- MSE(datos_test[[var_dep]], pred_h[1:365])
  
  # MSE modelo referencia
  resultados$MSE_1[h] <- MSE(datos_test[[var_dep]], pred_ref_h[1:365])
  
  # Guardar predicciones
  predicciones[[hora_str]] <- data.frame(
    fecha      = datos_test$fecha,
    pred_propia = pred_h[1:365],
    pred_ref   = pred_ref_h[1:365],
    real       = datos_test[[var_dep]]
  )
  
  cat("Hora", hora_str, "— MSE_0:", round(resultados$MSE_0[h], 6),
      "| MSE_1:", round(resultados$MSE_1[h], 6), "\n")
}

# ============================================================
# RESULTADOS GLOBALES
# ============================================================

cat("\n--- MSE GLOBAL (promedio 24 horas) ---\n")
cat("Modelo propio (sin retardos):", round(mean(resultados$MSE_0), 6), "\n")
cat("Modelo referencia REE:       ", round(mean(resultados$MSE_1), 6), "\n")

print(resultados)

# ============================================================
# ACF DE RESIDUOS — hora 01 como ejemplo representativo
# ============================================================

etsr_h01 <- residuals(modelos[["h01"]])

par(mfrow = c(1, 2))
acf(etsr_h01,  lag.max = 40, main = "ACF residuos — h01")
pacf(etsr_h01, lag.max = 20, main = "PACF residuos — h01")
par(mfrow = c(1, 1))

# ============================================================
# TABLA RESUMEN PARA EL TFG
# ============================================================

resultados$hora_label <- sprintf("h%02d", resultados$hora)
print(resultados[, c("hora_label", "MSE_0", "MSE_1")])
# ============================================================
# FIGURA: MSE por hora — modelo propio vs referencia REE
# ============================================================

library(ggplot2)
library(tidyr)

# Tabla en formato largo para ggplot
resultados_long <- tidyr::pivot_longer(
  resultados,
  cols      = c(MSE_0, MSE_1),
  names_to  = "modelo",
  values_to = "MSE"
)

resultados_long$modelo <- factor(
  resultados_long$modelo,
  levels = c("MSE_0", "MSE_1"),
  labels = c("Regresión lineal sin retardos", "Modelo referencia REE")
)

ggplot(resultados_long, aes(x = hora, y = MSE, color = modelo, group = modelo)) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 1.5) +
  scale_x_continuous(breaks = 1:24) +
  scale_color_manual(values = c("#E05C3A", "#2E75B6")) +
  labs(
    title  = "MSE por hora — modelo sin retardos vs referencia REE (test 2022)",
    x      = "Hora del día",
    y      = "MSE (escala logarítmica)",
    color  = NULL
  ) +
  theme_minimal(base_size = 11) +
  theme(legend.position = "bottom")

# ============================================================
# FIGURA: ACF y PACF de residuos — hora 01
# ============================================================

etsr_h01 <- residuals(modelos[["h01"]])

par(mfrow = c(1, 2))
acf(etsr_h01,
    lag.max = 40,
    main    = "ACF de residuos — h01 (sin retardos)",
    xlab    = "Retardo (días)",
    ylab    = "Autocorrelación")
pacf(etsr_h01,
     lag.max = 20,
     main    = "PACF de residuos — h01 (sin retardos)",
     xlab    = "Retardo (días)",
     ylab    = "Autocorrelación parcial")
par(mfrow = c(1, 1))

# ============================================================
# FUNCIONES DE MÉTRICAS
# Nota: real y pred están en escala LOG
# RMSE y MAPE se calculan en escala ORIGINAL (exp)
# MSE se mantiene en log para comparabilidad con REE
# ============================================================

MSE <- function(real_log, pred_log) {
  mean((real_log - pred_log)^2)
}

RMSE <- function(real_log, pred_log) {
  # Escala original
  real_orig <- real_log
  pred_orig <- pred_log
  sqrt(mean((real_orig - pred_orig)^2))
}

MAPE <- function(real_log, pred_log) {
  # Escala original — excluye ceros por seguridad
  real_orig <- exp(real_log)
  pred_orig <- exp(pred_log)
  mean(abs((real_orig - pred_orig) / real_orig)) * 100
}

# ============================================================
# BUCLE SOBRE LAS 24 HORAS
# ============================================================

resultados <- data.frame(
  hora  = 1:24,
  MSE_0 = NA,   # MSE modelo propio    (escala log)
  MSE_1 = NA,   # MSE referencia REE   (escala log)
  RMSE_0 = NA,  # RMSE modelo propio   (escala original)
  RMSE_1 = NA,  # RMSE referencia REE  (escala original)
  MAPE_0 = NA,  # MAPE modelo propio   (%)
  MAPE_1 = NA   # MAPE referencia REE  (%)
)

for (h in 1:24) {
  
  hora_str <- sprintf("h%02d", h)
  var_dep  <- paste0("dem_", hora_str)
  var_ref  <- paste0("pred_", hora_str)
  
  formula_h <- as.formula(paste(var_dep, "~", paste(regresores, collapse = " + ")))
  modelo_h  <- lm(formula_h, data = datos_train)
  modelos[[hora_str]] <- modelo_h
  
  # Predicción del modelo propio (en log)
  n_test  <- nrow(datos_test)          # evita el parche [1:365]
  pred_h  <- predict(modelo_h, newdata = datos_test)[1:n_test]
  
  # Predicción de referencia REE
  # IMPORTANTE: verificar si dp[[var_ref]] está en log o en escala original.
  # Si está en escala original (MWh): pred_ref_h <- log(dp[[var_ref]][1:n_test])
  # Si ya está en log:               pred_ref_h <- dp[[var_ref]][1:n_test]
  pred_ref_h <- log(dp[[var_ref]])[1:n_test]   # ajustar según escala del archivo
  
  real_h <- datos_test[[var_dep]][1:n_test]
  
  # --- Métricas ---
  resultados$MSE_0[h]  <- MSE(real_h,  pred_h)
  resultados$MSE_1[h]  <- MSE(real_h,  pred_ref_h)
  resultados$RMSE_0[h] <- RMSE(real_h, pred_h)
  resultados$RMSE_1[h] <- RMSE(real_h, pred_ref_h)
  resultados$MAPE_0[h] <- MAPE(real_h, pred_h)
  resultados$MAPE_1[h] <- MAPE(real_h, pred_ref_h)
  
  cat(sprintf(
    "Hora %s | MSE_0: %.6f | MSE_1: %.6f | RMSE_0: %.2f | RMSE_1: %.2f | MAPE_0: %.3f%% | MAPE_1: %.3f%%\n",
    hora_str,
    resultados$MSE_0[h],  resultados$MSE_1[h],
    resultados$RMSE_0[h], resultados$RMSE_1[h],
    resultados$MAPE_0[h], resultados$MAPE_1[h]
  ))
}

# ============================================================
# RESUMEN GLOBAL
# ============================================================

cat("\n--- MÉTRICAS GLOBALES (promedio 24 horas) ---\n")
cat(sprintf("                     Modelo propio   Ref. REE\n"))
cat(sprintf("MSE   (escala log):  %.6f       %.6f\n",
            mean(resultados$MSE_0),  mean(resultados$MSE_1)))
cat(sprintf("RMSE  (escala orig): %.4f         %.4f\n",
            mean(resultados$RMSE_0), mean(resultados$RMSE_1)))
cat(sprintf("MAPE  (%%):           %.4f         %.4f\n",
            mean(resultados$MAPE_0), mean(resultados$MAPE_1)))