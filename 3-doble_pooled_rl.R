# ============================================================
# MODELO POOLED CON ETAPA DE CORRECCIÓN DE RESIDUOS
# ============================================================

library(dplyr)
library(tidyr)
library(stats)
library(zoo)

# ── 1. CARGA Y RESHAPE ───────────────────────────────────────
d = read.csv("datos_input.csv")

cols_meta       = names(d)[1:25]
cols_regresores = names(d)[26:317]
cols_demanda    = grep("^dem_h", names(d), value = TRUE)

d_long <- d %>%
  pivot_longer(
    cols      = all_of(cols_demanda),
    names_to  = "hora_str",
    values_to = "demanda"
  ) %>%
  mutate(
    hora = as.integer(sub("dem_h", "", hora_str))
  ) %>%
  select(-hora_str) %>%
  arrange(hora, fecha)

# ── 2. LAGS (misma hora, días anteriores) ────────────────────
d_long = d_long %>%
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
    demanda_lag21  = lag(demanda, 21),
    demanda_lag28 = lag(demanda, 28),
    demanda_lag35 = lag(demanda, 35),
    demanda_lag42 = lag(demanda, 42)
  ) %>%
  ungroup() %>%
  drop_na()

# ── 3. FACTOR HORA ───────────────────────────────────────────
d_long = d_long %>%
  mutate(hora_f = factor(hora))

# ── 4. TRAIN / TEST SPLIT ────────────────────────────────────
año = substr(d_long$fecha, 1, 4)

datos_train = d_long[año >= 2010 & año <= 2021, ]
datos_test  = d_long[año == 2022, ]

cat("Train:", nrow(datos_train), "obs |",
    "Test:", nrow(datos_test), "obs\n")

# ── 6. CONSTRUCCIÓN DE LA FÓRMULA ────────────────────────────
lags = c("demanda_lag1", "demanda_lag2", "demanda_lag3",
          "demanda_lag4", "demanda_lag5", "demanda_lag6",
          "demanda_lag7", "demanda_lag14", "demanda_lag21",
          "demanda_lag28", "demanda_lag35", "demanda_lag42")

formula_str = paste(
  "demanda ~",
  "hora_f",
  "+", paste(paste0("hora_f:", lags), collapse = " + "),
  "+", paste(cols_regresores, collapse = " + ")
)

formula_modelo = as.formula(formula_str)

# ── 7. MODELO ETAPA 1 ─────────────────────────────────────────
modelo = lm(formula_modelo, data = datos_train)

# ── 8. PREDICCIÓN Y MÉTRICAS ETAPA 1 ─────────────────────────
pred = predict(modelo, newdata = datos_test)

MSE  = function(real, pred) mean((real - pred)^2, na.rm = TRUE)
RMSE = function(real, pred) sqrt(MSE(real, pred))
MAPE = function(real_log, pred_log) {
  mean(abs((exp(real_log) - exp(pred_log)) / exp(real_log)), na.rm = TRUE) * 100
}

cat("\n── Métricas en TEST (2022) — Etapa 1 ──\n")
cat("MSE :", round(MSE(datos_test$demanda, pred), 7), "\n")
cat("RMSE:", round(RMSE(datos_test$demanda, pred), 5), "\n")
cat("MAPE:", round(MAPE(datos_test$demanda, pred), 4), "%\n")

# ── DIAGNÓSTICO DE RESIDUOS ETAPA 1 ──────────────────────────
et = residuals(modelo)

par(mfrow = c(2, 3))
acf(et,  lag.max = 48, main = "ACF residuos (train)")
pacf(et, lag.max = 48, main = "PACF residuos (train)")
plot(fitted(modelo), et, pch = ".", col = rgb(0,0,0,0.2),
     xlab = "Ajustados", ylab = "Residuos", main = "Residuos vs Ajustados")
abline(h = 0, col = "red", lwd = 2)
qqnorm(et, pch = ".", main = "QQ-plot")
qqline(et, col = "red", lwd = 2)
plot(et, type = "l", col = "steelblue", main = "Residuos en el tiempo")
abline(h = 0, col = "red")
hist(et, breaks = 80, col = "steelblue", border = "white",
     main = "Histograma residuos")
par(mfrow = c(1,1))



# ============================================================
# SEGUNDA ETAPA (POOLED): REGRESIÓN LINEAL SOBRE RESIDUOS
# ============================================================

library(dplyr)
library(tidyr)
library(zoo)

# Retardos del error (replican el conjunto K de la etapa 1)

lags_error <- c("e_lag1", "e_lag2", "e_lag3", "e_lag4", "e_lag5",
                "e_lag6", "e_lag7", "e_lag14", "e_lag21",
                "e_lag28", "e_lag35", "e_lag42")

# Features dinámicas 

feat_con_NA <- c(lags_error, "e_ma7", "e_ma14", "e_ma28", "e_var7")


# ── Encodings compartidos train/test (misma lógica, sin fugas) ──
# Niveles FIJADOS explícitamente para que predict() nunca vea un
# nivel nuevo y las matrices de diseño de train y test coincidan.
anyadir_encodings = function(df) {
  df %>%
    mutate(
      hora_f       = factor(hora,       levels = 1:24),
      dia_semana_f = factor(dia_semana, levels = 1:7),
      mes_f        = factor(mes,        levels = 1:12),
      doy_sin      = sin(2 * pi * dia_anyo / 365.25),
      doy_cos      = cos(2 * pi * dia_anyo / 365.25)
    )
}


# ── Fórmula pooled de la etapa 2 ─────────────────────────────

construir_formula_pooled = function() {
  termino_FE       = "hora_f"
  termino_interac  = paste0("hora_f:", lags_error)          # hora × retardo del error
  termino_dinamico = c("e_ma7", "e_ma14", "e_ma28", "e_var7")
  termino_calend   = c("dia_semana_f", "mes_f", "doy_sin", "doy_cos")
  rhs = c(termino_FE, termino_interac, termino_dinamico, termino_calend)
  as.formula(paste("error ~", paste(rhs, collapse = " + ")))
}

# ── construcción features ──────────────────────────
construir_features_train <- function(datos_train, et_train) {
  
  df = datos_train %>%
    mutate(error = et_train) %>%
    arrange(hora, fecha) %>%
    group_by(hora) %>%                       # lags SIEMPRE dentro de cada hora
    mutate(
      e_lag1  = lag(error, 1),
      e_lag2  = lag(error, 2),
      e_lag3  = lag(error, 3),
      e_lag4  = lag(error, 4),
      e_lag5  = lag(error, 5),
      e_lag6  = lag(error, 6),
      e_lag7  = lag(error, 7),
      e_lag14 = lag(error, 14),
      e_lag21 = lag(error, 21),
      e_lag28 = lag(error, 28),
      e_lag35 = lag(error, 35),
      e_lag42 = lag(error, 42),
      
      # Medias móviles — el lag(1) adicional evita incluir el error de hoy
      e_ma7  = lag(zoo::rollmean(error, k = 7,  fill = NA, align = "right"), 1),
      e_ma14 = lag(zoo::rollmean(error, k = 14, fill = NA, align = "right"), 1),
      e_ma28 = lag(zoo::rollmean(error, k = 28, fill = NA, align = "right"), 1),
      
      # Varianza móvil — proxy de incertidumbre reciente del modelo
      e_var7 = lag(zoo::rollapply(error, width = 7, FUN = var,
                                  fill = NA, align = "right"), 1),
      
    ) %>%
    ungroup() %>%
    mutate(
      fecha_date = as.Date(as.character(fecha), format = "%Y%m%d"),
      dia_semana = as.integer(format(fecha_date, "%u")),   # 1 = lunes ... 7 = domingo
      mes        = as.integer(format(fecha_date, "%m")),
      dia_anyo   = as.integer(format(fecha_date, "%j"))
    ) %>%
    select(-fecha_date) %>%
    anyadir_encodings()
  
  return(df)
}


# ── entrenamiento de la etapa 2 ──────────────────────────
entrenar_lm_residuos <- function(df_train_features) {
  
  formula_lm = construir_formula_pooled()
  modelo_lm  = lm(formula_lm, data = df_train_features)
  
  return(modelo_lm)
}


# ── prediccion con residuos observados ──────────────────────────
predecir_rolling_lm = function(datos_train, datos_test,
                                pred_train_base, pred_test_base,
                                modelo_lm2, verbose = TRUE) {
  
  # Buffer inicial: errores OLS reales de TODO el train
  buffer = data.frame(
    fecha = as.character(datos_train$fecha),
    hora  = datos_train$hora,
    error = datos_train$demanda - pred_train_base
  )
  
  fechas_test <- sort(unique(as.character(datos_test$fecha)))
  pred_final  <- numeric(nrow(datos_test))
  pred_error  <- numeric(nrow(datos_test))
  
  for (i in seq_along(fechas_test)) {
    
    fecha_d = fechas_test[i]
    if (verbose && i %% 30 == 1) {
      cat(sprintf("  Día %d/%d: %s\n", i, length(fechas_test), fecha_d))
    }
    
    fecha_d_date  = as.Date(fecha_d, format = "%Y%m%d")
    idx_dia       = which(as.character(datos_test$fecha) == fecha_d)
    horas_dia     = datos_test$hora[idx_dia]
    pred_base_dia = pred_test_base[idx_dia]
    
    # ── Construir features para cada hora del día D ──────────
    rows_features = lapply(seq_along(horas_dia), function(j) {
      h = horas_dia[j]
      
      hist_h = buffer %>%
        filter(hora == h, as.Date(fecha, format = "%Y%m%d") < fecha_d_date) %>%
        arrange(fecha)
      
      e_h = hist_h$error
      n_h = length(e_h)
      
      gl <- function(k) if (n_h >= k) e_h[n_h - k + 1] else NA_real_
      gm <- function(k) if (n_h >= k) mean(tail(e_h, k)) else NA_real_
      gv <- function(k) if (n_h >= k) var(tail(e_h, k))  else NA_real_
      
      data.frame(
        hora        = h,
        dia_semana  = as.integer(format(fecha_d_date, "%u")),
        mes         = as.integer(format(fecha_d_date, "%m")),
        dia_anyo    = as.integer(format(fecha_d_date, "%j")),
        e_lag1      = gl(1),  e_lag2  = gl(2),  e_lag3  = gl(3),
        e_lag4      = gl(4),  e_lag5  = gl(5),  e_lag6  = gl(6),
        e_lag7      = gl(7),  e_lag14 = gl(14), e_lag21 = gl(21),
        e_lag28     = gl(28), e_lag35 = gl(35), e_lag42 = gl(42),
        e_ma7       = gm(7),  e_ma14  = gm(14), e_ma28  = gm(28),
        e_var7      = gv(7)
      )
    })
    
    features_dia = bind_rows(rows_features) %>% anyadir_encodings()
    
    # ── Predecir error con el modelo pooled ───────────────────
    pred_error_dia = predict(modelo_lm2, newdata = features_dia)
    
    pred_final[idx_dia] = pred_base_dia + pred_error_dia
    pred_error[idx_dia] = pred_error_dia
    
    # ── Actualizar buffer con el error OLS REAL del día D ─────
    buffer <- bind_rows(buffer, data.frame(
      fecha = as.character(datos_test$fecha[idx_dia]),
      hora  = horas_dia,
      error = datos_test$demanda[idx_dia] - pred_base_dia
    ))
  }
  
  return(list(pred_final = pred_final, pred_error = pred_error))
}

# ── DIAGNÓSTICO DE RESIDUOS ETAPA 2 ──────────────────────────

diagnostico_lm2 = function(datos_test, pred_base, pred_final,
                            modelo_lm2, lags_max = 48) {
  
  res_base  = datos_test$demanda - pred_base
  res_final = datos_test$demanda - pred_final
  
  cat("\n── Diagnóstico de residuos ──\n")
  cat("Media res. base:   ", round(mean(res_base,  na.rm = TRUE), 6), "\n")
  cat("Media res. LM2:    ", round(mean(res_final, na.rm = TRUE), 6), "\n")
  cat("SD res. base:      ", round(sd(res_base,    na.rm = TRUE), 6), "\n")
  cat("SD res. LM2:       ", round(sd(res_final,   na.rm = TRUE), 6), "\n")
  
  par(mfrow = c(2, 2))
  acf(res_base,  lag.max = lags_max, main = "ACF residuos OLS base",  na.action = na.pass)
  acf(res_final, lag.max = lags_max, main = "ACF residuos OLS + LM2", na.action = na.pass)
  pacf(res_base,  lag.max = lags_max, main = "PACF residuos OLS base",  na.action = na.pass)
  pacf(res_final, lag.max = lags_max, main = "PACF residuos OLS + LM2", na.action = na.pass)
  par(mfrow = c(1, 1))
  
  invisible(list(lb_base = lb_base, lb_final = lb_final))
}


# ============================================================
# EJECUCIÓN COMPLETA
# ============================================================
pred_train_base = fitted(modelo)
pred_test_base = predict(modelo, newdata = datos_test)

cat("── Modelo base (Etapa 1) ──\n")
cat("MSE  base:", round(MSE(datos_test$demanda, pred_test_base), 7), "\n")
cat("RMSE base:", round(RMSE(datos_test$demanda, pred_test_base), 5), "\n")
cat("MAPE base:", round(MAPE(datos_test$demanda, pred_test_base), 4), "%\n\n")

# ── Paso 1: features de train ────────────────────────────────
cat("── Construyendo features de train...\n")
et_train      = datos_train$demanda - pred_train_base
df_feat_train = construir_features_train(datos_train, et_train)

df_feat_clean = df_feat_train %>%
  filter(complete.cases(.[, feat_con_NA]))

cat(sprintf("Obs. train disponibles: %d / %d\n",
            nrow(df_feat_clean), nrow(df_feat_train)))

# ── Paso 2: entrenar el modelo pooled de residuos ────────────
cat("\n── Entrenamiento LM POOLED sobre residuos...\n")
modelo_lm2 = entrenar_lm_residuos(df_feat_clean)

# ── Paso 3: predicción rolling en test ───────────────────────
cat("\n── Predicción rolling en test (2022)...\n")
res_lm2 = predecir_rolling_lm(
  datos_train     = datos_train,
  datos_test      = datos_test,
  pred_train_base = pred_train_base,
  pred_test_base  = pred_test_base,
  modelo_lm2      = modelo_lm2,
  verbose         = TRUE
)
pred_lm2 = res_lm2$pred_final

# ── Paso 4: métricas ─────────────────────────────────────────
cat("\n── Métricas en TEST (2022) ──\n")
cat("MSE  base:  ", round(MSE(datos_test$demanda, pred_test_base), 7), "\n")
cat("MSE  OLS+LM:", round(MSE(datos_test$demanda, pred_lm2),       7), "\n")
cat("Mejora MSE: ", round((1 - MSE(datos_test$demanda, pred_lm2) /
                             MSE(datos_test$demanda, pred_test_base)) * 100, 2), "%\n\n")
cat("RMSE base:  ", round(RMSE(datos_test$demanda, pred_test_base), 5), "\n")
cat("RMSE OLS+LM:", round(RMSE(datos_test$demanda, pred_lm2),       5), "\n\n")
cat("MAPE base:  ", round(MAPE(datos_test$demanda, pred_test_base), 4), "%\n")
cat("MAPE OLS+LM:", round(MAPE(datos_test$demanda, pred_lm2),       4), "%\n")

# ── Paso 5: diagnóstico ──────────────────────────────────────
diag_lm2 = diagnostico_lm2(
  datos_test = datos_test,
  pred_base  = pred_test_base,
  pred_final = pred_lm2,
  modelo_lm2 = modelo_lm2
)