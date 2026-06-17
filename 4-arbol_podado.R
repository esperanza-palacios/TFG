# ============================================================
# ÁRBOL DE REGRESIÓN PODADO
# Features: regresores externos + 24 dummies hora + lags
# ============================================================

library(dplyr)
library(tidyr)
library(rpart)
library(rpart.plot)

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
MAPE <- function(real, pred) mean(abs((real - pred) / real)) * 100

# ── 8. ÁRBOL COMPLETO (base para la poda) ────────────────────
set.seed(321)
cat("Ajustando árbol base...\n")

arbol_full <- rpart(
  demanda ~ .,
  data    = datos_train,
  method  = "anova",
  control = rpart.control(
    minsplit  = 20,
    minbucket = 10,
    cp        = 0.0001,
    maxdepth  = 30
  )
)

# ── 9. SELECCIÓN DEL CP ÓPTIMO ───────────────────────────────
# El cp óptimo es el que minimiza el error de validación cruzada (xerror)
printcp(arbol_full)

cp_optimo <- arbol_full$cptable[
  which.min(arbol_full$cptable[, "xerror"]), "CP"
]
cat("CP óptimo:", round(cp_optimo, 6), "\n")

# Gráfico del error de CV en función del cp
plotcp(arbol_full)

# ── 10. PODA ─────────────────────────────────────────────────
arbol_podado <- prune(arbol_full, cp = cp_optimo)

cat("Nodos árbol completo:", sum(arbol_full$frame$var  == "<leaf>"), "\n")
cat("Nodos árbol podado:  ", sum(arbol_podado$frame$var == "<leaf>"), "\n")

# ── 11. PREDICCIÓN Y MÉTRICAS ────────────────────────────────
pred_podado <- predict(arbol_podado, newdata = datos_test)

cat("\n── Árbol podado — TEST 2022 ──\n")
cat("MSE :", round(MSE(datos_test$demanda, pred_podado), 7), "\n")
cat("RMSE:", round(RMSE(datos_test$demanda, pred_podado), 5), "\n")
cat("MAPE:", round(MAPE(datos_test$demanda, pred_podado), 4), "%\n")
#MSE : 0.0020751 
#RMSE: 0.04555 
#MAPE: 0.9934 %

# ── 12. VISUALIZACIÓN ────────────────────────────────────────
rpart.plot(arbol_podado, type = 4, extra = 101,
           main = "Árbol de regresión podado")

# ── 13. DIAGNÓSTICO DE RESIDUOS ──────────────────────────────
et_podado <- datos_test$demanda - pred_podado

par(mfrow = c(1, 2))
acf(et_podado,  lag.max = 48, main = "ACF residuos — Árbol podado")
pacf(et_podado, lag.max = 48, main = "PACF residuos — Árbol podado")
par(mfrow = c(1, 1))

lb <- Box.test(et_podado, lag = 24, type = "Ljung-Box")
cat("Ljung-Box: p =", round(lb$p.value, 4), "\n")
cat(ifelse(lb$p.value > 0.05,
           "✓ Residuos sin autocorrelación",
           "✗ Autocorrelación detectada"), "\n")