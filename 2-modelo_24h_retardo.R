#modelo 24h retardo
library(dplyr)
library(stats)
d=read.csv("datos_input.csv")


retardos = c(1,2,3,4,5,6,7,14,21,28,35,42)

d_retardos = d

for (h in 1:24) {
  
  h_str = sprintf("%02d", h)
  var_y= paste0("dem_h", h_str)
  
  for (lag_i in retardos) {
    new_name = paste0(var_y, "_lag", lag_i)
    d_retardos[[new_name]] = lag(d_retardos[[var_y]], lag_i)
  }
}


año = substr(d_retardos$fecha, 1, 4)

pos_train = (año >= "2010" & año <= "2021")
pos_test  = (año == "2022")

datos_train =d_retardos[pos_train, ]
datos_test =d_retardos[pos_test, ]


cols_comunes = 26:317     # regresores comunes
n_lags = 12               # retardos por hora

modelos = list()
preds   = list()
mse     = numeric(24)
rmse <- numeric(24)
mape <- numeric(24)
source("funciones/MSE.R")

#bucle por horas

for (h in 1:24) {
  
  h_str = sprintf("%02d", h)
  y_var = paste0("dem_h", h_str)
  
  # columnas de retardos EXACTAS de la hora h
  ini_lag = 318 + (h - 1) * n_lags
  fin_lag = ini_lag + n_lags - 1
  
  regresores_h = names(d_retardos)[c(cols_comunes, ini_lag:fin_lag)]
  
  formula_h = as.formula(
    paste(y_var, "~", paste(regresores_h, collapse = " + "))
  )
  
  # modelo
  modelos[[h_str]] =lm(formula_h, data = datos_train)
  
  # predicción
  preds[[h_str]] = predict(modelos[[h_str]], newdata = datos_test)
  
  
  real_log  <- datos_test[[y_var]]
  pred_log  <- preds[[h_str]]
  # error
  mse[h] = MSE(datos_test[[y_var]], preds[[h_str]])
  rmse[h] <- sqrt(mse[h])                             # en escala log
  mape[h] <- mean(abs((exp(real_log) - exp(pred_log)) /
                        exp(real_log))) * 100          # en escala original (%)
}


mse_horas = data.frame(
  hora = sprintf("%02d", 1:24),
  MSE  = mse
)

print(mse_horas)
mean(mse)
metricas <- data.frame(
  hora = sprintf("%02d", 1:24),
  MSE  = round(mse,  6),
  RMSE = round(rmse, 5),
  MAPE = round(mape, 4)
)
print(metricas)

cat("\n--- GLOBALES (media 24 horas) ---\n")
cat("MSE  (log):      ", round(mean(mse),  6), "\n")
cat("RMSE (log):      ", round(mean(rmse), 5), "\n")
cat("MAPE (%):        ", round(mean(mape), 4), "\n")
#diagnostico para una hora

summary(modelos[["01"]])

par(mfrow = c(2,2))
plot(modelos[["01"]])


et = residuals(modelos[["01"]])
par(mfrow = c(1,2))
acf(et, lag.max = 40,  main = "ACF residuos — h01 (modelo con retardos)")


pacf(et, lag.max = 20,  main = "PACF residuos — h01 (modelo con retardos)")


