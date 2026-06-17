# ============================================================
# RED NEURONAL MLP 
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


d = pd.read_csv("datos_input.csv")

cols_demanda       = [c for c in d.columns if c.startswith("dem_h")]
regresores_externos = d.columns[25:317].tolist()


cols_no_demanda = [c for c in d.columns if c not in cols_demanda]

d_long = d.melt(
    id_vars    = cols_no_demanda,
    value_vars = cols_demanda,
    var_name   = "hora_str",
    value_name = "demanda"
)

d_long["hora"] = (
    d_long["hora_str"]
    .str.replace("dem_h", "", regex=False)
    .astype(int)
)

d_long = d_long.drop(columns=["hora_str"])

# ── construcción de retardos ─────────────────────────────────────────

d_long = (
    d_long
    .sort_values(["hora", "fecha"])
    .reset_index(drop=True)
)

lags_diarios = [1, 2, 3, 4, 5, 6, 7, 14, 21, 28, 35, 42]

for lag in lags_diarios:
    d_long[f"demanda_lag{lag}"] = (
        d_long
        .groupby("hora")["demanda"]
        .shift(lag)
    )

d_long = d_long.dropna().reset_index(drop=True)

# ── Codificación cíclica de la hora ─────────────────────────────────────────

d_long["hora_sin"] = np.sin(2 * np.pi * d_long["hora"] / 24)
d_long["hora_cos"] = np.cos(2 * np.pi * d_long["hora"] / 24)

# ── Definición de features ─────────────────────────────────────────

lags_names = [f"demanda_lag{l}" for l in lags_diarios]

# 292 regresores externos + 2 cíclicas + 12 retardos = 306
features = regresores_externos + ["hora_sin", "hora_cos"] + lags_names

print(f"Total features: {len(features)}")

# ── Partición temporal train/val/test ─────────────────────────────────────────

anio = d_long["fecha"].astype(str).str[:4]

train = d_long[(anio >= "2010") & (anio <= "2020")]
val   = d_long[anio == "2021"]
test  = d_long[anio == "2022"]

X_train = train[features].values
y_train = train["demanda"].values

X_val   = val[features].values
y_val   = val["demanda"].values

X_test  = test[features].values
y_test  = test["demanda"].values

print(
    f"Train: {len(X_train):,} | "
    f"Val: {len(X_val):,} | "
    f"Test: {len(X_test):,}"
)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

# ── entrenamiento con 10 réplicas─────────────────────────────────────────

import os
os.environ["PYTHONHASHSEED"] = "0"

seeds  = range(42, 52)
n_runs = len(seeds)

resultados = []
modelo_ref = None
history_ref = None
pred_ref    = None

for i, s in enumerate(seeds):

    print(f"\n══ Réplica {i+1}/{n_runs} (seed={s}) ══")

    # Fijar todas las semillas
    np.random.seed(s)
    tf.random.set_seed(s)

    # Construir modelo desde cero
    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(256, activation="relu"),
        BatchNormalization(),
        Dropout(0.1),
        Dense(128, activation="relu"),
        BatchNormalization(),
        Dropout(0.1),
        Dense(64, activation="relu"),
        Dense(1)
    ])

    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4),
        loss      = "mse",
        metrics   = ["mae"]
    )

    early_stop = EarlyStopping(
        monitor="val_loss", patience=20,
        restore_best_weights=True, verbose=0
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.5,
        patience=5, min_lr=1e-6, verbose=0
    )

    history = model.fit(
        X_train, y_train,
        validation_data = (X_val, y_val),
        epochs          = 200,
        batch_size      = 128,
        callbacks       = [early_stop, reduce_lr],
        verbose         = 0
    )

    # Predicción
    pred = model.predict(X_test, verbose=0).flatten()

    # Métricas
    mse_i  = mean_squared_error(y_test, pred)
    rmse_i = np.sqrt(mse_i)
    real_orig = np.exp(y_test)
    pred_orig = np.exp(pred)
    mape_i = np.mean(np.abs((real_orig - pred_orig) / real_orig)) * 100

    resultados.append({
        "seed": s, "MSE": mse_i,
        "RMSE": rmse_i, "MAPE": mape_i,
        "epochs": len(history.history["loss"])
    })

    print(f"   MSE={mse_i:.7f}  RMSE={rmse_i:.5f}  MAPE={mape_i:.4f}%  ({len(history.history['loss'])} épocas)")

    # Guardar modelo de referencia (seed=42) para análisis posterior
    if s == 42:
        modelo_ref  = model
        history_ref = history
        pred_ref    = pred


df_res = pd.DataFrame(resultados)

print("\n" + "=" * 60)
print("RESULTADOS MLP: 10 RÉPLICAS (seeds 42–51)")
print("=" * 60)
print(df_res.to_string(index=False))
print(f"\nMSE  : {df_res['MSE'].mean():.7f} ± {df_res['MSE'].std():.7f}")
print(f"RMSE : {df_res['RMSE'].mean():.5f} ± {df_res['RMSE'].std():.5f}")
print(f"MAPE : {df_res['MAPE'].mean():.4f}% ± {df_res['MAPE'].std():.4f}%")

# ── curva de aprendizaje modelo seed 42─────────────────────────────────────────

plt.figure(figsize=(9, 4))
plt.plot(history_ref.history["loss"],     label="Train")
plt.plot(history_ref.history["val_loss"], label="Validación 2021", linestyle="--")
plt.yscale("log")
plt.xlabel("Época")
plt.ylabel("MSE (escala log)")
plt.title("Curva de aprendizaje — Red Neuronal MLP (seed = 42)")
plt.legend()
plt.tight_layout()
plt.savefig("curva_aprendizaje_nn.png", dpi=150)
plt.show()

# ── métricas por hora modelo seed 42────────────────────────────────────────

resultados_hora       = test.copy()
resultados_hora["pred"] = pred_ref

def mse_h(x):
    return mean_squared_error(x["demanda"], x["pred"])

def mape_h(x):
    r = np.exp(x["demanda"].values)
    p = np.exp(x["pred"].values)
    return np.mean(np.abs((r - p) / r)) * 100

metricas_hora = (
    resultados_hora
    .groupby("hora")
    .apply(lambda x: pd.Series({
        "MSE" : mse_h(x),
        "RMSE": np.sqrt(mse_h(x)),
        "MAPE": mape_h(x)
    }))
    .reset_index()
)

print("\n── Métricas por hora — seed=42 (test 2022) ──")
print(metricas_hora.to_string(index=False))



plt.figure(figsize=(9, 4))
plt.bar(metricas_hora["hora"], metricas_hora["MSE"], color="steelblue")
plt.xlabel("Hora del día")
plt.ylabel("MSE")
plt.title("MSE por hora — Red Neuronal MLP (test 2022, seed = 42)")
plt.xticks(range(1, 25))
plt.tight_layout()
plt.savefig("mse_por_hora_nn.png", dpi=150)
plt.show()

# ── guardar resultados─────────────────────────────────────────

df_res.to_csv("resultados_replicas_mlp.csv", index=False)
print("\nResultados guardados en: resultados_replicas_mlp.csv")