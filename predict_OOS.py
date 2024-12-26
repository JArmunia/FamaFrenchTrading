import sys
import os
from pathlib import Path
import pandas as pd
from scipy.stats import spearmanr
import lightgbm as lgb
import numpy as np
from tqdm import tqdm
import argparse
from utils import MultipleTimeSeriesCV

import warnings

warnings.filterwarnings("ignore")

np.random.seed(42)
idx = pd.IndexSlice


parser = argparse.ArgumentParser(description="Predicción OOS")
parser.add_argument(
    "--data_store",
    type=Path,
    default=Path("data/assets_neutralized.h5"),
    help="Ruta al archivo de datos",
)
parser.add_argument(
    "--data_store_item",
    type=str,
    default="engineered_features_trimmed",
    help="Nombre del item en el archivo de datos",
)
parser.add_argument(
    "--results_path",
    type=Path,
    default=Path("results", "us_stocks"),
    help="Ruta para guardar resultados",
)
parser.add_argument(
    "--predictions_store",
    type=Path,
    default=Path("data/prueba_predictions.h5"),
    help="Ruta para guardar predicciones",
)

args = parser.parse_args()

DATA_STORE = args.data_store
DATA_STORE_ITEM = args.data_store_item
RESULTS_PATH = args.results_path
PREDICTIONS_STORE = args.predictions_store

YEAR = 52
YEARS_OOS = 4.9
LOOKAHEAD = 1
N_BEST_MODELS = 10
LABEL = "target_1w"

CATEGORICALS = [
    "sector",
]

scope_params = ["lookahead", "train_length", "test_length"]
daily_ic_metrics = [
    "daily_ic_mean",
    "daily_ic_mean_n",
    "daily_ic_median",
    "daily_ic_median_n",
]
lgb_train_params = [
    "learning_rate",
    "num_leaves",
    "feature_fraction",
    "min_data_in_leaf",
]


base_params = dict(boosting="gbdt", objective="regression", random_state=42, verbose=-1)


def get_lgb_params(data, t=5, best=0):
    param_cols = scope_params[1:] + lgb_train_params + ["boost_rounds"]
    df = data[data.lookahead == t].sort_values("ic", ascending=False).iloc[best]
    return df.loc[param_cols]


# ### Get Data


def get_data(data_store, data_store_item, categoricals, label):
    data = pd.read_hdf(data_store, data_store_item).sort_index()  # modificado

    labels = sorted(data.filter(like="target").columns)
    features = data.columns.difference(labels).tolist()

    # completamos con los valores del periodo anterior, para evitar que el último dato apareza nan
    data = data.fillna(method="ffill")

    # datos desde 2010
    data = data.loc[idx[:, "2010":], features + [LABEL]].dropna()

    for feature in CATEGORICALS:
        data[feature] = pd.factorize(data[feature], sort=True)[0]

    return data, features


def get_lgb_key(t, p):
    key = f"{t}/{int(p.train_length)}/{int(p.test_length)}/{p.learning_rate}/"
    return key + f"{int(p.num_leaves)}/{p.feature_fraction}/{int(p.min_data_in_leaf)}"


def get_lgb_daily_ic(results_path):
    int_cols = ["lookahead", "train_length", "test_length", "boost_rounds"]

    lgb_ic = []
    with pd.HDFStore(results_path / "tuning_lgb.h5") as store:
        keys = [k[1:] for k in store.keys()]
        for key in keys:
            _, t, train_length, test_length = key.split("/")[:4]
            if key.startswith("daily_ic"):
                df = (
                    store[key]
                    .drop(["boosting", "objective", "verbose"], axis=1)
                    .assign(
                        lookahead=t, train_length=train_length, test_length=test_length
                    )
                )
                lgb_ic.append(df)
        lgb_ic = pd.concat(lgb_ic).reset_index()

    id_vars = ["date"] + scope_params + lgb_train_params
    lgb_ic = pd.melt(
        lgb_ic, id_vars=id_vars, value_name="ic", var_name="boost_rounds"
    ).dropna()
    lgb_ic.loc[:, int_cols] = lgb_ic.loc[:, int_cols].astype(int)

    lgb_daily_ic = (
        lgb_ic.groupby(id_vars[1:] + ["boost_rounds"])
        .ic.mean()
        .to_frame("ic")
        .reset_index()
    )
    return lgb_daily_ic


def save_best_predictions_train(
    results_path, predictions_store, lgb_daily_ic, lookahead, topn
):
    lookahead = 1
    topn = 10
    for best in range(topn):
        best_params = get_lgb_params(lgb_daily_ic, t=lookahead, best=best)
        key = get_lgb_key(lookahead, best_params)
        rounds = str(int(best_params.boost_rounds))
        if best == 0:
            best_predictions = pd.read_hdf(
                results_path / "tuning_lgb.h5", "predictions/" + key
            )
            best_predictions = best_predictions[rounds].to_frame(best)
        else:
            best_predictions[best] = pd.read_hdf(
                results_path / "tuning_lgb.h5", "predictions/" + key
            )[rounds]
    best_predictions = best_predictions.sort_index()
    best_predictions.to_hdf(predictions_store, f"lgb/train/{lookahead:02}")


def train_model(lgb_data, data, params, n_best_models=10):
    # for par las 10 mejores configuracones de paramentros de las cuales almacenaremos sus predicciones
    for position in range(n_best_models):
        params = get_lgb_params(lgb_daily_ic, t=LOOKAHEAD, best=position)

        params = params.to_dict()  # parametros a diccionario

        for p in ["min_data_in_leaf", "num_leaves"]:
            params[p] = int(params[p])
        train_length = int(
            params.pop("train_length")
        )  # Extrae y elimina el parámetro 'train_length' del diccionario de parámetros y lo convierte a un entero
        test_length = int(params.pop("test_length"))
        num_boost_round = int(params.pop("boost_rounds"))
        params.update(base_params)

        print(f"\nPosition: {position:02}")

        # 1-year out-of-sample period
        # vamos a ir haciendo el walk forward con periodos de test de un mes, moveremos el modelo para volver a entrenar y predeciremos el siguiente mes
        n_splits = int(YEAR * YEARS_OOS / test_length)
        cv = MultipleTimeSeriesCV(
            n_splits=n_splits,
            test_period_length=test_length,
            lookahead=LOOKAHEAD,
            train_period_length=train_length,
        )

        predictions = []
        for i, (train_idx, test_idx) in tqdm(enumerate(cv.split(X=data), 1)):
            # Crea un conjunto de datos de entrenamiento para LightGBM
            lgb_train = lgb_data.subset(
                used_indices=train_idx.tolist(), params=params
            ).construct()
            # Entrena el modelo LightGBM
            model = lgb.train(
                params=params,
                train_set=lgb_train,
                num_boost_round=num_boost_round,
            )

            test_set = data.iloc[test_idx, :]
            y_test = test_set.loc[:, LABEL].to_frame("y_test")
            # Realiza predicciones en el conjunto de datos de prueba
            y_pred = model.predict(test_set.loc[:, model.feature_name()])
            predictions.append(y_test.assign(prediction=y_pred))

        if position == 0:
            test_predictions = pd.concat(predictions).rename(
                columns={"prediction": position}
            )
        else:
            test_predictions[position] = pd.concat(predictions).prediction

    by_day = test_predictions.groupby(level="date")  # Agrupa las predicciones por fecha
    for position in range(n_best_models):
        # Si es la primera iteración, calcula el coeficiente de correlación de Spearman
        # entre las predicciones y las etiquetas verdaderas y lo almacena en `ic_by_day`
        if position == 0:
            ic_by_day = by_day.apply(
                lambda x: spearmanr(x.y_test, x[position])[0]
            ).to_frame()
        else:
            ic_by_day[position] = by_day.apply(
                lambda x: spearmanr(x.y_test, x[position])[0]
            )

    return test_predictions


if __name__ == "__main__":
    data, features = get_data(DATA_STORE, DATA_STORE_ITEM, CATEGORICALS, LABEL)

    lgb_data = lgb.Dataset(
        data=data[features],
        label=data[LABEL],
        categorical_feature=CATEGORICALS,
        free_raw_data=False,
    )

    lgb_daily_ic = get_lgb_daily_ic(RESULTS_PATH)

    save_best_predictions_train(
        RESULTS_PATH, PREDICTIONS_STORE, lgb_daily_ic, LOOKAHEAD, 10
    )

    params = get_lgb_params(
        lgb_daily_ic,
        t=LOOKAHEAD,
    )

    test_predictions = train_model(lgb_data, data, params, N_BEST_MODELS)
    test_predictions.to_hdf(PREDICTIONS_STORE, f"lgb/test/{LOOKAHEAD:02}")
