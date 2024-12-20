import pandas as pd
import pandas_datareader.data as web

# replaces pyfinance.ols.PandasRollingOLS (no longer maintained)
from statsmodels.regression.rolling import RollingOLS
import statsmodels.api as sm
import numpy as np
import os
import dotenv
import yfinance as yf

from fred_client import FREDClient

dotenv.load_dotenv(".env.local")

idx = pd.IndexSlice

normaliza = True  # normalizamos por volatilidad
neutraliza = False  # normalizado cross sectional media y vola

DATA_STORE = "data/assets.h5"
START = 2000
END = 2025

TICKER_LIST = [
    "XLE",
    "XLB",
    "XLI",
    "XLK",
    "XLF",
    "XLP",
    "XLY",
    "XLV",
    "XLU",
    "IYR",
    "VOX",
]

FACTORS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]

INDICATORS = [
        "JHDUSRGDPBR",
        "T10Y3M",
        "BAMLC0A0CM",
        "BAMLH0A0HYM2",
        "BAMLHE00EHYIOAS",
        "UMCSENT",
        "UNRATE",
        "GDPC1",
        "DCOILWTICO",
        "CORESTICKM159SFRBATL",
        "USSLIND",
        "VIXCLS",
        "ICSA",
        "MARTSMPCSM44000USS",
        "RSXFS",
        "DGS1",
        ##########################
        "TOTBKCR",
        "BUSLOANS",
        "WPU08",
        "PCOTTINDUSDM",
        "PWHEAMTUSDM",
        "PMAIZMTUSDM",
        "PCOFFOTMUSDM",
        "PNRGINDEXM",
        "PCOPPUSDM",
        "PNGASEUUSDM",
        "BUSINV",
        "CP",
        "PCU33443344",
        "CPIAUCSL",
        "M2SL",
        "REAINTRATREARAT10Y",
        "HOUST",
        "CUSR0000SERA02",
        "IP7108",
    ]  # ,'OVXCLS'
INDICATOR_NAMES = [
        "recession",
        "yield_curve",
        "corp_oas",
        "hy_oas",
        "eu_hy_oas",
        "sentiment",
        "empleo",
        "real_gdp",
        "oil",
        "inflacion",
        "leading",
        "vix",
        "weekjobclaims",
        "retail_sales_percent",
        "retail_sales",
        "1y_yield",
        #####################
        "tot_bank_credit",
        "commercial_industrial_loans",
        "lumber",
        "cotton",
        "wheat",
        "corn",
        "coffee",
        "energy_price",
        "copper",
        "natural_gas",
        "business_inventory",
        "corporate_profits",
        "semiconductor_electronics_manufacturing",
        "consumer_price_index",
        "M2_money_supply",
        "10y_real_interest_rate",
        "new_homes",
        "streaming_media_consumption",
        "gold",
    ]  # ,'vixoil'



def repair_VOX(data):

    # Descargamos QQQ para reemplazar VOX antes de 2004-09-28
    qqq = yf.download(
        tickers=["QQQ"],
        group_by="ticker",
        start=f"{START}-01-01",
        end=f"{END}-12-31",
    )["QQQ"]

    vox = data[["VOX"]].copy()
    cutoff_date = "2004-09-28"
    data = data.drop(columns=["VOX"])

    vox.loc[pd.IndexSlice[:cutoff_date, "VOX"]] = qqq.loc[
        pd.IndexSlice[:cutoff_date]
    ].values
    data = pd.concat((data, vox), axis=1)
    return data


def repair_IYR(data):
    iyr = data[["IYR"]].copy()
    cutoff_date = "2000-06-18"
    data = data.drop(columns=["IYR"])
    xlf = data["XLF"].copy()

    iyr.loc[pd.IndexSlice[:cutoff_date, "IYR"]] = xlf.loc[
        pd.IndexSlice[:cutoff_date]
    ].values
    data = pd.concat((data, iyr), axis=1)
    return data


def download_data(ticker_list):

    data = yf.download(
        # passes the ticker
        tickers=ticker_list,
        # used for access data[ticker]
        group_by="ticker",
        start=f"{START}-01-01",
        end=f"{END}-12-31",
    )

    data = repair_VOX(data)
    data = repair_IYR(data)

    data.index = data.index.tz_localize(None)
    data = data.stack(-2)
    data = data.rename_axis(["date", "ticker"])

    # Reordenar y renombrar las columnas directamente
    new_order = ["Open", "High", "Low", "Close", "Volume", "Adj Close"]
    new_names = ["open", "high", "low", "close", "volume", "Adj Close"]

    # Reordenar las columnas
    prices = data[new_order]

    # Renombrar las columnas
    prices.columns = new_names
    prices = prices.sort_index()

    # Crear un nuevo DataFrame sin entradas duplicadas en el índice
    prices = prices.loc[~prices.index.duplicated(keep="first")]
    prices = prices.swaplevel(0, 1)
    prices = prices.sort_index(level=list(range(len(prices.index.names))))
    return prices


def create_weekly_returns(prices, start, end, column="close", lags=None):
    if lags is None:
        lags = [1, 2, 3, 6, 12, 52]

    prices = prices.loc[idx[:, str(start) : str(end)], column].unstack("ticker")
    weekly_prices = prices.resample("W").last()
    outlier_cutoff = 0.01
    data = pd.DataFrame()
    for lag in lags:
        data[f"return_{lag}w"] = (
            weekly_prices.pct_change(lag)
            .stack()
            .pipe(
                lambda x: x.clip(
                    lower=x.quantile(outlier_cutoff),
                    upper=x.quantile(1 - outlier_cutoff),
                )
            )
            .add(1)
            .pow(1 / lag)
            .sub(1)
        )
    data = data.swaplevel().dropna()
    return data


def drop_stocks_with_less_than_n_years_of_returns(returns, n_years=10):
    min_obs = 52 * n_years  # semanal
    nobs = returns.groupby(level="ticker").size()
    keep = nobs[nobs > min_obs].index

    returns = returns.loc[idx[keep, :], :]
    return returns


def save_data(data, name):
    with pd.HDFStore(DATA_STORE) as store:
        store.put(name, data.sort_index())
        print(store.info())


def normalize(data, lags=None):
    if lags is None:
        lags = [1, 2, 3, 6, 12, 52]

    def _normalize_by_rolling_std(series):
        return series / series.rolling(52).std().shift(1)

    for lag in lags:
        data[f"return_{lag}w"] = data.groupby(level="ticker")[f"return_{lag}w"].transform(
            _normalize_by_rolling_std
        )
    return data


def neutralize(data, lags=None):
    if lags is None:
        lags = [1, 2, 3, 6, 12, 52]

    def _neutralize(group):
        return (group - group.mean()) / group.std()

    for lag in lags:
        data[f"return_{lag}w"] = data.groupby(level="date")[f"return_{lag}m"].transform(
            _neutralize
        )
    return data


def get_factor_betas(weekly_returns, reg_window=52 * 2):
    factor_data = web.DataReader(
        "F-F_Research_Data_5_Factors_2x3", "famafrench", start=str(START)
    )[0].drop("RF", axis=1)
    factor_data.index = factor_data.index.to_timestamp()
    factor_data = factor_data.resample("W").last().div(100)  # semanal

    factor_data.index.name = "date"
    factor_data = factor_data.join(weekly_returns["return_1w"]).sort_index()

    T = reg_window  # 2 años en semanas
    betas = factor_data.groupby(level="ticker", group_keys=False).apply(
        lambda x: RollingOLS(
            endog=x.return_1w,
            exog=sm.add_constant(x.drop("return_1w", axis=1)),
            window=min(T, x.shape[0] - 1),
        )
        .fit(params_only=True)
        .params.drop("const", axis=1)
    )

    return betas


def add_momentum_factor(data, lags=None):
    if lags is None:
        lags = [2, 3, 6, 12, 52]

    for lag in lags:
        data[f"momentum_{lag}"] = data[f"return_{lag}w"].sub(data.return_1w)
    data["momentum_3_12"] = data["return_12w"].sub(data.return_3w)

    return data


def add_date_indicators(data):
    dates = data.index.get_level_values("date")
    data["year"] = dates.year
    data["month"] = dates.month
    # data["week"] = dates.isocalendar().week.values.astype(int)
    return data


def add_sector(data):
    # Crear una Serie con el índice del DataFrame y los valores del nivel 'ticker'
    ticker_series = pd.Series(data.index.get_level_values("ticker"), index=data.index)

    # Usar esta Serie para llenar los valores NA
    data["sector"] = ticker_series
    return data


def add_lagged_returns(data, lags=None):
    if lags is None:
        lags = [1, 2, 3, 4, 5, 6]
    for t in lags:
        data[f"return_1w_t-{t}"] = data.groupby(level="ticker").return_1w.shift(t)
    return data


def add_targets(data, weeks=None):
    if weeks is None:
        weeks = [1, 2, 3, 6, 12]
    for t in weeks:
        data[f"target_{t}w"] = data.groupby(level="ticker")[f"return_{t}w"].shift(-t)
    return data


def add_fred_data(data):

    data_fred = (
        web.DataReader(INDICATORS, "fred", START, END + 1)
        .ffill()
        .resample("W")
        .last()
        .dropna()
    )

    data_fred.columns = INDICATOR_NAMES
    data_fred.index.name = "date"
    for columna in data_fred.columns:
        data_fred[columna + "_diff"] = data_fred[columna].diff()

    # eliminamos algunas variables que tienen mucha dependencia del nivel historico
    data_fred = data_fred.drop(["empleo",], axis=1)
    data = data.join(data_fred)
    return data


def main():
    prices = download_data(TICKER_LIST)
    save_data(prices, "data_close")
    weekly_returns = create_weekly_returns(prices, START, END)

    # save_data(weekly_returns, "data_close")

    weekly_returns = drop_stocks_with_less_than_n_years_of_returns(weekly_returns, 10)

    save_data(weekly_returns, "data_raw")

    if normaliza:
        weekly_returns = normalize(weekly_returns)

    if neutraliza:
        weekly_returns = neutralize(weekly_returns)

    factor_betas = get_factor_betas(weekly_returns)

    weekly_returns = weekly_returns.join(factor_betas.groupby(level="ticker").shift())

    # Imputamos los valores faltantes por la media de la columna
    weekly_returns.loc[:, FACTORS] = weekly_returns.groupby("ticker")[
        FACTORS
    ].transform(lambda x: x.fillna(x.mean()))

    weekly_returns = add_momentum_factor(weekly_returns)
    weekly_returns = add_date_indicators(weekly_returns)
    weekly_returns = add_sector(weekly_returns)
    weekly_returns = add_lagged_returns(weekly_returns)
    weekly_returns = add_targets(weekly_returns)
    weekly_returns = add_fred_data(weekly_returns)

    save_data(weekly_returns, "engineered_features")

    # Realizamos PCA manteniendo 95% de la varianza
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    # Excluimos todos los targets
    features = weekly_returns.select_dtypes(include=["float64", "int64"]).drop(
        columns=[col for col in weekly_returns.columns if "target_" in col]
    )
    # Separamos los datos hasta 2020 para entrenamiento
    mask_2020 = features.index.get_level_values("date").year <= 2020
    features_train = features[mask_2020]

    # Estandarizamos los datos usando solo datos hasta 2020
    scaler = StandardScaler()
    scaler.fit(features_train)
    features_scaled = scaler.transform(features)

    # Aplicamos PCA usando solo datos hasta 2020
    pca = PCA(n_components=0.99)  # Mantener 95% de la varianza
    # Primero imputamos con el valor anterior
    features_scaled_train = pd.DataFrame(features_scaled[mask_2020]).fillna(
        method="ffill"
    )
    # Si quedan nulos, imputamos con la media
    features_scaled_train = features_scaled_train.fillna(features_scaled_train.mean())
    pca.fit(features_scaled_train)
    # Transformamos todos los datos con el PCA entrenado
    features_scaled = np.nan_to_num(
        features_scaled, nan=np.nanmean(features_scaled[mask_2020])
    )
    features_pca = pca.transform(features_scaled)

    # Convertimos a DataFrame
    pca_cols = [f"PC{i+1}" for i in range(features_pca.shape[1])]
    features_pca_df = pd.DataFrame(features_pca, index=features.index, columns=pca_cols)

    # Añadimos el target original
    features_pca_df["target_1w"] = weekly_returns["target_1w"]

    # Guardamos el resultado
    save_data(features_pca_df, "engineered_features_pca")

    print(features_pca_df.tail())
    print(features_pca_df.info())
    print(features_pca_df.shape)


if __name__ == "__main__":
    main()
