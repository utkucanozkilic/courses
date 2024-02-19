import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', 1881)
pd.set_option("display.width", 1881)
pd.options.display.float_format = '{:.6f}'.format

df_ = pd.read_excel(
    "C:/Users/Souljah_Pc/PycharmProjects/Customer_Lifecylce_Analytics/online_retail_II.xlsx",
    sheet_name = "Year 2009-2010"
    )
df = df_.copy()


def outlier_thresholds(dataframe, variable):

    quantile_1 = dataframe[variable].quantile(0.01)
    quantile_3 = dataframe[variable].quantile(0.99)

    interquantile_range = quantile_3 - quantile_1

    up_limit = quantile_3 + 1.5 * interquantile_range
    low_limit = quantile_1 - 1.5 * interquantile_range

    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):

    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


def create_cltv_p(dataframe, month):

    if dataframe.isnull().values.any():
        dataframe.dropna(inplace = True)

    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na = False)]

    if len(dataframe[dataframe["Quantity"] <= 0]) > 0:
        dataframe = dataframe[dataframe["Quantity"] > 0]

    if len(dataframe[dataframe["Price"] <= 0]) > 0:
        dataframe = dataframe[dataframe["Price"] > 0]

    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")

    dataframe["Total_Price"] = dataframe["Quantity"] * dataframe["Price"]

    today_date = dt.datetime(2011, 12, 11)

    cltv = dataframe.groupby("Customer ID").agg(
        {
            "InvoiceDate": [lambda x: (x.max() - x.min()).days,
                            lambda x: (today_date - x.min()).days],
            "Invoice": lambda x: x.nunique(),
            "Total_Price": lambda x: x.sum()
            }
        )

    cltv.columns = cltv.columns.droplevel(level = 0)
    cltv.columns = ["Recency", "T", "Frequency", "Monetary"]

    cltv["Monetary"] = cltv["Monetary"] / cltv["Frequency"]

    cltv = cltv[cltv["Frequency"] > 1]

    cltv["Recency"] = cltv["Recency"] / 7
    cltv["T"] = cltv["T"] / 7

    bgf = BetaGeoFitter(penalizer_coef = 0.001)
    bgf.fit(cltv['Frequency'], cltv['Recency'], cltv['T'])

    cltv["expected_purc_1_week"] = bgf.predict(
        1,
        cltv['Frequency'],
        cltv['Recency'],
        cltv['T']
        )

    cltv["expected_purc_2_week"] = bgf.predict(
        2,
        cltv['Frequency'],
        cltv['Recency'],
        cltv['T']
        )

    cltv["expected_purc_3_week"] = bgf.predict(
        3,
        cltv['Frequency'],
        cltv['Recency'],
        cltv['T']
        )

    ggf = GammaGammaFitter(penalizer_coef = 0.01)
    ggf.fit(cltv['Frequency'], cltv["Monetary"])
    cltv["expected_avarage_profit"] = ggf.conditional_expected_average_profit(cltv["Frequency"], cltv["Monetary"])

    cltv["cltv"] = ggf.customer_lifetime_value(
        bgf,
        cltv["Frequency"],
        cltv["Recency"],
        cltv["T"],
        cltv["Monetary"],
        time = month,
        freq = "W",  # T'nin frekans bilgisi
        discount_rate = 0.01
        )

    cltv.reset_index(inplace = True)

    cltv["segment"] = pd.qcut(cltv["cltv"], 4, labels = ["D", "C", "B", "A"])

    return cltv

print(create_cltv_p(df, 3))

