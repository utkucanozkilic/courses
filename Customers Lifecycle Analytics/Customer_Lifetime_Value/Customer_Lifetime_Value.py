# CUSTOMER LIFETIME VALUE

# 1. Data Preperation
# 2. Average Order Value (average_order_value = total_price / total_transaction)
# 3. Purchase Frequency (total_transaction / total_number_of_customers)
# 4. Repeat Rate & Churn Rate (birden fazla alışveriş yapan müşteri sayısı / tüm müşteriler)
# 5. Profit Margin (profit_margin =  total_price * 0.10)
# 6. Customer Value (customer_value = average_order_value * purchase_frequency)
# 7. Customer Lifetime Value (CLTV = (customer_value / churn_rate) x profit_margin)
# 8. Define Segments


import pandas as pd

pd.set_option('display.max_columns', 1881)
pd.set_option('display.width', 1881)
pd.options.display.float_format = '{:.6f}'.format

df_ = pd.read_excel("online_retail_II.xlsx", sheet_name = "Year 2009-2010")
df = df_.copy()


# Data Preperation
def data_cleaning(dataframe):

    if dataframe.isnull().values.any():
        dataframe.dropna(inplace = True)

    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na = False)]

    if len(dataframe[dataframe["Quantity"] <= 0]) > 1:
        dataframe = dataframe[dataframe["Quantity"] > 0]

    if len(dataframe[dataframe["Price"] <= 0]) > 1:
        dataframe = dataframe[dataframe["Price"] > 0]

    dataframe["Total_Price"] = dataframe["Quantity"] * dataframe["Price"]

    return dataframe


def get_av_ord_val_per_cus(dataframe):

    dataframe = dataframe.groupby("Customer ID").agg({
        "Invoice": lambda x: x.nunique(),
        "Quantity": lambda x: x.sum(),
        "Total_Price": lambda x: x.sum()
        })

    dataframe.columns = ["Total_Transaction", "Unit_Quantity", "Total_Price"]

    dataframe["avarage_order_value"] = dataframe["Total_Price"] / dataframe["Total_Transaction"]

    return dataframe


def get_purchase_frequency(dataframe):

    dataframe["Purchase_Frequency"] = dataframe["Total_Transaction"] / len(dataframe)

    return dataframe


def get_churn_rate(dataframe):

    return 1 - (len(dataframe[dataframe["Total_Transaction"] > 1]) / len(dataframe))


def get_profit_margin(dataframe, profit_rate = 0.1):

    dataframe["Profit_Margin"] = dataframe["Total_Price"] * profit_rate

    return dataframe


def get_customer_value(dataframe):

    dataframe["Customer_Value"] = dataframe["avarage_order_value"] * dataframe["Purchase_Frequency"]

    return dataframe


def get_clt_value(dataframe):

    dataframe["CLT_Value"] = (dataframe["Customer_Value"] / churn_rate) * dataframe["Profit_Margin"]

    return dataframe


def get_segmented_cltv(dataframe, segments = 4):

    dataframe["Segments"] = pd.qcut(dataframe["CLT_Value"], segments, labels = ["D", "C", "B", "A"])

    return dataframe


df = data_cleaning(df)

cltv_c = get_av_ord_val_per_cus(df)

cltv_c = get_purchase_frequency(cltv_c)

churn_rate = get_churn_rate(cltv_c)

cltv_c = get_profit_margin(cltv_c)

cltv_c = get_customer_value(cltv_c)

cltv_c = get_clt_value(cltv_c)

cltv_c.sort_values(by = ["CLT_Value"], ascending = True)

get_segmented_cltv(cltv_c)
cltv_c.groupby("Segments").agg({"count", "mean", "sum"})