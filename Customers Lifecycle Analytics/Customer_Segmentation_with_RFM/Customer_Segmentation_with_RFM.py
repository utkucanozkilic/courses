# CUSTOMER SEGMANTATION WITH RFM

# 1. Business Problem
# 2. Data Understanding
# 3. Data Preparation
# 4. Calculating RFM Metrics
# 5. Calculating RFM Scores
# 6. Creating & Analysing RFM Segments
# 7. Functionalization all process


# 1) A company wants to segment own customers and determine marketing strategies according to their segments.

# Attributes
#
# Invoice: Unique number for every order. If start with 'C' it's a canceled transaction.
# StockCode: Unique number for every product.
# Description: Name of product
# Quantity:
# InvoiceDate: Date of invoice
# Price: by Sterlin
# Customer ID: Unique customer number
# Country: The country that the customer lives

# 2) Data Understanding

import datetime as dt
import re

import pandas as pd

pd.set_option('display.max_columns', 1881)
pd.set_option('display.width', 1881)

# Functionalization all process

# For this dataset, the below code is out of function
df = pd.read_excel("online_retail_II.xlsx", sheet_name = "Year 2009-2010")

df = df[~df["Invoice"].str.contains("C", na = False)]


def create_rfm(dataframe):

    if dataframe.isnull().values.any():
        dataframe.dropna(inplace = True)

    dataframe["Total_Price"] = dataframe["Quantity"] * dataframe["Price"]

    today_date = dt.datetime(2011, 12, 11)

    rfm = dataframe.groupby("Customer ID").agg(
        {
            "InvoiceDate": lambda x: (today_date - x.max()).days,
            "Invoice": lambda x: x.nunique(),
            "Total_Price": lambda x: x.sum()
            }
        )

    rfm.columns = ["Recency", "Frequency", "Monetary"]

    rfm["Recency_Score"] = pd.qcut(rfm["Recency"], q = 5, labels = [5, 4, 3, 2, 1])
    rfm["Frequency_Score"] = pd.qcut(rfm["Frequency"].rank(method = "first"), q = 5, labels = [1, 2, 3, 4, 5])
    rfm["Monetary_Score"] = pd.qcut(rfm["Monetary"], q = 5, labels = [1, 2, 3, 4, 5])

    rf_values = ["hibernating", "at risk", "can't loose them", "about to sleep", "need attention",
                 "loyal customers", "promising", "potential loyallists", "new customers", "champions"]
    rf_keys = [r"[12][12]", r"[12][34]", r"[12][5]", r"[3][12]", r"[3][3]",
               r"[34][45]", r"[4][1]", r"[45][23]", r"[5][1]", r"[5][45]"]
    rf_labels = dict(zip(rf_keys, rf_values))

    rfm["RFM_Score"] = rfm["Recency_Score"].astype(str) + rfm["Frequency_Score"].astype(str)

    rfm["Segment"] = rfm["RFM_Score"].replace(rf_labels, regex = True)

    return rfm


def save_customer_ids_by_segments(dataframe):
    rfm = dataframe[["Recency", "Frequency", "Monetary", "Segment"]]
    rfm.index = rfm.index.astype("int64")
    path = "Customer_Segmentation_with_RFM/"
    rfm.to_csv(path + "Customers_and_Segments.csv")


save_customer_ids_by_segments(create_rfm(df))
