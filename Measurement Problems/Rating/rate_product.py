import pandas as pd
import math
import scipy.stats as stats
import datetime as dt

# (50+ Saat) Python A-Z™: Veri Bilimi ve Machine Learning
# Puan: 4.8 (4.764925)
# Toplam Puan: 4611
# Puan Yüzdeleri: 75, 20, 4, 1, <1
# Yaklaşık Sayısal Karşılıkları: 3458, 922, 184, 46, 6


# Rating,Timestamp,Enrolled,Progress,Questions Asked,Questions Answered

pd.set_option('display.width', 1881)
pd.set_option('display.max_columns', 1881)

df = pd.read_csv("/Measurement Problems/Rating/course_reviews.csv")

df.groupby("Questions Asked").agg(
    {
        "Questions Asked": "count",
        "Rating": "mean"
        }
    ).T

df["Rating"].mean()

# Time-Based Weighted Avarage

df["Timestamp"] = pd.to_datetime(df["Timestamp"])
current_day = pd.to_datetime("2021-02-10 0:0:0")
df["days"] = (current_day - df["Timestamp"]).dt.days

# Son 30 günde, günlere göre yapılan oylamaların ortalamaları
df[df["days"] <= 30].groupby("days")["Rating"].agg(["mean"])

# Son 30 günde yapılan oyların ortalamaları
df.loc[df["days"] <= 30, "Rating"].mean()

# 31-90 arası oylamaların ortalaması
df.loc[(df["days"] > 30) & (df["days"] <= 90), "Rating"].mean()

# 91- 180 arası oylamaların ortalaması
df.loc[(df["days"] > 91) & (df["days"] <= 180), "Rating"].mean()

# 180 günden önce yapılan oylamaların ortalaması
df.loc[df["days"] > 181, "Rating"].mean()

# ağırlıklı ortalama ile tüm oylamaların ortalaması
(df.loc[df["days"] <= 30, "Rating"].mean() * 0.28 +
 df.loc[(df["days"] > 30) & (df["days"] <= 90), "Rating"].mean() * 0.26 +
 df.loc[(df["days"] > 90) & (df["days"] <= 180), "Rating"].mean() * 0.24 +
 df.loc[df["days"] > 180, "Rating"].mean() * 0.22)


def time_based_weighted_average(dataframe, *weight):
    if not weight:
        weight = [0.25, 0.25, 0.25, 0.25]

    return (dataframe.loc[df["days"] <= 30, "Rating"].mean() * weight[0] +
            dataframe.loc[(df["days"] > 30) & (dataframe["days"] <= 90), "Rating"].mean() * weight[1] +
            dataframe.loc[(df["days"] > 90) & (dataframe["days"] <= 180), "Rating"].mean() * weight[2] +
            dataframe.loc[df["days"] > 180, "Rating"].mean() * weight[3])


time_based_weighted_average(df, 0.30, 0.26, 0.22, 0.22)


# User-Based Weighted Average
def user_based_weighted_average(dataframe, *weight):
    if not weight:
        weight = [0.25, 0.25, 0.25, 0.25]

    return (dataframe.loc[df["Progress"] <= 10, "Rating"].mean() * weight[0] +
            dataframe.loc[(df["Progress"] > 10) & (dataframe["Progress"] <= 45), "Rating"].mean() * weight[1] +
            dataframe.loc[(df["Progress"] > 45) & (dataframe["Progress"] <= 75), "Rating"].mean() * weight[2] +
            dataframe.loc[df["Progress"] > 75, "Rating"].mean() * weight[3])


user_based_weighted_average(df, 0.20, 0.24, 0.26, 0.30)


# Weighted Rating
def course_weighted_rating(dataframe, time_w = 0.5, user_w = 0.5):
    return (time_based_weighted_average(dataframe, 0.30, 0.26, 0.22, 0.22) *
            time_w + user_based_weighted_average(dataframe, 0.20, 0.24, 0.26, 0.30) * user_w)


course_weighted_rating(df)