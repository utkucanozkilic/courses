import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import math
import scipy.stats as stats

pd.set_option('display.max_columns', 1881)
pd.set_option('display.width', 1881)

df = pd.read_csv("/Measurement Problems/Sorting Products/product_sorting.csv")

df.head()

# rating'e göre sıralama
df[["rating"]].sort_values(ascending = False, by = "rating")

# satın alınma sayılarına göre sıralama
df.sort_values(by = "purchase_count", ascending = False)

# yorum sayılarına göre sıralama
df.sort_values(by = "comment_count", ascending = False)


# oy puanı, yorum sayısı ve satın alıma göre sıralama
def min_max_scaler(dataframe, column_will_scale, scaled_column_name, min_value = 0, max_value = 1):
    scaler = MinMaxScaler((min_value, max_value))
    scaler.fit(df[[column_will_scale]])
    dataframe[scaled_column_name] = scaler.transform(df[[column_will_scale]])


min_max_scaler(df, "commment_count", "comment_count_scaled", 1, 5)
min_max_scaler(df, "purchase_count", "purchase_count_scaled", 1, 5)


def weighted_score(dataframe, weighted_score_column_name, **kwargs):
    # shape(len(df), 1), tüm değerleri 0 olan sütun df'e eklenir
    dataframe[weighted_score_column_name] = pd.DataFrame(np.zeros((len(dataframe), 1)))

    # argümanlardan gelen sütundaki değerler, ağırlıklarıyla çarpılır ve ilgili sütuna eklenir
    for key, value in kwargs.items():
        dataframe[weighted_score_column_name] += dataframe[key] * value

    return dataframe[weighted_score_column_name]


weighted_score(
    df, "weighted_score", comment_count_scaled = 0.32, purchase_count_scaled = 0.26,
    rating = 0.42
    )

df.sort_values(by = "weighted_score", ascending = False)


# Bayesian Average Rating Score
def bayesian_average_rating(n, confidence = 0.95):
    if sum(n) == 0:
        return 0
    K = len(n)
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    N = sum(n)
    first_part = 0.0
    second_part = 0.0
    for k, n_k in enumerate(n):
        first_part += (k + 1) * (n[k] + 1) / (N + K)
        second_part += (k + 1) * (k + 1) * (n[k] + 1) / (N + K)
    score = first_part - z * math.sqrt((second_part - first_part * first_part) / (N + K + 1))
    return score


df["bar_score"] = df.apply(
    lambda x: bayesian_average_rating(
        x[["1_point",
           "2_point",
           "3_point",
           "4_point",
           "5_point"]]
        ), axis = 1
    )

df.sort_values(by = "bar_score", ascending = False)


# Hybrid Soritng(BAR + Diğer faktörler)
def hybid_score(dataframe, weight_bar = 0.6, weight_ws = 0.4):
    return (dataframe.apply(
        lambda x: bayesian_average_rating(
            x[["1_point",
               "2_point",
               "3_point",
               "4_point",
               "5_point"]]
            ), axis = 1
        ) * weight_bar +
            weighted_score(
                dataframe, "weighted_score",
                comment_count_scaled = 0.32,
                purchase_count_scaled = 0.26,
                rating = 0.42
                ) * weight_ws)


df["hybid_score"] = hybid_score(df)
df.sort_values(by = "hybid_score", ascending = False).head(20)