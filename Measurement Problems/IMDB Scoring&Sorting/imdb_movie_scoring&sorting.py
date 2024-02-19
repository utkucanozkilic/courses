import pandas as pd
import math
import scipy.stats as stats
from sklearn.preprocessing import MinMaxScaler

pd.set_option("display.max_columns", 1881)
pd.set_option("display.width", 1881)

df = pd.read_csv(
    "/Measurement Problems/IMDB Scoring&Sorting/movies_metadata.csv",
    low_memory = False
    )

df = df[["title", "vote_average", "vote_count"]]

df["vote_count"].describe([0.1, 0.25, 0.5, 0.8, 0.9, 0.95, 0.99])

df[df["vote_count"] > 400].describe([0.1, 0.25, 0.5, 0.8, 0.9, 0])

scaler = MinMaxScaler((1, 10))
scaler.fit(df[["vote_count"]])
df["vote_count_score"] = scaler.transform(df[["vote_count"]])

df["average_count_score"] = df["vote_average"] * df["vote_count_score"]

df.sort_values(by = "vote_count_score", ascending = False)

# IMBD Weighted Rating
# weighted_rating = (v / (v + M) * r) + (M / (v + M) * C)  (up to 2015)

# r = vote average
# v = vote count
# M = minimum votes required to be listed in the Top 250
# C = the mean vote across the whole reportt (currently 7.0)

M = 2500
C = df["vote_average"].mean()


def weighted_rating(r, v, M, C):
    return (v / (v + M) * r) + (M / (v + M) * C)


df.sort_values(by = "average_count_score", ascending = False).head(20)

df["weighted_rating"] = df.apply(lambda x: weighted_rating(x["vote_average"], x["vote_count"], M, C), axis = 1)

df.sort_values(by = "weighted_rating", ascending = False).head(20)


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


df = pd.read_csv("IMDB Scoring&Sorting/imdb_ratings.csv")
df = df.iloc[0:, 1:]

df["bar_score"] = df.apply(
    lambda x: bayesian_average_rating(
        x[["one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten"]]
        ), axis = 1
    )

