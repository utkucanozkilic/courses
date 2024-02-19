# ASSOCIATION RULE LEARNING

# 1) Data Preprocessing
# 2) Preparation ARL data structe (Invoice-Product Matrix)
# 3) get association rules
# 4) Preparation the script of work
# 5) product recommendation in cart


#################################
# 1. Veri Önişleme
# Attributes
# Invoice: Fatura Numarası (C ile başlayanlar, iptal olanlar)
# StockCode: Ürün Kodu
# Description: Ürün ismi
# Quantity: Ürün adedi
# InvoiceDate: Fatura Tarihi
# UnitPrice: Fatura fiyatı(sterlin)
# CustomerID: Eşsiz müşteri numarası
# Country


import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

pd.set_option('display.width', 1881)
pd.set_option('display.max_columns', 1881)

df_ = pd.read_excel(r"C:\Users\Souljah_Pc\PycharmProjects\Recommendation Systems\online_retail_II.xlsx",
                    sheet_name = "Year 2010-2011")
df = df_.copy()


def outlier_threshold(dataframe, variable):
    quantile_1 = dataframe[variable].quantile(0.01)
    quantile_3 = dataframe[variable].quantile(0.99)
    interquantile_range = quantile_3 - quantile_1
    up_bound = quantile_3 + 1.5 * interquantile_range
    down_bound = quantile_1 - 1.5 * interquantile_range
    return up_bound, down_bound


def replace_with_thresholds(dataframe, variable):
    up_bound, down_bound = outlier_threshold(dataframe, variable)
    dataframe.loc[(dataframe[variable] > up_bound), variable] = up_bound
    dataframe.loc[(dataframe[variable] < down_bound), variable] = down_bound


def retail_data_prep(dataframe):
    dataframe.dropna(inplace = True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na = False)]
    dataframe = dataframe[(dataframe["Quantity"] > 0) & (dataframe["Price"] > 0)]
    replace_with_thresholds(dataframe, variable = "Quantity")
    replace_with_thresholds(dataframe, variable = "Price")
    return dataframe


def arl_matrix(dataframe, value = False):
    if value:
        return dataframe.groupby(["Invoice", "Description"]).agg({"Quantity": "sum"}).unstack().fillna(0) \
            .map(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(["Invoice", "StockCode"])["Quantity"].sum().unstack().fillna(0) \
            .map(lambda x: 1 if x > 0 else 0)


def check_id(dataframe, stock_code):
    return dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()


def create_rules(dataframe, country, value = False):
    dataframe = dataframe[dataframe["Country"] == country]
    dataframe = arl_matrix(dataframe, value)
    frequent_itemsets = apriori(dataframe, min_support = 0.01, use_colnames = True)
    rules = association_rules(frequent_itemsets, metric = "support", min_threshold = 0.01)
    return rules

# Çok tatlı bir çözüm:
# def zero_to_boolean(dataframe):
#     return (dataframe != 0).astype(int)


df = retail_data_prep(df)


# Örnek - Sepette Ürün Önerme

# # Neden çalışmadığını anlamadığım çözümüm
# for antecedent in df_rules["antecedents"]:
#     for element in list(antecedent):
#         if element == product_id:
#             recommend_list.append(df_rules.loc[(df_rules["antecedents"] == antecedent)]["consequents"])


def arl_recommender(dataframe, product_id, country, rec_count = 1):
    dataframe_sorted_rules = create_rules(dataframe, country).sort_values("lift", ascending = False)
    recommend_list = []

    for i, product in enumerate(dataframe_sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
                recommend_list.append(list(dataframe_sorted_rules.iloc[i]["consequents"])[0])
                continue
    return recommend_list[0: rec_count]


arl_recommender(df, 22492, "France")