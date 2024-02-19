import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

df = sns.load_dataset("titanic")
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 500)


# # Print some important knowledges of DataFrame:
# def df_check(dataframe, head = 10):
#     print("************************HEAD************************")
#     print(dataframe.head(head))
#     print("************************TAIL************************")
#     print(dataframe.tail(head))
#     print("THE SHAPE OF DF:", dataframe.shape)
#     print("************************THE INFO OF DF************************")
#     print(dataframe.info())
#     print("COLUMNS OF DF:", dataframe.columns)
#     print("THE INDEX OF DF:", dataframe.index)
#     print("************************THE DESCRIBE OF DF************************")
#     print(dataframe.describe())
#     if dataframe.isnull().values.any():
#         print("The dataframe has NaN values")
#     else:
#         print("The dataframe has not NaN values")
#     print("************************NUMBER OF NULL VALUES OF DF************************")
#     print(dataframe.isnull().sum())
#     print(dataframe.describe([0, 0.15, 0.35, 0.95, 1]).T)
# df_check(df)


# Choose categorical features
# def choose_cat_columns(dataframe):
#     print("All categories:{}".format(dataframe.columns))
#     print("**************************************")
#     cat_col = [col for col in dataframe.columns if dataframe[col].dtype in ["category", "object", "bool"]]
#
#     cat_col_but_num = [col for col in dataframe.columns if dataframe[col].dtype in ["int64", "float64"]
#                        and dataframe[col].nunique() < 10]
#
#     cardinal_cols = [col for col in dataframe.columns if dataframe[col].dtype in ["category"]
#                      and dataframe[col].nunique() > 20]
#     cat_col = cat_col + cat_col_but_num
#
#     if cardinal_cols:
#         [cat_col.pop(cat_col.index(col)) for col in cardinal_cols if col in cat_col]
#
#     return cat_col

# cat_col = choose_cat_columns(df)

# num_col = [col for col in df.columns if col not in cat_col]


# # Print percentiles of feature and plot same feature
# def cat_summary(dataframe, col_name, plot = False):
#     print(
#         pd.DataFrame({col_name: dataframe[col_name].value_counts(),
#                       "Ratio": (dataframe[col_name].value_counts() / dataframe[col_name].count()) * 100
#                       }
#                      )
#         )
#     print("****************************************")
#
#     if plot:
#         if dataframe[col_name].dtypes == "bool":
#             dataframe[col_name] = dataframe[col_name].astype("int")
#         sns.countplot(data = dataframe, x = dataframe[col_name])
#         plt.show()
#
# for col in df.columns:
#     cat_summary(df, col, plot = True)


# Print describes of feature and plot same feature
def num_summary(dataframe, numerical_columns, percentiles_list = [0.25, 0.5, 0.75], plot = False):

    print(dataframe[[numerical_columns]].describe(percentiles = percentiles_list).T)

    if plot:
        sns.histplot(data = dataframe, x = numerical_columns)
        plt.title("Histogram of 'AGE")
        plt.xlabel(numerical_columns)
        plt.ylabel("Frequency")
        plt.show()


# Filter features by numerical and categorical
def grab_col_names(dataframe, categorical_threshold = 10, cardinal_threshold = 20):

    cat_cols = [col for col in dataframe.columns if dataframe[col].dtype
                in ["category", "object", "bool"]]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].dtype
                   in ["int64", "float64"] and dataframe[col].nunique() < categorical_threshold]

    cat_but_car = [col for col in dataframe.columns if dataframe[col].dtype
                   in ["category"] and dataframe[col].nunique() > cardinal_threshold]

    cat_cols = cat_cols + num_but_cat
    num_cols = [col for col in dataframe.columns if col not in cat_cols]

    if cat_but_car:
        [cat_cols.pop(cat_cols.index(col)) for col in cat_but_car if col in cat_cols]
        [num_cols.pop(num_cols.index(col)) for col in cat_but_car if col in cat_cols]

    print("Observation: {}" .format(dataframe.shape[0]))
    print("Features: {}" .format(dataframe.shape[1]))
    print("Categorical cols: {}" .format(len(cat_cols)))
    print("Numerical cols: {}" .format(len(num_cols)))
    print("Categorical but cardinal cols: {}" .format(len(cat_but_car)))
    print("Numerical but categorical cols: {}" .format(len(num_but_cat)))

    return cat_cols, num_cols, categorical_threshold, num_but_cat


# summary target by categorical features
def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({
        "MEAN_OF_TARGET": dataframe.groupby(categorical_col)[target].mean()
        }))


def target_summary_with_num(dataframe, target, numerical_col):
    # print(pd.DataFrame({
    #     "MEAN_OF_NUM_COL": dataframe.groupby(target)[numerical_col].mean()
    #     }))
    print(pd.DataFrame(dataframe.groupby(target).agg({numerical_col: "mean"})))


target_summary_with_cat(df, "survived", "sex")
target_summary_with_num(df, "survived", "age")
