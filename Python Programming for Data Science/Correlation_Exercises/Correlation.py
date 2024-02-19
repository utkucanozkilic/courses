import numpy as np
import pandas as pd


def correlated_cols(dataframe, plot = False, corr_th = 0.9):
    # Seperate numeric and non-numeric columns
    numerical_columns = [col for col in dataframe.columns if dataframe[col].dtype in ["int64", "float64"]]
    non_numerical_columns = [col for col in dataframe.columns if col not in numerical_columns]

    # Drop non-numeric columns
    dataframe.drop(non_numerical_columns, axis = 1, inplace = True)

    # Create correlation matrix by taking absolute
    correlation_matrix = dataframe.corr().abs()

    # Select and create triangle matrix
    upper_triangle_matrix = correlation_matrix.where(
        np.triu(np.ones(shape = correlation_matrix.shape), k = 1).astype(bool)
        )

    # Create the drop list that will drop from correlation matrix by using threshold
    will_drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]

    # Apply correlation matrix
    correlation_matrix.drop(will_drop_list, axis = 1, inplace = True)
    correlation_matrix.drop(will_drop_list, axis = 0, inplace = True)

    # Plot correlation matrix
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc = {'figure.figsize': (15, 15)})
        print(len(correlation_matrix.columns))
        sns.heatmap(correlation_matrix, cmap = 'RdBu')
        plt.show()


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

df = pd.read_csv(
    "breast_cancer.csv"
    )

df = df.iloc[:, 1:-1]

correlated_cols(df, plot = True)
