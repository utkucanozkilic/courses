# Principal Component Analysis
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

pd.set_option('display.width', 1881)
pd.set_option('display.max_columns', 1881)


df = pd.read_csv(r'C:\Users\Souljah_Pc\PycharmProjects\courses\Machine Learning\datasets\hitters.csv')

# Veriseti etiketli. Etiketi ve kategorik verileri eliyoruz:
num_cols = [col for col in df.columns if df[col].dtypes not in ['object'] and 'Salary' not in col]

df = df[num_cols]

df.dropna(inplace = True)

# Standartlaştırma gerekir:
sc = StandardScaler()
sc.fit_transform(df)

pca = PCA()
pca_fit = pca.fit_transform(df)

# Oluşturulan bileşenlerin bilgi açıklama oranları:
pca.explained_variance_ratio_
# Kümülatif şekilde:
np.cumsum(pca.explained_variance_ratio_)

# Optimum Bileşen Sayısı (Elbow Yöntemi ile)
pca = PCA()
pca.fit(df)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Bileşen Sayısı')
plt.ylabel('Kümülatif Varyans Oranı')
plt.show()

# Final PCA'in Oluşturulması (bileşen sayısını belirledik)
pca = PCA(n_components = 3)
pca_fit = pca.fit_transform(df)

pca.explained_variance_ratio_
np.cumsum(pca.explained_variance_ratio_)


# Principal Component Regression
# Varsayımlar:
# Veri setini doğrusal model ile modellemek istiyoruz. Değişkenler arasında çoklu doğrusal bağlantı problemi var.
# Değişkenler arasında yüksek korelasyon var.

df = pd.read_csv(r'C:\Users\Souljah_Pc\PycharmProjects\courses\Machine Learning\datasets\hitters.csv')

num_cols = [col for col in df.columns if df[col].dtypes not in ['object'] and 'Salary' not in col]

others = [col for col in df.columns if col not in num_cols]

# Değişkenler arasındaki yüksek korelasyonu kırdık, yeni değişkenlerle yeni df'i oluşturduk:
final_df = pd.concat([pd.DataFrame(pca_fit, columns = ['PC1', 'PC2', 'PC3']), df[others]], axis = 1)

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

binary_columns = ['NewLeague', 'Division', 'League']
for col in binary_columns:
    labelencoder = LabelEncoder()
    final_df[col] = labelencoder.fit_transform(final_df[col])

final_df.dropna(inplace = True)

X = final_df.drop('Salary', axis = 1)
y = final_df['Salary']

# Regression Model:
lm = LinearRegression()

rmse = np.mean(np.sqrt(-cross_val_score(lm, X, y, cv = 5, scoring = 'neg_mean_squared_error')))
# rmse ile ortalama kıyaslaması:
rmse, y.mean()


# Decision Tree Regression Model:
cart = DecisionTreeRegressor()

rmse = np.mean(np.sqrt(-cross_val_score(cart, X, y, cv = 5, scoring = 'neg_mean_squared_error')))

# DT için hiperparametre optimizasyonu:
cart_params = {
    'max_depth': range(1, 11),
    'min_samples_split': range(2, 20)
    }

cart_best_grid = GridSearchCV(cart, cart_params, cv = 5, n_jobs = -1, verbose =True)
cart_best_grid.fit(X, y)

cart_final = DecisionTreeRegressor(**cart_best_grid.best_params_, random_state = 17)
cart_final.fit(X, y)

rmse = np.mean(np.sqrt(-cross_val_score(cart_final, X, y, cv = 5, scoring = 'neg_mean_squared_error')))


# PCA ile Çok Boyutlu Veriyi 2 Boyutta Görselleştirme
df = pd.read_csv(r'C:\Users\Souljah_Pc\PycharmProjects\courses\Machine Learning\datasets\breast_cancer.csv')

X = df.drop(['diagnosis', 'id'], axis = 1)
y = df['diagnosis']


def create_pca_df(X, y):
    s_scaler = StandardScaler()
    X = s_scaler.fit_transform(X)
    pca = PCA(n_components = 2)
    pca = pca.fit_transform(X)
    pca_df = pd.DataFrame(data = pca, columns = ['PC1', 'PC2'])
    final_df = pd.concat([pca_df, pd.DataFrame(y)], axis = 1)
    return final_df


pca_df = create_pca_df(X, y)


# Dersten alındı
def plot_pca(dataframe, target):
    fig = plt.figure(figsize = (7, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('PC1', fontsize = 15)
    ax.set_ylabel('PC2', fontsize = 15)
    ax.set_title(f'{target.capitalize()} ', fontsize = 20)

    targets = list(dataframe[target].unique())
    colors = random.sample(['r', 'b', "g", "y"], len(targets))

    for t, color in zip(targets, colors):
        indices = dataframe[target] == t
        ax.scatter(dataframe.loc[indices, 'PC1'], dataframe.loc[indices, 'PC2'], c = color, s = 50)
    ax.legend(targets)
    ax.grid()
    plt.show()


plot_pca(pca_df, 'diagnosis')


# Iris veri setine uyarlama:
import seaborn as sns
df = sns.load_dataset('iris')

X = df.drop(['species'], axis = 1)
y = df['species']

pca_df = create_pca_df(X, y)
plot_pca(pca_df, 'species')