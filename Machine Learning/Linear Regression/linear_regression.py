# Sales Prediction with Linear Regression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', 1881)
pd.set_option('display.width', 1881)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score

# Simple Linear Regression with OLS Using Scikit-Learn

df = pd.read_csv(r"C:\Users\Souljah_Pc\PycharmProjects\courses\Machine Learning\datasets\advertising.csv")

X = df[['TV']]
y = df[['sales']]

# Model
reg_model = LinearRegression().fit(X, y)

reg_model.intercept_  # bias >>> 7.03259355]
reg_model.coef_  # w1 (tv'nin katsayısı) >>> 0.04753664]

# 150 birimlik TV harcaması olması durumunda beklenen satış:
reg_model.intercept_[0] + reg_model.coef_[0][0] * 150
reg_model.predict([[150]])[0][0]

# Modelin görselleştirilmesi:
g = sns.regplot(x = X, y = y, scatter_kws = {'color': 'b', 's': 9},
                ci = False, color = 'r')
g.set_title('Model Denklemi: Sales = {} + TV * {}' .format(round(reg_model.intercept_[0], 2),
                                                           round(reg_model.coef_[0][0], 2)))
g.set_ylabel('Satış Sayısı')
g.set_xlabel('TV Harcamaları')
plt.xlim(-10, 310)
plt.ylim(bottom = 0)  # y ekseni 0'dan başlasın.
plt.show()

# Modelin Tahmin Başarısı
y_pred = reg_model.predict(X)
mean_squared_error(y, y_pred)  # >>> 10.512652915656757 (MSE)
y.mean(), y.std()  # >>> 14.0225, 5.217457

np.sqrt(mean_squared_error(y, y_pred))  # >>> 3.2423221486546887 (RMSE)

mean_absolute_error(y, y_pred)  # >>> 2.549806038927486 (MAE)

# Bağımsız değişkenler bağımlı değişkenin %61'ini açıklayabilir.
# Değişken sayısı arttıkça R-kare şişer. Düzeltilmiş R-kare dikkate alınmalıdır.
reg_model.score(X, y)  # >>> 0.611875050850071 (R-Kare, Bağımsız değişkenlerin bağımlı değişkenleri açıklama yüzdesi.)


# Multiple Linear Regression (Çok Değişkenli)
df = pd.read_csv(r"C:\Users\Souljah_Pc\PycharmProjects\courses\Machine Learning\datasets\advertising.csv")
X = df.drop('sales', axis = 1)
y = df[['sales']]

# Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
reg_model = LinearRegression()
reg_model.fit(X_train, y_train)

reg_model.intercept_  # >>> 2.90794702
reg_model.coef_  # >>> 0.0468431 , 0.17854434, 0.00258619

# Tahmin Başarısı
# Train RMSE
y_pred = reg_model.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))  # >>>  1.7369025901470923

# Train R-Kare
reg_model.score(X_train, y_train)  # >>> 0.8959372632325174 Açıklama oranı %90'a ulaştı. Yeni özellik etkisi.

# Test RMSE
y_pred = reg_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))  # >>> 1.4113417558581587

# Test R-Kare
reg_model.score(X_test, y_test)  # >>> 0.8927605914615384

# 10-Fold Cross Validation RMSE
# Veri seti küçük olduğundan CV tüm veri setine uygulandı:
np.mean(np.sqrt(-cross_val_score(reg_model, X, y, cv = 10, scoring = 'neg_mean_squared_error')))  # > 1.6913531708051797


# Simple Linear Regression with Gradient Descent from Scratch (2 özellik olan basit doğrusal bağlanım)
def cost_function(Y, b, w, X):
    m = len(Y)  # Gözlem sayısı
    sse = 0  # Sum of Square Error
    for i in range(0, m):
        y_hat = b + w * X[i]
        y = Y[i]
        sse += (y_hat - y) ** 2

    mse = sse / m
    return mse


def update_weights(Y, b, w, X, learning_rate):
    m = len(Y)

    b_deriv_sum = 0
    w_deriv_sum = 0

    for i in range(0, m):
        y_hat = b + w * X[i]
        y = Y[i]
        b_deriv_sum += (y_hat - y)
        w_deriv_sum += (y_hat - y) * X[i]
    new_b = b - (learning_rate * 1 / m * b_deriv_sum)
    new_w = w - (learning_rate * 1 / m * w_deriv_sum)
    return new_b, new_w


def train(Y, initial_b, initial_w, X, learning_rate, num_iters):
    # İlk hatayı raporlama:
    print("Starting gradient descent at b = {}, w = {}, mse = {}" .format(initial_b, initial_w,
                                                                          cost_function(Y, initial_b, initial_w, X)))
    b = initial_b
    w = initial_w

    cost_history = []
    for i in range(num_iters):
        b, w = update_weights(Y, b, w, X, learning_rate)
        mse = cost_function(Y, b, w, X)
        cost_history.append(mse)

        if i % 100 == 0:
            print("iter = {:d}  b = {:.2f}  w = {:.4f}  mse = {:.4}" .format(i, b, w, mse))

    print("After {0} iterations b = {1}, w = {2}, mse = {3}".format(num_iters, b, w, cost_function(Y, b, w, X)))
    return cost_history, b, w