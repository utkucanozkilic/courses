# Amaç: Özellikleri verildiğinde kişilerin diyabet hastası olduğunun/olmadığının tahminini yapacak model

# Veri seti ABD'deki Ulusal Diyabet-Sindirim-Böbrek Hastalıkları Enstitüleri'nde tutulan büyük veri setinin
# parçasıdır. ABD'deki Arizona Eyaleti'nin en büyük 5. şehri olan Phoenix şehrinde yaşayan 21 yaş ve üzerinde olan
# Pima Indian kadınları üzerinde yapılan diyabet araştırması için kullanılan verilerdir. 768 gözlem ve 8 sayısal
# bağımsız değişkenden oluşmaktadır. Hedef değişken "outcome" olarak belirtilmiş olup; 1 diyabet test sonucunun
# pozitif oluşunu, 0 ise negatif oluşunu belirtmektedir.

# Değişkenler
# Pregnancies: Hamilelik sayısı
# Glucose: Glikoz.
# BloodPressure: Kan basıncı.
# SkinThickness: Cilt Kalınlığı
# Insulin: İnsülin.
# BMI: Beden kitle indeksi.
# DiabetesPedigreeFunction: Soyumuzdaki kişilere göre diyabet olma ihtimalimizi hesaplayan bir fonksiyon.
# Age: Yaş (yıl)
# Outcome: Kişinin diyabet olup olmadığı bilgisi. Hastalığa sahip (1) ya da değil (0)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import train_test_split, cross_validate


def outlier_threshold(dataframe, column, first_percent = 0.25, third_percent = 0.75):
    q1 = dataframe[column].quantile(first_percent)
    q3 = dataframe[column].quantile(third_percent)
    iqr = q3 - q1
    up_limit = q3 + 1.5 * iqr
    low_limit = q1 - 1.5 * iqr

    return low_limit, up_limit


def check_outlier(dataframe, column, first_percent = 0.25, third_percent = 0.75):
    q1 = dataframe[column].quantile(first_percent)
    q3 = dataframe[column].quantile(third_percent)
    iqr = q3 - q1
    up_limit = q3 + 1.5 * iqr
    low_limit = q1 - 1.5 * iqr

    return dataframe[(dataframe[column] < low_limit) | (dataframe[column] > up_limit)].any(axis = None)


def replacement_with_thresholds(dataframe, column):
    low_limit, up_limit = outlier_threshold(dataframe, column)
    dataframe.loc[(dataframe[column] < low_limit), column] = low_limit
    dataframe.loc[(dataframe[column] > up_limit), column] = up_limit


def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")


pd.set_option('display.width', 1881)
pd.set_option('display.max_columns', 1881)

df = pd.read_csv(r"C:\Users\Souljah_Pc\PycharmProjects\courses\Machine Learning\datasets\diabetes.csv")

# Hedef Değişken Analizleri:
df['Outcome'].value_counts()

sns.countplot(x = 'Outcome', data = df)
plt.show()

df['Outcome'].value_counts() / len(df)

############
cols = [col for col in df.columns if col not in ['Outcome']]
for col in cols:
    target_summary_with_num(df, 'Outcome', col)

# Robust Scaler:
# Herbir gözlem değerinden median'ı çıkartıyor ve range değerine bölüyor. Aykırı değ.'den az etkileniyor.
df.info()
for col in cols:
    scaler = RobustScaler()
    df[col] = scaler.fit_transform(df[[col]])

# Modelleme ve Tahmin
y = df['Outcome']
X = df.drop('Outcome', axis = 1)

log_model = LogisticRegression()
log_model.fit(X, y)

log_model.intercept_
log_model.coef_

y_pred = log_model.predict(X)


# Model Evaluation
def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot = True, fmt = ".0f")
    plt.xlabel('y_pred')
    plt.ylabel('y')
    plt.title('Accuracy Score: {0}'.format(acc), size = 10)
    plt.show()


plot_confusion_matrix(y, y_pred)
# Tüm sonuçlara detaylı erişmek:
print(classification_report(y, y_pred))

# ROC AUC
y_prob = log_model.predict_proba(X)[:, 1]
roc_auc_score(y, y_prob)


# Model Validation: Holdout (Training/Testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 17)

log_model = LogisticRegression()
log_model.fit(X_train, y_train)

y_pred = log_model.predict(X_test)
y_prob = log_model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))

RocCurveDisplay.from_estimator(log_model, X_test, y_test)
plt.title('ROC CURVE')
plt.plot([0, 1], [0, 1], 'r--')
plt.show()

# AUC
roc_auc_score(y_test, y_prob)

# K-Fold CV
y = df['Outcome']
X = df.drop('Outcome', axis = 1)

log_model = LogisticRegression()
log_model.fit(X, y)

# cross_val_score'a göre aynı anda 1'den fazla metriğe göre başarı döndürebilir:
cv_results = cross_validate(log_model, X, y, cv = 5, scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'])

cv_results['test_accuracy'].mean()

cv_results['test_precision'].mean()

cv_results['test_recall'].mean()

cv_results['test_f1'].mean()

cv_results['test_roc_auc'].mean()

# Prediction for A New Observation
random_user = X.sample(1, random_state = 45)
log_model.predict(random_user)
log_model.predict_proba(random_user)