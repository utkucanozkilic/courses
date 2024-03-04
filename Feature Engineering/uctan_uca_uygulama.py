import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler


def grab_col_names(dataframe, cat_th = 10, car_th = 20, info = False):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtype in ['object', 'category', 'bool']]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].dtype in ['int64', 'float64']
                   and dataframe[col].nunique() < cat_th]

    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th
                   and dataframe[col].dtype in ['object', 'category']]

    cat_cols = cat_cols + num_but_cat

    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtype in ['float64', 'int64']]

    num_cols = [col for col in num_cols if col not in num_but_cat]

    if not info:
        return cat_cols, num_cols, cat_but_car
    else:
        print("Observations: {}, Variables: {}" .format(dataframe.shape[0], dataframe.shape[1]))
        print("Caterogical columns:", len(cat_cols))
        print("Numerical columns:", len(num_cols))
        print('Caterogical but cardinal columns:', len(cat_but_car))
        print('Numerical but caterogical columns:', len(num_but_cat))

        return cat_cols, num_cols, cat_but_car


def check_outlier(dataframe, column, first_percent = 0.25, third_percent = 0.75):
    q1 = dataframe[column].quantile(first_percent)
    q3 = dataframe[column].quantile(third_percent)
    iqr = q3 - q1
    up_limit = q3 + 1.5 * iqr
    low_limit = q1 - 1.5 * iqr

    return dataframe[(dataframe[column] < low_limit) | (dataframe[column] > up_limit)].any(axis = None)


def outlier_threshold(dataframe, column, first_percent = 0.25, third_percent = 0.75):
    q1 = dataframe[column].quantile(first_percent)
    q3 = dataframe[column].quantile(third_percent)
    iqr = q3 - q1
    up_limit = q3 + 1.5 * iqr
    low_limit = q1 - 1.5 * iqr

    return low_limit, up_limit


def grab_outliers(dataframe, column, index = False):
    low, up = outlier_threshold(dataframe, column)

    if len(dataframe[(dataframe[column] < low) | (dataframe[column] > up)]) > 10:
        print(dataframe[(dataframe[column] < low) | (dataframe[column] > up)].head())
    else:
        print(dataframe[(dataframe[column] < low) | (dataframe[column] > up)])

    if index:
        return dataframe[(dataframe[column] < low) | (dataframe[column] > up)].index


def replacement_with_thresholds(dataframe, column):
    low_limit, up_limit = outlier_threshold(dataframe, column)
    dataframe.loc[(dataframe[column] < low_limit), column] = low_limit
    dataframe.loc[(dataframe[column] > up_limit), column] = up_limit


def missing_values_table(dataframe, na_name = False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending = False)
    ratio = (dataframe[na_columns].isnull().sum() / len(dataframe) * 100).sort_values(ascending = False)
    missin_df = pd.concat(objs = [n_miss, np.round(ratio, 2)], axis = 1, keys = ['n_miss', 'ratio'])

    print(missin_df)

    if na_name:
        return na_columns


def rare_analyser(dataframe, target, categorical_col):
    for col in categorical_col:
        print(col, ":", dataframe[col].nunique())
        print(
            pd.DataFrame(
                {
                    'COUNT': dataframe[col].value_counts(),
                    'RATIO(%)': dataframe[col].value_counts() / len(dataframe) * 100,
                    'TARGET_MEAN': dataframe.groupby(col)[target].mean()
                    }
                )
            )
        print("\n")


def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtype in ['object', 'category']
                    and ((temp_df[col].value_counts() / len(temp_df)) < rare_perc).any()]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df) * 100
        # Zaten yukarıda 'rare_perc'ten küçük olanları aldık. aşağıdaki işlem gereksiz:
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df


def one_hot_encoder(dataframe, categorical_cols, drop_first = True, dtype = 'int64'):
    dataframe = pd.get_dummies(dataframe, columns = categorical_cols, drop_first = drop_first, dtype = dtype)
    return dataframe


def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    # Eğer eksik değerler var ise bunları da sıradaki sayı ile doldurur.
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


pd.set_option('display.max_columns', 1881)
pd.set_option('display.width', 1881)

df = pd.read_csv(r'C:\Users\Souljah_Pc\PycharmProjects\courses\Feature Engineering\datasets\titanic.csv')

df.columns = [col.upper() for col in df.columns]

# 1. Feature Engineering(Değişken Mühendisliği):
# Cabin bool:
df['NEW_CABIN_BOOL'] = df['CABIN'].notnull().astype('int64')

# Name Count:
df['NEW_NAME_COUNT'] = df['NAME'].str.len()

# name word count:
df['NEW_NAME_WORD_COUNT'] = df['NAME'].apply(lambda x: len(str(x).split(' ')))

# name dr:
df['NEW_NAME_DR'] = df['NAME'].apply(lambda x: len([x for x in x.split() if x.startswith('Dr')]))

# name title:
df['NEW_TITLE'] = df['NAME'].str.extract(' ([A-Za-z]+)\.', expand=False)

# family size:
df['NEW_FAMILY_SIZE'] = df['SIBSP'] + df['PARCH'] + 1

# age_class:
df['NEW_AGE_PCLASS'] = df['AGE'] * df['PCLASS']

# is alone:
df.loc[((df['SIBSP'] + df['PARCH']) > 0), 'NEW_IS_ALONE'] = 'NO'
df.loc[((df['SIBSP'] + df['PARCH']) == 0), 'NEW_IS_ALONE'] = 'YES'

# age level:
df.loc[df['AGE'] < 18, 'NEW_AGE_CAT'] = 'young'
df.loc[(df['AGE'] >= 18) & (df['AGE'] < 56), 'NEW_AGE_CAT'] = 'mature'
df.loc[df['AGE'] >= 56, 'NEW_AGE_CAT'] = 'senior'

# sex x age:
df.loc[(df['SEX'] == 'male') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
df.loc[(df['SEX'] == 'male') & ((df['AGE'] > 21) & (df['AGE'] <= 50)), 'NEW_SEX_CAT'] = 'maturemale'
df.loc[(df['SEX'] == 'male') & (df['AGE'] > 50), 'NEW_SEX_CAT'] = 'seniormale'

df.loc[(df['SEX'] == 'female') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
df.loc[(df['SEX'] == 'female') & ((df['AGE'] > 21) & (df['AGE'] <= 50)), 'NEW_SEX_CAT'] = 'maturefemale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] > 50), 'NEW_SEX_CAT'] = 'seniorfemale'

cat_cols, num_cols, cat_but_car = grab_col_names(df, info = True)

num_cols.remove('PASSENGERID')

# Aykırı değerler:
for col in num_cols:
    replacement_with_thresholds(df, col)

# Missing Values:
missing_values_table(df)
# 'CABIN' yerine 'NEW_CABIN_BOOL' var.
df.drop('CABIN', inplace = True, axis = 1)

df.drop(['TICKET', 'NAME'], inplace = True, axis = 1)

df['AGE'] = df['AGE'].fillna(df.groupby('NEW_TITLE')['AGE'].transform('median'))

# 'AGE'in eksik değerleri dolduruldu ve 'AGE' e bağlı diğer sütunlar tekrar oluşturuldu:
df["NEW_AGE_PCLASS"] = df["AGE"] * df["PCLASS"]

df.loc[(df['AGE'] < 18), 'NEW_AGE_CAT'] = 'young'
df.loc[(df['AGE'] >= 18) & (df['AGE'] < 56), 'NEW_AGE_CAT'] = 'mature'
df.loc[(df['AGE'] >= 56), 'NEW_AGE_CAT'] = 'senior'

df.loc[(df['SEX'] == 'male') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
df.loc[(df['SEX'] == 'male') & (df['AGE'] > 21) & (df['AGE'] < 50), 'NEW_SEX_CAT'] = 'maturemale'
df.loc[(df['SEX'] == 'male') & (df['AGE'] >= 50), 'NEW_SEX_CAT'] = 'seniormale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] > 21) & (df['AGE'] < 50), 'NEW_SEX_CAT'] = 'maturefemale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] >= 50), 'NEW_SEX_CAT'] = 'seniorfemale'

df = df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype in ['category', 'object'] and
                                                  len(x.unique()) <= 10) else x, axis=0)

# Label Encoding:
binary_cols = [col for col in df.columns if df[col].dtype not in ['int64', 'float64']
               and df[col].nunique() == 2]
for col in binary_cols:
    df = label_encoder(df, col)

# Rare Encoding:
rare_analyser(df, 'SURVIVED', cat_cols)

df = rare_encoder(df, 1)

# One-Hot Encoding:
ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]
df = one_hot_encoder(df, ohe_cols)

cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols.remove('PASSENGERID')
rare_analyser(df, 'SURVIVED', cat_cols)

# 2 sınıflı olup, sınıflardan biri yüzde 1'den az olan sütunları elde etme(istersen sil):
useless_cols = [col for col in df.columns if df[col].nunique() == 2
                and ((df[col].value_counts() / len(df) * 100) < 1).any(axis = None)]

# Standartlaştırma (Bu uygulama için gerekli değil):
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# Modelleme:
X = df.drop(['PASSENGERID', 'SURVIVED'], axis = 1)
y = df['SURVIVED']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state =17)

from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(random_state = 46).fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)


# Değişkenlerin model başarısındaki etkisi:
def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                      ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(rf_model, X_train)