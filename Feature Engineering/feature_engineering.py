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


pd.set_option('display.max_columns', 1881)
pd.set_option('display.width', 1881)

df = pd.read_csv(r'/Feature Engineering/datasets/titanic.csv')
df_app_train = pd.read_csv(r'/Feature Engineering/datasets/application_train.csv')

sns.boxplot(df['Age'])
plt.show()


# Sınırları elde etme:
def outlier_threshold(dataframe, column, first_percent = 0.25, third_percent = 0.75):
    q1 = dataframe[column].quantile(first_percent)
    q3 = dataframe[column].quantile(third_percent)
    iqr = q3 - q1
    up_limit = q3 + 1.5 * iqr
    low_limit = q1 - 1.5 * iqr

    return low_limit, up_limit


# Aykırı değer kontrolü:
def check_outlier(dataframe, column, first_percent = 0.25, third_percent = 0.75):
    q1 = dataframe[column].quantile(first_percent)
    q3 = dataframe[column].quantile(third_percent)
    iqr = q3 - q1
    up_limit = q3 + 1.5 * iqr
    low_limit = q1 - 1.5 * iqr

    return dataframe[(dataframe[column] < low_limit) | (dataframe[column] > up_limit)].any(axis = None)


check_outlier(df, 'Fare')


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


caterogical_columns, numerical_columns, caterogical_but_cardinal_columns = grab_col_names(df)
# PassengerId istenmeyen sayısal sütun:
numerical_columns.remove('PassengerId')


cat, num, cat_b_car = grab_col_names(df_app_train, info = True)

num.remove('SK_ID_CURR')


def grab_outliers(dataframe, column, index = False):
    low, up = outlier_threshold(dataframe, column)

    if len(dataframe[(dataframe[column] < low) | (dataframe[column] > up)]) > 10:
        print(dataframe[(dataframe[column] < low) | (dataframe[column] > up)].head())
    else:
        print(dataframe[(dataframe[column] < low) | (dataframe[column] > up)])

    if index:
        return dataframe[(dataframe[column] < low) | (dataframe[column] > up)].index


outlier_threshold(df, 'Age')
grab_outliers(df, 'Age', index = True)


# Aykırı değer silme:
low, up = outlier_threshold(df, 'Fare')
df.shape


def remove_outlier(dataframe, column):
    low, up = outlier_threshold(dataframe, column)
    return dataframe[~((dataframe[column] < low) | (dataframe[column] > up))]


cat_cols, num_cols, cat_but_car_cols = grab_col_names(df)
num_cols.remove('PassengerId')

for col in num_cols:
    new_df = remove_outlier(df, col)


# Baskılama (re-assignment with thresholds)
def replacement_with_thresholds(dataframe, column):
    low_limit, up_limit = outlier_threshold(dataframe, column)
    dataframe.loc[(dataframe[column] < low_limit), column] = low_limit
    dataframe.loc[(dataframe[column] > up_limit), column] = up_limit


for col in num_cols:
    replacement_with_thresholds(df, col)

# Özet:
df = pd.read_csv(r'/Feature Engineering/datasets/titanic.csv')
outlier_threshold(df, 'Age')
check_outlier(df, 'Age')
grab_outliers(df, 'Age', index = True)

remove_outlier(df, 'Age')
replacement_with_thresholds(df, 'Age')
check_outlier(df, 'Age')


# Çok değişkenli Aykırı Değer Analizi: Local Outlier Factor

df = sns.load_dataset('diamonds')
df = df.select_dtypes(include = ['float64', 'int64'])
df = df.dropna()
df.head()

# Her bir özellikte/değişkende aykırı değerler için işlemler yapsaydık, çok fazla veri manipüle ederdik:
low, up = outlier_threshold(df, 'carat')

clf = LocalOutlierFactor(n_neighbors = 20)
clf.fit_predict(df)

# Nesnenin içindeki bir veriye erişiyorsun. O yüzden renklendirme yok.
df_scores = clf.negative_outlier_factor_

# Sıralama:
np.sort(df_scores)

# Verilerin (gözlemlerin) skorlarıyla birlikte çizdirilmesi:
# Böylece, görsel yoldan eşik değer belirlemek için fikir elde edilir:
scroes = pd.DataFrame(np.sort(df_scores))
scroes.plot(xlim = [0, 20], style = '.-')
plt.show()

th = np.sort(scroes)[4]

# df_scores < th ile True/False elde ederiz. Sonra seçim yaparız:
df[df_scores < th]

# Aykırı değerleri atmak istersek:
df.drop(axis = 0, labels = df[df_scores < th].index, inplace = True)

############ Missing Values ################

######### Eksik Değerlerin Yakalanması ###############

df = pd.read_csv(r'/Feature Engineering/datasets/titanic.csv')

df.isnull().values.any()  # True

df.isnull().sum()  # Her sütundaki eksik değerlerin toplam sayısı

df.isnull().sum().sum()  # Tüm df'teki eksik değerlerin toplam sayısı

df[df.isnull().any(axis = 1)]  # en az bir tane eksik değeri olan örnekler

# Her sütundaki eksik değerlerin, veri setine göre yüzdesel olarak ne kadar eksik değer barındırdığı:
((df.isnull().sum() / len(df) * 100).sort_values(ascending = False))

# Yalnızca eksik değere sahip sütun isimlerini alma:
na_cols = [col for col in df.columns if df[col].isnull().sum() > 0]


# Eksik değerlerle ilgili bilgilendirme yazdıran fonksiyonumuz:
def missing_values_table(dataframe, na_name = False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending = False)
    ratio = (dataframe[na_columns].isnull().sum() / len(dataframe) * 100).sort_values(ascending = False)
    missin_df = pd.concat(objs = [n_miss, np.round(ratio, 2)], axis = 1, keys = ['n_miss', 'ratio'])

    print(missin_df)

    if na_name:
        return na_columns


missing_values_table(df, True)

# Eksik değerlere yaklaşımlar:
# Silme:


# Doldurma:

df['Age'].fillna(df['Age'].mean())  # Belirli değerlerle doldurma
# Sayısal değerleri seçerek doldurma:
dff = df.apply(lambda x: x.fillna(x.mean()) if x.dtype not in ['object', 'category'] else x, axis = 0)
# Kategorik değerleri doldurma yöntemi olarak en çok tekrar eden değer ile doldurma:
df['Embarked'].fillna(df['Embarked'].mode()[0])
# ya da:
df['Embarked'].fillna('missing')

# .apply ile kategorik tüm sütunlardaki eksik değerleri doldurma
dff = df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype in ['object'] and len(x.unique()) <= 10) else x, axis = 0)
df['Embarked'].nunique()

# Sütunların kırılım özelinde eksik değerlerinin doldurulması:
df.groupby('Sex')['Age'].mean()  # >> female    27.915709  male      30.726645

# .transform metoduyla groupby ile oluşturulan tüm alt gruplara istenen aggregation işlemi ayrı ayrı uygulanır:
df['Age'].fillna(df.groupby('Sex')['Age'].transform('mean')).isnull().sum()

# .transform ile tek seferde yapılan işlemin parça parça .loc ile yapılması:
df.loc[(df['Age'].isnull() & (df['Sex'] == 'female')), 'Age'] = df.groupby('Sex')['Age'].mean()['female']
df.loc[(df['Age'].isnull() & (df['Sex'] == 'male')), 'Age'] = df.groupby('Sex')['Age'].mean()['male']


# Tahmine Dayalı Atama
cat_cols, num_cols, cat_but_car_cols = grab_col_names(df)
num_cols.remove('PassengerId')

# tipi kategorik olan değişkenlerin one-hot-encoding işlemi(tüm sütunlar gönderilse de cat. ile işlem yapar.):
dff = pd.get_dummies(df[cat_cols + num_cols], drop_first = True)

# değişkenlerin standartlaştırılması:
scaler = MinMaxScaler()
dff = pd.DataFrame(scaler.fit_transform(dff), columns = dff.columns)

from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors = 5)
dff = pd.DataFrame(imputer.fit_transform(dff), columns = dff.columns)
dff.head()

# standartlaştırılmanın geri alınması:
dff = pd.DataFrame(scaler.inverse_transform(dff), columns = dff.columns)
df.to_string()
# df'e eksik değerlerin sütun olarak eklenmesi:
df['age_imputed_knn'] = dff['Age']
df.drop(labels = 'age_imputed_knn', axis = 1, inplace = True)

df.loc[df['Age'].isnull(), ['Age', 'age_imputed_knn']]

# Eksik veriler ile Analiz:
# sns.heatmap(df.isnull(), cmap = 'viridis', cbar = False)
# plt.show()

# Isı haritasına benzer matrix
msno.matrix(df, figsize = (10, 10))
plt.show()

# Eksik değerler üzerinden korelasyon (eksik değerler arasında ilişki var mı?)
msno.heatmap(df, figsize = (10, 10))
plt.show()

# Tüm sütunların frekansı
msno.bar(df, figsize = (10, 10))
plt.show()


# Eksik değerlerin bağımlı değişken ile ilişkisinin incelenmesi

null_columns = missing_values_table(df, True)


def missing_vs_target(dataframe, target, na_columns):
    for col in na_columns:
        target_df = pd.DataFrame(
            data =
            {
                'TARGET_MEAN': np.array(
                    [
                        np.round((dataframe[dataframe[col].notnull()][target].mean()), 3),
                        np.round((dataframe[dataframe[col].isnull()][target].mean()), 3)
                        ]
                    ),
                'Count': np.array(
                    [
                        (dataframe[dataframe[col].notnull()][target].count()),
                        (dataframe[dataframe[col].isnull()][target].count())
                        ]
                    )
                }
            )

        target_df.index.name = col + "_NA_FLAG"

        print(target_df, '\n\n')


missing_vs_target(df, 'Survived', null_columns)

# Dersteki fonksiyon
# # Dersteki fonksiyon:
# def missing_vs_target(dataframe, target, na_columns):
#     temp_df = dataframe.copy()
#
#     for col in na_columns:
#         temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)
#
#     na_flags = temp_df.loc[:, temp_df.columns.str.contains('_NA_')].columns
#
#     for col in na_flags:
#         print(pd.DataFrame(
#             {
#                 'TARGET_MEAN': temp_df.groupby(col)[target].mean(),
#                 'Count': temp_df.groupby(col)[target].count()
#                 }
#             ), '\n\n\n')

# Label Encoding
# Alfabetik olarak 0, 1, 2, ... verir
le = LabelEncoder()
le.fit_transform(df['Sex'])
# geri dönüştürmek için
le.inverse_transform([0, 1])


def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    # Eğer eksik değerler var ise bunları da sıradaki sayı ile doldurur.
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


# binary columns seçimi:
binary_cols = [col for col in df.columns if df[col].dtype not in ['int64', 'float64']
               and df[col].nunique() == 2]

# Bundan sonra application_train veriseti ile çalışacağız:
df = df_app_train

binary_cols = [col for col in df.columns if df[col].dtype not in ['int64', 'float64']
               and df[col].nunique() == 2]

for col in binary_cols:
    label_encoder(df, col)

df[binary_cols]

# One-Hot Encoding
df = pd.read_csv(r'/Feature Engineering/datasets/titanic.csv')
df['Embarked'].value_counts()

pd.get_dummies(df, columns = ['Embarked'], drop_first = True, dtype = 'int64').head()

# Eksik değerler için de sınıf oluşturma:
pd.get_dummies(df, columns = ['Embarked'], drop_first = True, dummy_na = True, dtype = 'int64').head()

# Label encoding(binary encoding) ile one-hot encoding beraber kullanılabilir:
pd.get_dummies(df, columns = ['Sex', 'Embarked'], drop_first = True, dtype = 'int64')


def one_hot_encoder(dataframe, categorical_cols, drop_first = True, dtype = 'int64'):
    dataframe = pd.get_dummies(dataframe, columns = categorical_cols, drop_first = drop_first, dtype = dtype)
    return dataframe


cat_cols, num_cols, cat_but_car_cols = grab_col_names(df)

ohe_cols = [col for col in df.columns if 2 < df[col].nunique() <= 10]

one_hot_encoder(df, ohe_cols)


# Rare Encoding (görece az sayıda olan kategorileri (100bin örnekte 2 tane olan bir kategori gibi) işleme)
# 1. Kategorik deişkenlerin azlık çokluk durumunun analiz edilmesi:
df = df_app_train
df['NAME_EDUCATION_TYPE'].value_counts()

cat_cols, num_cols, cat_but_car_cols = grab_col_names(df)


def cat_col_summary(dataframe, col_name, plot = False):
    print(pd.DataFrame(
        {
            col_name: dataframe[col_name].value_counts(),
            'Ratio(%)': dataframe[col_name].value_counts() / len(dataframe) * 100
            }
        ),
        '\n###########################')

    if plot:
        sns.countplot(x = dataframe[col_name], data = dataframe)
        plt.show()


for col in cat_cols:
    cat_col_summary(df, col)


# 2. Rare kategoriler ile bağımlı değişken arasındaki ilişkinin analizi:
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


rare_analyser(df, 'TARGET', cat_cols)


# 3. Rare encoder fonksiyonunun yazılması:
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


new_df = rare_encoder(df, 1)
rare_analyser(new_df, target = 'TARGET', categorical_col = cat_cols)


# Feature Scaling

# StandarScaler (z-score)
df = pd.read_csv(r'/Feature Engineering/datasets/titanic.csv')
ss = StandardScaler()
df['Age_standard_scaler'] = ss.fit_transform(df[['Age']])

# RobustScaler
rs = RobustScaler()
df['Age_robust_scaler'] = rs.fit_transform(df[['Age']])

# MinMaxScaler
mms = MinMaxScaler()
df['Age_min_max_scaler'] = mms.fit_transform(df[['Age']])


# Sayısal değişkenleri kategorik biçime çevirme(Binning):
df['Age_qcut'] = pd.cut(df['Age'], 5)


# Feature Extraction
# Binary Features
df['NEW_CABIN_BOOL'] = df['Cabin'].notnull().astype('int64')

df.groupby('NEW_CABIN_BOOL').agg({'Survived': 'mean'})

from statsmodels.stats.proportion import proportions_ztest
# Z test'in formatına uygun olarak başarılı/başarlı+başarısız:
test_stat, pvalue = proportions_ztest(
    count = [
        df.loc[df['NEW_CABIN_BOOL'] == 1, 'Survived'].sum(),
        df.loc[df['NEW_CABIN_BOOL'] == 0, 'Survived'].sum()],
    nobs = [
        df.loc[df['NEW_CABIN_BOOL'] == 1, 'Survived'].shape[0],
        df.loc[df['NEW_CABIN_BOOL'] == 0, 'Survived'].shape[0]
        ]
    )
# Kabin numarası olan ve olmayanlar arasında fark yoktur, hipotezi reddedildi:
print('Test Stat = %.4f, p-value = %4.f' % (test_stat, pvalue))

# Akrabalık/Yakınlıktan özellik çıkarımı:
df.loc[(df['SibSp'] + df['Parch'] > 0), 'NEW_IS_ALONE'] = 'NO'
df.loc[(df['SibSp'] + df['Parch'] == 0), 'NEW_IS_ALONE'] = 'YES'

test_stat, p_value = proportions_ztest(
    count = [
        df.loc[df['NEW_IS_ALONE'] == 'YES', 'Survived'].sum(),
        df.loc[df['NEW_IS_ALONE'] == 'NO', 'Survived'].sum()
        ],
    nobs = [
        len(df.loc[df['NEW_IS_ALONE'] == 'YES', 'Survived']),
        len(df.loc[df['NEW_IS_ALONE'] == 'NO', 'Survived'])
        ]
    )
# Yalnız olanlar ile olmayanların hayatta kalma oranları arasında fark yoktur, hipotezi reddedildi:
print('Test Stat = %.4f, p-value = %4.f' % (test_stat, pvalue))


# Metinden Özellik Çıkarımı:
df = pd.read_csv(r'/Feature Engineering/datasets/titanic.csv')
# letter count:
df['NEW_NAME_COUNT'] = df['Name'].str.len()

# Word Count:
df['NEW_NAME_WORD_COUNT'] = df['Name'].apply(lambda x: len(str(x).split(' ')))

# Özel Yapıları Yakalamak(Anlamlı olabilecek bilgileri elde etmek):
df['NEW_NAME_DR'] = df['Name'].apply(lambda x: len([x for x in x.split() if x.startswith('Dr')]))
df.groupby('NEW_NAME_DR').agg({'Survived': ['mean', 'count']})

# Regex ile Değişken Türetme:
df['NEW_TITLE'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand = False)
df.groupby('NEW_TITLE').agg({'Survived': 'mean', 'Age': ['mean', 'count']})

# Date Değişkenleri Türetmek
dff = pd.read_csv(r'/Feature Engineering/datasets/course_reviews.csv')
# 'Timestamp' türünü değiştirme:
dff['Timestamp'] = pd.to_datetime(dff['Timestamp'], format = '%Y-%m-%d %H:%M:%S')

dff['year'] = dff['Timestamp'].dt.year
dff['month'] = dff['Timestamp'].dt.month

dff['year_diff'] = date.today().year - dff['Timestamp'].dt.year

# 2 tarih arasındaki fark (ay olarak):
dff['mont_diff'] = (((date.today().year - dff['Timestamp'].dt.year) * 12) +
                    (date.today().month - dff['Timestamp'].dt.month))

# gün isimlerine erişmek:
dff['day_name'] = dff['Timestamp'].dt.day_name()

# Feature Interactions (Özellik Etkileşimleri)
df['NEW_AGE_PCLASS'] = df['Age'] * df['Pclass']

df['NEW_FAMILY_SIZE'] = df['SibSp'] + df['Parch'] + 1

# 'NEW_SEX_CAT' sütununu oluşturur ve boş değerlere 'nan' atar:
df.loc[(df['Sex'] == 'male') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
df.loc[(df['Sex'] == 'male') & ((df['Age'] > 21) & (df['Age'] <= 50)), 'NEW_SEX_CAT'] = 'maturemale'
df.loc[(df['Sex'] == 'male') & (df['Age'] > 50), 'NEW_SEX_CAT'] = 'seniormale'

df.loc[(df['Sex'] == 'female') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
df.loc[(df['Sex'] == 'female') & ((df['Age'] > 21) & (df['Age'] <= 50)), 'NEW_SEX_CAT'] = 'maturefemale'
df.loc[(df['Sex'] == 'female') & (df['Age'] > 50), 'NEW_SEX_CAT'] = 'seniorfemale'

df.groupby('NEW_SEX_CAT')['Survived'].mean()