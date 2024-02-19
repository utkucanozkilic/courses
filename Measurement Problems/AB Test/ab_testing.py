import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, pearsonr, spearmanr, kendalltau, \
    pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest

pd.set_option('display.max_columns', 1881)
pd.set_option('display.width', 1881)

populasyon = np.random.randint(0, 80, 10000)

np.random.seed(115)

orneklem = np.random.choice(populasyon, 100)

populasyon.mean(), orneklem.mean()  # (39.5572, 43.23)

np.random.seed(10)
orneklem1 = np.random.choice(populasyon, 100)
orneklem2 = np.random.choice(populasyon, 100)
orneklem3 = np.random.choice(populasyon, 100)
orneklem4 = np.random.choice(populasyon, 100)
orneklem5 = np.random.choice(populasyon, 100)
orneklem6 = np.random.choice(populasyon, 100)
orneklem7 = np.random.choice(populasyon, 100)
orneklem8 = np.random.choice(populasyon, 100)
orneklem9 = np.random.choice(populasyon, 100)
orneklem10 = np.random.choice(populasyon, 100)

(orneklem1.mean() + orneklem2.mean() + orneklem3.mean() + orneklem4.mean() + orneklem5.mean() +
 orneklem6.mean() + orneklem7.mean() + orneklem8.mean() + orneklem9.mean() + orneklem10.mean()) / 10  # 40.059

df = sns.load_dataset("tips")
df.describe().T

# stats.model.api / as sms ile güven aralığı
sms.DescrStatsW(df["total_bill"]).tconfint_mean()  # (18.66333170435847, 20.908553541543164) mean = 19.785943
sms.DescrStatsW(df["tip"]).tconfint_mean()  # (2.8237993062818205, 3.172758070767359)

df["total_bill"] = df["total_bill"] - df["tip"]

df.plot.scatter(x = "tip", y = "total_bill")
plt.show()

df["tip"].corr(df["total_bill"])  # 0.5766634471096374

# AB Testing (Bağımsız iki örneklem t testi)
# 1. Hipotezleri Kur
# 2. Varsayım Kontrolü
# 2.1. Normallik Varsayımı
# 2.2. Varyans Homojenliği
# 3. Hipotezin Uygulanması
# 3.1. Varsayımlar sağlanıyorsa bağımsız iki örneklem t testi (parametrik test)
# 3.2. Varsayımlar sağlanmıyorsa mannwhitneyu testi (non-parametrik test)
# 4. p-value'ya göre değerlendir
# not:
# - Normallik sağlanmıyorsa direk 3.2. numara. Varyans homojenliği sağlanmıyorsa 1 numarasaya argüman girilir.
# Yani, 'homojenlik sağlanmıyor aklında olsun.' de
# - Normallik incelemesi öncesi aykırı değer incelemesi ve düzeltmesi yapmak faydalı olabilir.

# Uygulama 1: Sigara içenlewr ile içmeyenler arasında hesap ortalamaları açısından istatistiksel olarak anlamlı fark?
df = sns.load_dataset("tips")

# 1. h0: m1 = m2 h1: m1 != m2
# 2. Varsayım kontrolü
# Normallik varsayımı
# varyans homojenliği
# h0: normal dağılım varsayımı sağlanmaktadır.
# h1: normal dağılım varsayımı sağlanmamaktadır.

# 2 grup için de p < 0.05 >>>>> h0 red edilir
test_stat, pvalue = shapiro(df.loc[df["smoker"] == "Yes", "total_bill"])
print("Test Stat = {:.4f}, p-value = {:.4f}".format(test_stat, pvalue))  # Test Stat = 0.9367, p-value = 0.0002

test_stat, pvalue = shapiro(df.loc[df["smoker"] == "No", "total_bill"])
print("Test Stat = {:.4f}, p-value = {:.4f}".format(test_stat, pvalue))  # Test Stat = 0.9045, p-value = 0.0000

# Varyans homojenliği (normallik varsayımı sağlanmadı fakat örnek olması için)
# h0: varyanslar homojendir
# h1: varyanslar homojen değildir

# p < 0.05 >>>>>>> h0 red edilir
test_stat, pvalue = levene(
    df.loc[df["smoker"] == "Yes", "total_bill"],
    df.loc[df["smoker"] == "No", "total_bill"]
    )
print("Test Stat = {:.4f}, p-value = {:.4f}".format(test_stat, pvalue))  # Test Stat = 4.0537, p-value = 0.0452

# 3. Hipotezin Uygulanması

# 1. Varsayımlar sağlanıyorsa t testi (parametrik test) (sağlanmadı fakat örnek olması için)
# p > 0.05 h0 reddedilmez.
test_stat, pvalue = ttest_ind(
    df.loc[df["smoker"] == "Yes", "total_bill"],
    df.loc[df["smoker"] == "No", "total_bill"],
    equal_var = True
    )
print("Test Stat = {:.4f}, p-value = {:.4f}".format(test_stat, pvalue))  # Test Stat = 1.3384, p-value = 0.1820

# 2. Varsayımlar sağlanmadığı için mannwithneyu testi (non-parametrik test)
# p > 0.05 ho reddedilmedi.
test_stat, pvalue = mannwhitneyu(
    df.loc[df["smoker"] == "Yes", "total_bill"],
    df.loc[df["smoker"] == "No", "total_bill"]
    )
print("Test Stat = {:.4f}, p-value = {:.4f}".format(test_stat, pvalue))  # Test Stat = 7531.5000, p-value = 0.3413

# Uygulama 2: Titanic Kadın ve Erkek Yolcuların Yaş Ortalamaları arasında ist.önm. fark var mıdır?
df = sns.load_dataset("titanic")

# 1. Hipotezi kur:
# h0: Kadın/Erkek yaş ortalamaları arasında fark yoktur. (m1=m2)
# h1: Kadın/Erkek yaş ortalamaları arasında fark vardır. (m1!=m2)

# 2. Varsayımları incele
# 2.1. Normallik Varsayımını incele (iki grup için de normallik varsayımı sağlanmamaktadır.)
test_stat, pvalue = shapiro(df.loc[df["sex"] == "female", "age"].dropna())
# Test Stat = 0.9848, p-value = 0.0071 < 0.05 ---- h0 reddedildi.
print("Test Stat = {:.4f}, p-value = {:.4f}".format(test_stat, pvalue))

# Test Stat = 0.9747, p-value = 0.0000 < 0.05 ------------ h0 reddedildi.
test_stat, pvalue = shapiro(df.loc[df["sex"] == "male", "age"].dropna())
print("Test Stat = {:.4f}, p-value = {:.4f}".format(test_stat, pvalue))

# 2. Varyans Homojenliği
# 2.1. h0: varyanslar homojendir
# 2.2. h1: varyanslar homojen değildir
test_stat, pvalue = levene(
    df.loc[df["sex"] == "female", "age"].dropna(),
    df.loc[df["sex"] == "male", "age"].dropna()
    )
# Test Stat = 0.0013, p-value = 0.9712 > 0.05 --------- h0 reddedilir.
print("Test Stat = {:.4f}, p-value = {:.4f}".format(test_stat, pvalue))

# nonparametrik test gerçekleştirilir
test_stat, pvalue = mannwhitneyu(
    df.loc[df["sex"] == "female", "age"].dropna(),
    df.loc[df["sex"] == "male", "age"].dropna()
    )
# Test Stat = 53212.5000, p-value = 0.0261 < 0.05 ------ h0 reddedildi.
print("Test Stat = {:.4f}, p-value = {:.4f}".format(test_stat, pvalue))

# Uygulama 3: Diyabet hastası olanların/olmayanların yaşlarının ortalamaları arasında ist.önm. fark var mı?
df = pd.read_csv("/Measurement Problems/AB Test/diabetes.csv")

# 1. Hipotezi kur
# h0: yaşlar arasında fark yoktur.
# h1: ... fark vardır.

# 2. Varsayımlarınm kontrolü
# 2.1. Normallik Varsayımı >>>> h0: normallik yoktur, h: normallik vardır
test_stat, pvalue = shapiro(df.loc[df["Outcome"] == 1, "Age"].dropna())
# Test Stat = 0.9546, p-value = 0.0000 < 0.05 -------- h0 reddedildi
print("Test Stat = {:.4f}, p-value = {:.4f}".format(test_stat, pvalue))

# Test Stat = 0.8012, p-value = 0.0000 < 0.05 -------- h0 reddedildi
test_stat, pvalue = shapiro(df.loc[df["Outcome"] == 0, "Age"].dropna())
print("Test Stat = {:.4f}, p-value = {:.4f}".format(test_stat, pvalue))

# mannwhiteneyu
test_stat, pvalue = mannwhitneyu(
    df.loc[df["Outcome"] == 1, "Age"],
    df.loc[df["Outcome"] == 0, "Age"]
    )
# Test Stat = 92050.0000, p-value = 0.0000 < 0.05 --------- h0 reddedildi
print("Test Stat = {:.4f}, p-value = {:.4f}".format(test_stat, pvalue))

# Uygulama: Kursun büyük çoğunluğunu izleyenler ile izlemeyenlerin puanları birbirinden farklı mı?

df = pd.read_csv("/Measurement Problems/AB Test/course_reviews.csv")

# Hipotezi kur:
# h0: fark yoktur.
# h1: fark vardır.

# 2. Varsayımlar kontrolü
# 2.1. Normallik kontrolü
test_stat, pvalue = shapiro(df.loc[df["Progress"] > 75, "Rating"].dropna())
# Test Stat = 0.3160, p-value = 0.0000 < 0.05 ----- h0 red
print("Test Stat = {:.4f}, p-value = {:.4f}".format(test_stat, pvalue))

test_stat, pvalue = shapiro(df.loc[df["Progress"] < 40, "Rating"].dropna())
# Test Stat = 0.5612, p-value = 0.0000 < 0.05 ------ h0 red
print("Test Stat = {:.4f}, p-value = {:.4f}".format(test_stat, pvalue))

# mannwhitenyu
test_stat, pvalue = mannwhitneyu(
    df.loc[df["Progress"] > 75, "Rating"].dropna(),
    df.loc[df["Progress"] < 40, "Rating"].dropna()
    )
# Test Stat = 813814.0000, p-value = 0.0000 < 0.05 ----- h0 red / fark var
print("Test Stat = {:.4f}, p-value = {:.4f} ".format(test_stat, pvalue))

################################################################################
################################################################################
################################################################################
# AB Testing (iki örneklem oran testi)
################################################################################
################################################################################
################################################################################


# h0: p1 = p2, h1: p1 != p2
# yeni tasarımın dönüşüm oranı ile eski tasarımın dönüşüm oranı arasında ist.olr.anlm.fark var mı?

basari_sayisi = np.array([300, 250])

gozlem_sayisi = np.array([1000, 1100])

# Out[52]: (3.7857863233209255, 0.0001532232957772221) < 0.05 ---- h0 red
proportions_ztest(count = basari_sayisi, nobs = gozlem_sayisi)

# Uygulama: Kadın ve Erkeklerin Hayatta kalma oranları arasında ist.olr.anlm. fark var mıdır?
# h0: p1 = p2
# h1: p!= p2

df = sns.load_dataset("titanic")

df.loc[df["sex"] == "female", "survived"].mean()  # Out[5]: 0.7420382165605095
df.loc[df["sex"] == "male", "survived"].mean()  # Out[6]: 0.18890814558058924 Fark bariz

test_stat, pvalue = proportions_ztest(
    count = [
        df.loc[df["sex"] == "female", "survived"].sum(),
        df.loc[df["sex"] == "male", "survived"].sum()
        ],
    nobs = [
        df.loc[df["sex"] == "female", "survived"].shape[0],
        df.loc[df["sex"] == "male", "survived"].shape[0]
        ]
    )
# Test Stat = 16.2188, p-value = 0.0000 ---- h0 red
print("Test Stat = {:.4f}, p-value = {:.4f}".format(test_stat, pvalue))

########################################################################################


# ANOVA (Analysis of Variance)

df = sns.load_dataset("tips")

df.groupby("day")["total_bill"].mean()

# 1. Hipotezler
# m1 = m2 = m3 = m4 mı?
# 2. Varsayım Kontrolü

# Varsayım sağlanıyorsa one way anova
# varsayım sağlanmıyorsa kruskal

# Normallik Varsayımı
# All p-values < 0.05 ---------- hepsi için h0 red
# For Sun group Test Stat = 0.9479, p-value = 0.0036
# For Sat group Test Stat = 0.9035, p-value = 0.0000
# For Thur group Test Stat = 0.8845, p-value = 0.0000
# For Fri group Test Stat = 0.8958, p-value = 0.0409

for group in list(df["day"].unique()):
    test_stat, pvalue = shapiro(df.loc[df["day"] == group, "total_bill"])
    print("For {} group Test Stat = {:.4f}, p-value = {:.4f}".format(group, test_stat, pvalue))

# Varyans homojenliği varsayımı
# h0: sağlanır, h1: sağlanmaz
# Test Stat = 0.6654, p-value = 0.5741 > 0.05 h0 reddedilemez
test_stat, pvalue = levene(
    df.loc[df["day"] == "Sun", "total_bill"],
    df.loc[df["day"] == "Sat", "total_bill"],
    df.loc[df["day"] == "Thur", "total_bill"],
    df.loc[df["day"] == "Fri", "total_bill"]
    )
print("Test Stat = {:.4f}, p-value = {:.4f}".format(test_stat, pvalue))

# 3. Hipotez testi ve p-value yorumlama

# Varsayımların sağlandığını varsayarsak:
# parametrik anova testi
# Out[21]: F_onewayResult(statistic=2.7674794432863363, pvalue=0.04245383328952047) p < 0.05 h0 red
f_oneway(
    df.loc[df["day"] == "Sun", "total_bill"],
    df.loc[df["day"] == "Sat", "total_bill"],
    df.loc[df["day"] == "Thur", "total_bill"],
    df.loc[df["day"] == "Fri", "total_bill"]
    )

# nonparametrik anova testi
# Out[22]: KruskalResult(statistic=10.403076391436972, pvalue=0.015433008201042065) < 0.05 h0 red
kruskal(
    df.loc[df["day"] == "Sun", "total_bill"],
    df.loc[df["day"] == "Sat", "total_bill"],
    df.loc[df["day"] == "Thur", "total_bill"],
    df.loc[df["day"] == "Fri", "total_bill"]
    )


from statsmodels.stats.multicomp import MultiComparison
comparison = MultiComparison(df["total_bill"], df["day"])
tukey = comparison.tukeyhsd(0.05)
print(tukey)