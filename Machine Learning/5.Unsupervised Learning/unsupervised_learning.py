import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram

pd.set_option('display.width', 1881)
pd.set_option('display.max_columns', 1881)

# K-Means
df = pd.read_csv(r'C:\Users\Souljah_Pc\PycharmProjects\courses\Machine Learning\datasets\USArrests.csv',
                 index_col = 0)

sc = MinMaxScaler((0, 1))
# fit_transform, bir numpy array döndürür. sütun sayısı az olduğu için df oluşturmadı:
df = sc.fit_transform(df)

kmeans = KMeans(n_clusters = 4, random_state = 17)
kmeans.fit(df)

# kmeans modelinin tüm parametreleri:
kmeans.get_params()

# Küme sayısı:
kmeans.n_clusters
# Oluşturulan kümelerin merkezleri:
kmeans.cluster_centers_
# Gözlemlerin etiketlendirme sonucu:
kmeans.labels_
# Gözlemlerin SSE'si:
kmeans.inertia_
# Memory Leak için önlem:
# import os
#
# # OMP_NUM_THREADS değişkenini ayarla
# os.environ['OMP_NUM_THREADS'] = '1'

# En iyi küme sayısını belirleme (İş problemi özelinde fikir oluşturması için):
ssd = []

for k in range(1, 30):
    kmeans = KMeans(n_clusters = k)
    kmeans.fit(df)
    ssd.append(kmeans.inertia_)

plt.plot(range(1, 30), ssd, 'bx-')
plt.xlabel('K değerleri için SSE sonuçları')
plt.title('Optimum Küme Sayısı için Elbow Yöntemi')
plt.show()

# Elbow Metodunu KElbowVisualizer ile gerçekleştirme:
# Modellerin 'fit' süreleri, örnekler arası uzaklık skorları ve küme sayısını içeren grafik:
kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k = (2, 20))
elbow.fit(df)
elbow.show()
# Elbow yönteminden gelen 'k' değerini elde etme:
elbow.elbow_value_

# Final Cluster Oluşturma:
kmeans = KMeans(n_clusters = elbow.elbow_value_)
kmeans.fit(df)

kmeans.n_clusters
kmeans.cluster_centers_
kmeans.labels_

clusters = kmeans.labels_

df = pd.read_csv(r'C:\Users\Souljah_Pc\PycharmProjects\courses\Machine Learning\datasets\USArrests.csv',
                 index_col = 0)

df['cluster'] = clusters
df.head()

df.groupby('cluster').agg(['count', 'mean', 'median'])

# Hierarchical Clustering
df = pd.read_csv(r'C:\Users\Souljah_Pc\PycharmProjects\courses\Machine Learning\datasets\USArrests.csv',
                 index_col = 0)

sc = MinMaxScaler((0, 1))
df = sc.fit_transform(df)

hc_average = linkage(df, 'average')

plt.figure(figsize = (10, 5))
plt.title('Hiyerarşik Kümeleme Dendogramı')
plt.xlabel('Gözlem Birimleri')
plt.ylabel('Uzaklıklar')
dendrogram(hc_average,
           truncate_mode = 'lastp',
           p = 10,
           show_contracted = True,
           show_leaf_counts = True,
           leaf_font_size = 10)
plt.show()

# Küme Sayısının Belirlenmesi:
plt.figure(figsize = (10, 5))
plt.title('Dendogram')
dend = dendrogram(hc_average,
                  truncate_mode = 'lastp',
                  p = 10,
                  show_contracted = True,
                  leaf_font_size = 10)
plt.axhline(y = 0.6, color = 'r', linestyle = '--')
plt.axhline(y = 0.5, color = 'r', linestyle = '--')
plt.show()

# Final Model (Öncesinde Küme sayısını belirledik):
cluster = AgglomerativeClustering(n_clusters = 8, linkage = 'average')

clusters = cluster.fit_predict(df)

df = pd.read_csv(r'C:\Users\Souljah_Pc\PycharmProjects\courses\Machine Learning\datasets\USArrests.csv',
                 index_col = 0)
df['hiy_cluster_no'] = clusters