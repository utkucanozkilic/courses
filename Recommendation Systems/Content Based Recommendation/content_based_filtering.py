# Content Based Recommendation

# Filmlerin overview(genel bakış) tanımlarına göre öneri sistemi geliştirme

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

pd.set_option('display.max_columns', 1881)
pd.set_option('display.width', 1881)


def cosine_similarity_matrix(dataframe, language = 'english'):

    tfidf = TfidfVectorizer(stop_words = language)
    tfidf_matrix = tfidf.fit_transform(dataframe['overview'])
    return cosine_similarity(tfidf_matrix, tfidf_matrix)


def content_based_recommender(dataframe, film, cosine_similarty_matrix):
    # Film isimlerini indeks ve indekslerini de değerleri olacak şekilde indices'ı oluştur:
    indices = pd.Series(df.index, index = df['title'])
    # Tekrarlı kayıtları, son kaydı tutacak şekilde sil:
    indices = indices[~indices.index.duplicated(keep = 'last')]
    # İlgili film ile diğer filmlerin benzerlik oranlarını similarty_scores'ta tut:
    similarty_scores = pd.DataFrame(cosine_similarty_matrix[indices[film]], columns = ["score"])
    # İlgili film ile benzerliği en yüksek 10 filmi döndür:
    return dataframe.iloc[similarty_scores.sort_values(by = "score", ascending = False)[1:11].index]['title']


# df'i oku:
df = pd.read_csv(r"C:\Users\Souljah_Pc\PycharmProjects\Recommendation Systems\movies_metadata.csv")

# Boş değerleri hesaplamada etkisiz hale getir:
df['overview'] = df["overview"].fillna(' ')

cosine_sim = cosine_similarity_matrix(df)
content_based_recommender(df, 'Sherlock Holmes', cosine_sim)