# User-Based Collaborative Filtering

import pandas as pd

pd.set_option('display.max_columns', 1881)
pd.set_option('display.width', 1881)
pd.set_option('display.expand_frame_repr', False)  # Çıktıyı alt satıra geçmeden tek satırda göster


def create_user_based_collaborative_filtering_df():
    movie = pd.read_csv(r'/Recommendation Systems/movie_lens_dataset/movie.csv')
    rating = pd.read_csv(r'/Recommendation Systems/movie_lens_dataset/rating.csv')
    df = pd.merge(movie, rating, on = 'movieId')
    rating_counts = pd.DataFrame(df['title'].value_counts())
    rare_movies = rating_counts[rating_counts['count'] <= 1000].index
    common_movies = df[~df['title'].isin(rare_movies)]
    user_movie_df_ = common_movies.pivot_table(index = 'userId', columns = 'title', values = 'rating')
    return user_movie_df_


def user_based_recommender(random_user, user_movie_df, ratio = 0.6, corr_th = 0.6, score = 3.5):
    # Sadece kullanıcı özelinde sütunları filmler olan df:
    random_user_df = user_movie_df[user_movie_df.index == random_user]
    # Kullanıcının izlediği filmlerin listesi:
    movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()
    # Sütunları, kullanıcının izlediği filmler ve satırları da o filmlere değerlendirme yapan diğer kullanıcılar df'i
    movies_watched_df = user_movie_df[movies_watched]
    # Diğer kullancıların izlediği filmlerin toplamı
    user_movie_count = movies_watched_df.T.notnull().sum()
    # İndeks resetleme
    user_movie_count = user_movie_count.reset_index()
    # Sütunları adlandırma
    user_movie_count.columns = ['userId', 'movie_count']
    # Asıl kullanıcı ile belirlenen orandan fazla ortak film izleyen diğer kullanıcıların id'si
    users_same_movies = user_movie_count[user_movie_count['movie_count'] > len(movies_watched) * ratio]['userId']
    # indeksi tüm kullanıcılar ve sütunları izlenen filmler olan df
    final_df = pd.concat(
        [
            movies_watched_df[movies_watched_df.index.isin(users_same_movies)],
            random_user_df[movies_watched]
            ]
        )
    # korelasyon matris
    corr_df = final_df.T.corr().stack().sort_values().drop_duplicates()
    corr_df = pd.DataFrame(data = corr_df, columns = ['corr'])
    corr_df.index.names = ['user_id_1', 'user_id_2']
    corr_df = corr_df.reset_index()
    # korelasyon eşik değeri üzerindeki kullanıcılar
    top_users = corr_df[
        (corr_df['user_id_1'] == random_user) & (corr_df['corr'] >= corr_th)][
        ['user_id_2', 'corr']].reset_index(drop = True)
    top_users.drop_duplicates(subset = ['user_id_2'], inplace = True)
    top_users.reset_index(inplace = True, drop = True)
    top_users = top_users.sort_values(by = 'corr', ascending = False)
    top_users.rename(columns = {'user_id_2': 'userId'}, inplace = True)

    rating = pd.read_csv(r'/Recommendation Systems/movie_lens_dataset/rating.csv')
    # Seçilen kullanıcıların tüm filmler üzerindeki değerlendirmeleri
    top_users_ratings = top_users.merge(rating[['userId', 'movieId', 'rating']], how = 'inner')
    top_users_ratings = top_users_ratings[top_users_ratings['userId'] != random_user]

    # Basit bir ağırlıklı skor hesabı
    top_users_ratings['weighted_rating'] = top_users_ratings['rating'] * top_users_ratings['corr']
    recommendation_df = top_users_ratings.groupby('movieId').agg({'weighted_rating': 'mean'})
    recommendation_df.reset_index(inplace = True)
    # belirlenen ağırlıklı skor üzerini elde etme
    movies_to_be_recommend = recommendation_df[recommendation_df['weighted_rating'] > score]. \
        sort_values(by = 'weighted_rating', ascending = False)

    movie = pd.read_csv(r'/Recommendation Systems/movie_lens_dataset/movie.csv')

    return movies_to_be_recommend.merge(movie[['movieId', 'title']], how = 'inner')  # defaul 'inner'


user_movie_df = create_user_based_collaborative_filtering_df()
random_user_ = user_movie_df.sample(1).index[0]
user_based_recommender(random_user_, user_movie_df)