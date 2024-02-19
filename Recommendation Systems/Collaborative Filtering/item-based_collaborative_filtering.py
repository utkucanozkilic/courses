# Item-Based Collaborative Filtering

import pandas as pd

pd.set_option('display.max_columns', 1881)
pd.set_option('display.width', 1881)


def get_user_movie_df(dataframe, count = 1000):
    rating_counts = pd.DataFrame(dataframe['title'].value_counts())
    rare_movies = rating_counts[rating_counts['count'] <= count].index
    common_movies = df[~df['title'].isin(rare_movies)]
    user_movie_df_ = common_movies.pivot_table(index = 'userId', columns = 'title', values = 'rating')
    return user_movie_df_


def get_item_based_collaborative_filtering(film_name, user_movie_df_):
    film_name_list = [col for col in user_movie_df_.columns if film_name in col]
    print(film_name_list)
    for film in film_name_list:
        print("###Film Name:", film)
        similar_films = user_movie_df_.corrwith(user_movie_df_[film]).sort_values(ascending = False).head(10)
        print(similar_films, "\n")


movie = pd.read_csv(r'/Recommendation Systems/movie_lens_dataset/movie.csv')
rating = pd.read_csv(r'/Recommendation Systems/movie_lens_dataset/rating.csv')
df = pd.merge(movie, rating, on = 'movieId')


user_movie_df = get_user_movie_df(df)

get_item_based_collaborative_filtering("Mission: Impossible (1996)", user_movie_df)