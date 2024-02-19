import pandas as pd
from surprise import Reader, SVD, Dataset, accuracy
from surprise.model_selection import GridSearchCV, train_test_split, cross_validate


movie = pd.read_csv(r"/Recommendation Systems/movie_lens_dataset/movie.csv")
rating = pd.read_csv(r"/Recommendation Systems/movie_lens_dataset/rating.csv")
df = movie.merge(rating, how = 'left', on = 'movieId')

movie_ids = [130219, 356, 4422, 541]
movies = ["The Dark Knight (2011)", "Cries and Whispers (Viskningar och rop) (1972)",
          "Forrest Gump (1994)", "Blade Runer (1982)"]

sample_df = df[df.movieId.isin(movie_ids)]

user_movie_df = pd.pivot_table(sample_df, values = 'rating',
                               index = 'userId',
                               columns = 'title')

# surprise kütüphanesi gereklilikleri:
# Reader'dan bir örnek oluştur ve ölçek sınırlarını belirt:
reader = Reader(rating_scale = (1, 5))
# df'i surprise kütüphanesinin istediği forma getirme
data = Dataset.load_from_df(sample_df[['userId',
                                       'movieId',
                                       'rating']], reader)

trainset, testset = train_test_split(data, test_size = 0.25)
svd_model = SVD()
svd_model.fit(trainset)
predictions = svd_model.test(testset)

# rmse elde etme:
accuracy.rmse(predictions)

# bir kullanıcı için tahmin:
svd_model.predict(uid = 1, iid = 541, verbose = True)
# aynı kullanıcının gerçek 'rating' değeri:
sample_df[(sample_df['userId'] == 1) & (sample_df['movieId'] == 541)]

# GridSearch
param_grid = {
    'n_epochs': [5, 10, 20],
    'lr_all': [0.002, 0.005, 0.007]}

gs = GridSearchCV(
    SVD,
    param_grid,
    measures = ['rmse', 'mae'],
    cv = 3, n_jobs = -1, joblib_verbose = True
    )

gs.fit(data)
# minimum hata ve bu hatayı veren hiperparametre değerlerini elde etme:
gs.best_score['rmse']
gs.best_params['rmse']

# SVD (singular value decomposition) modelini uygun hiperparametrelerle tekrar oluşturma:
svd_model = SVD(**gs.best_params['rmse'])

# Tüm verisetini eğitim veri setine dönüştürme:
data = data.build_full_trainset()
svd_model.fit(data)

svd_model.predict(uid = 1, iid = 541, verbose = True)