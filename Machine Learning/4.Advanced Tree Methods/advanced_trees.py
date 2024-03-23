################################################
# Random Forests, GBM, XGBoost, LightGBM, CatBoost
################################################


import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

pd.set_option("display.max_columns", 1881)
pd.set_option("display.width", 1881)

df = pd.read_csv(r"C:\Users\Souljah_Pc\PycharmProjects\courses\Machine Learning\datasets\diabetes.csv")

X = df.drop('Outcome', axis = 1)
y = df['Outcome']

# RandomForrest
rf_model = RandomForestClassifier(random_state = 17)
rf_model.get_params()

cv_results = cross_validate(rf_model, X, y, cv = 10, scoring = ['accuracy', 'f1', 'roc_auc'])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

rf_params = {
    'max_depth': [5, 8, None],
    'max_features': [3, 5, 7, 'sqrt'],
    'min_samples_split': [2, 5, 8, 15, 20],
    'n_estimators': [100, 200, 500]
    }

rf_best_grid = GridSearchCV(rf_model, rf_params, cv = 5, n_jobs = -1, verbose = True)
rf_best_grid.fit(X, y)

rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state = 17)
# rf_final = rf_model.set_params(**rf_best_grid.get_params())
rf_final.fit(X, y)

cv_results = cross_validate(rf_model, X, y, cv = 10, scoring = ['accuracy', 'f1', 'roc_auc'])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()


def val_curve_params(model, X, y, param_name, param_range, scoring="roc_auc", cv=10):
    train_score, test_score = validation_curve(
        model, X=X, y=y, param_name=param_name, param_range=param_range, scoring=scoring, cv=cv)

    mean_train_score = np.mean(train_score, axis=1)
    mean_test_score = np.mean(test_score, axis=1)

    plt.plot(param_range, mean_train_score,
             label="Training Score", color='b')

    plt.plot(param_range, mean_test_score,
             label="Validation Score", color='g')

    plt.title(f"Validation Curve for {type(model).__name__}")
    plt.xlabel(f"Number of {param_name}")
    plt.ylabel(f"{scoring}")
    plt.tight_layout()
    plt.legend(loc='best')
    plt.show(block=True)


def plot_importence(model, features, num = len(X), save = False):
    feature_imp = pd.DataFrame({
        'Value': model.feature_importances_,
        'Feature': features.columns
        })
    plt.figure(figsize = (10, 10))
    sns.set(font_scale = 1)
    sns.barplot(x = 'Value', y = 'Feature', data = feature_imp.sort_values(by = 'Value', ascending = False)[0:num])

    plt.title('Features')
    plt.tight_layout()
    plt.show()

    if save:
        plt.savefig('importance.png')


plot_importence(rf_final, X)
val_curve_params(rf_final, X, y, 'max_depth', range(1, 11), scoring = "roc_auc")

# GBM Model
gbm_model = GradientBoostingClassifier(random_state = 17)

cv_results = cross_validate(gbm_model, X, y, cv = 5, scoring = ['accuracy', 'f1', 'roc_auc'])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

gbm_params = {
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 8, 10],
    'n_estimators': [100, 500, 1000],
    'subsample': [1, 0.5, 0.7]  # Alt gözlem seçimi: Sırasıyla; hepsini, yarısını ve %70'ini seç.
    }

gbm_best_grid = GridSearchCV(gbm_model, gbm_params, cv = 5, n_jobs = -1, verbose = True)
gbm_best_grid.fit(X, y)

gbm_best_grid.best_params_

gbm_final = gbm_model.set_params(**gbm_best_grid.best_params_, random_state = 17)
gbm_final.fit(X, y)

cv_results = cross_validate(gbm_model, X, y, cv = 5, scoring = ['accuracy', 'f1', 'roc_auc'])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

# XGBoost
xgboost_model = XGBClassifier(random_state = 17)

cv_results = cross_validate(xgboost_model, X, y, cv = 5, scoring = ['accuracy', 'f1', 'roc_auc'])


cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

xgboost_params = {
    'learning_rate': [0.1, 0.01],
    'max_depth': [5, 8, None],
    'n_estimators': [100, 500, 1000],
    'colsample_bytree': [None, 0.7, 1]
    }

xboost_best_grid = GridSearchCV(xgboost_model, xgboost_params, cv = 5, n_jobs = -1, verbose = True)
xboost_best_grid.fit(X, y)

xgboost_final = xgboost_model.set_params(**xboost_best_grid.best_params_, random_state = 17)
xgboost_final.fit(X, y)

cv_results = cross_validate(xgboost_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

# LightGBM
lgbm_model = LGBMClassifier(random_state = 17)

cv_results = cross_validate(lgbm_model, X, y, cv = 5, scoring = ["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

lgbm_params = {
    'learning_rate': [0.01, 0.1],
    'n_estimators': [100, 300, 500, 1000],
    'colsample_bytree': [0.5, 0.7, 1]
    }

lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv = 5, n_jobs = -1, verbose = True)
lgbm_best_grid.fit(X, y)

lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state = 17)
cv_results = cross_validate(lgbm_best_grid, X, y, cv = 5, scoring = ['accuracy', 'f1', 'roc_auc'])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

# CatBoost
catboost_model = CatBoostClassifier(random_state = 17, verbose = True)

cv_results = cross_validate(catboost_model, X, y, cv = 5, scoring = ['accuracy', 'f1', 'roc_auc'])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

catboost_params = {
    'iterations': [200, 500],
    'learning_rate': [0.01, 0.1],
    'depth': [3, 6]
    }

catboost_best_grid = GridSearchCV(catboost_model, catboost_params, cv = 5, n_jobs = -1, verbose = True)
catboost_best_grid.fit(X, y)

catboost_final = catboost_model.set_params(**catboost_best_grid.best_params_, random_state = 17)
catboost_final.fit(X, y)

cv_results = cross_validate(catboost_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()


# Feature Importence
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


plot_importance(catboost_final, X)

# RandomSearchCV
rf_model = RandomForestClassifier(random_state = 17)

rf_random_params = {
    'max_depth': np.random.randint(5, 50, 10),
    'max_features': [3, 5, 7, 'auto', 'sqrt'],
    'min_samples_split': np.random.randint(2, 5, 50),
    "n_estimators": [int(x) for x in np.linspace(start = 200, stop = 1500, num = 10)]
    }

rf_random = RandomizedSearchCV(estimator = rf_model, param_distributions = rf_random_params, n_iter = 100, cv = 3,
                               verbose = True, random_state = 42, n_jobs = -1)

rf_random.fit(X, y)

rf_final = rf_model.set_params(**rf_random.best_params_, random_state = 17)

cv_results = cross_validate(rf_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

# Analyzing Model Complexity with Learning Curves

rf_val_params = [["max_depth", [5, 8, 15, 20, 30, None]],
                 ["max_features", [3, 5, 7, "auto"]],
                 ["min_samples_split", [2, 5, 8, 15, 20]],
                 ["n_estimators", [10, 50, 100, 200, 500]]]


rf_model = RandomForestClassifier(random_state=17)

for i in range(len(rf_val_params)):
    val_curve_params(rf_model, X, y, rf_val_params[i][0], rf_val_params[i][1])

rf_val_params[0][1]