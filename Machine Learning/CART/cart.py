################################################
# Decision Tree Classification: CART
################################################

# 1. Exploratory Data Analysis
# 2. Data Preprocessing & Feature Engineering
# 3. Modeling using CART
# 4. Hyperparameter Optimization with GridSearchCV
# 5. Final Model
# 6. Feature Importance
# 7. Analyzing Model Complexity with Learning Curves (BONUS)
# 8. Visualizing the Decision Tree
# 9. Extracting Decision Rules
# 10. Extracting Python/SQL/Excel Codes of Decision Rules
# 11. Prediction using Python Codes
# 12. Saving and Loading Model


import warnings
import joblib
import pydotplus
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, StratifiedKFold, validation_curve
from skompiler import skompile


pd.set_option('display.max_columns', 1881)
pd.set_option('display.width', 1881)
# warnings.simplefilter(action = 'ignore', category = Warning)


# 3. Modeling using CART
df = pd.read_csv(r'C:\Users\Souljah_Pc\PycharmProjects\courses\Machine Learning\datasets\diabetes.csv')

y = df['Outcome']
X = df.drop('Outcome', axis = 1)

cart_model = DecisionTreeClassifier(random_state = 1)
cart_model.fit(X, y)

# y_pred (Confusion Matrix için):
y_pred = cart_model.predict(X)

# y_prob (AUC için):
y_prob = cart_model.predict_proba(X)[:, 1]

# Confusion Matrix:
print(classification_report(y, y_pred))

# AUC:
roc_auc_score(y, y_prob)


# Success Validation with Holdout:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 45)

cart_model = DecisionTreeClassifier(random_state = 17)
cart_model.fit(X_train, y_train)

# Train Hatası:
y_pred = cart_model.predict(X_train)
y_prob = cart_model.predict_proba(X_train)[:, 1]
print(classification_report(y_train, y_pred))
roc_auc_score(y_train, y_prob)

# Test Hatası:
y_pred = cart_model.predict(X_test)
y_proba = cart_model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))
roc_auc_score(y_test, y_proba)

# Success Validation with CV:
cart_model = DecisionTreeClassifier(random_state = 17)
cart_model.fit(X, y)
# cart_model.fit(X, y) # modele CV yapacağız. cross_validate her cross için fit yapıyor. O yüzden gerek yok.

cv_results = cross_validate(cart_model, X, y, cv = 5, scoring = ['accuracy', 'f1', 'roc_auc'])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

# Hiperparametre optimizasyonu (GridSearchCV):
cart_model.get_params()

cart_params = {
    'max_depth': range(1, 11),
    'min_samples_split': range(2, 20)}

# GridSearchCV örneğinden oluşturulan model, en iyi parametrelere sahip modeli verir.
cart_best_grid = GridSearchCV(cart_model, cart_params, cv = 5, n_jobs = -1, verbose = 1)
cart_best_grid.fit(X, y)

# En iyi hiperparametreler:
cart_best_grid.best_params_
# Default olarak scoring = 'accuracy' metriğine göre skor:
cart_best_grid.best_score_

# 5. Final Modeli Oluştur:
cart_final = DecisionTreeClassifier(**cart_best_grid.best_params_, random_state = 17)
cart_final.fit(X, y)
cart_final.get_params()

# Alternatif:
cart_final = cart_model.set_params(**cart_best_grid.best_params_)
cart_final.fit(X, y)

cv_results = cross_validate(cart_final, X, y, cv = 5, scoring = ['accuracy', 'f1', 'roc_auc'])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

# Feature Importence:
cart_final.feature_importances_  # Özelliklerin önem skorları


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


# 7. Analyzing Model Complexity with Learning Curves:
train_score, test_score = validation_curve(cart_final, X, y,
                                           param_name = 'max_depth',
                                           param_range = range(1, 11),
                                           scoring = 'roc_auc',
                                           cv = 10)

mean_train_score = np.mean(train_score, axis = 1)
mean_test_score = np.mean(test_score, axis = 1)

plt.plot(range(1, 11), mean_train_score, label = 'Training Score', color = 'b')
plt.plot(range(1, 11), mean_test_score, label = 'Test Score', color = 'g')

plt.title('Test Curve for CART')
plt.xlabel("Number of max_depth")
plt.ylabel("AUC")
plt.tight_layout()
plt.legend(loc = 'best')
plt.show()


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


val_curve_params(cart_final, X, y, "max_depth", range(1, 11), scoring="f1")

cart_val_params = [["max_depth", range(1, 11)], ["min_samples_split", range(2, 20)]]

for i in range(len(cart_val_params)):
    val_curve_params(cart_model, X, y, cart_val_params[i][0], cart_val_params[i][1])


# 8. Visualizing the Desicion Tree
def tree_graph(model, col_names, file_name):
    tree_str = export_graphviz(model, feature_names=col_names, filled=True, out_file=None)
    graph = pydotplus.graph_from_dot_data(tree_str)
    graph.write_png(file_name)


tree_graph(model=cart_final, col_names=X.columns, file_name="cart_final.png")

# 9. Extracting Decision Rules
tree_rules = export_text(cart_final, feature_names=list(X.columns))
print(tree_rules)

# 10. Extracting Python Codes of Decision Rules
print(skompile(cart_final.predict).to('python/code'))

# pip install sqlalchemy
print(skompile(cart_final.predict).to('sqlalchemy/sqlite'))

print(skompile(cart_final.predict).to('excel'))


# Prediction using Python Codes
def predict_with_rules(x):
    return (((((0 if x[0] <= 7.5 else 1) if x[5] <= 30.949999809265137 else 0 if x[6] <=
    0.5005000084638596 else 0) if x[5] <= 45.39999961853027 else 1 if x[2] <=
    99.0 else 0) if x[7] <= 28.5 else (1 if x[5] <= 9.649999618530273 else
    0) if x[5] <= 26.350000381469727 else (1 if x[1] <= 28.5 else 0) if x[1
    ] <= 99.5 else 0 if x[6] <= 0.5609999895095825 else 1) if x[1] <= 127.5
     else (((0 if x[5] <= 28.149999618530273 else 1) if x[4] <= 132.5 else
    0) if x[1] <= 145.5 else 0 if x[7] <= 25.5 else 1 if x[7] <= 61.0 else
    0) if x[5] <= 29.949999809265137 else ((1 if x[2] <= 61.0 else 0) if x[
    7] <= 30.5 else 1 if x[6] <= 0.4294999986886978 else 1) if x[1] <=
    157.5 else (1 if x[6] <= 0.3004999905824661 else 1) if x[4] <= 629.5 else 0
    )


x = [6, 148, 70, 35, 0, 30, 0.62, 50]

predict_with_rules(x)

# 12. Saving and Loading Model
joblib.dump(cart_final, 'cart_final.pkl')
cart_model_from_disc = joblib.load("cart_final.pkl")