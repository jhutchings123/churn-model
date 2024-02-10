"""
Project Brief: Please write a short (2-3 paragraph) project brief using the following prompts. 
- If this initial work formed the basis of a longer-term project, what additional steps would you take
   to increase performance and create a production-ready model? 
- Assume an existing churn prediction model is currently deployed. How would you determine when your model
   is ready to replace the existing model in production? 
- After putting your model in production, what potential issues could arise a few months down the road
   and how would you address them?
- Any other limitations or considerations that you’d like to share
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pycaret.classification import *



ev_df = pd.read_csv("/Users/jh/Documents/Evolve/listing_churn_data.csv", header = 0)

# Evaluate the quality of data

# Initial read
print(ev_df)
print(ev_df.dtypes)

# Histograms of the features
for col in ev_df.columns:
    plt.hist(ev_df[col], color = "coral", edgecolor = "darkblue")
    plt.title(col)
    plt.grid()
    plt.show()

"""
General observations about the data:
- The data is not scaled
- All of the variables have a normal distribution (is that for real? probably not, but at least it looks clean, so I'll run with it)
- Rate of churn is low
"""

# Look at the number of churn = 1 vs. churn = 0
print(ev_df["churn"].value_counts())
# 117K where churn = 0
# 8.4K where churn = 1

# Churn rate
print("Churn rate: " + str(ev_df["churn"][ev_df["churn"] == 1].sum()/ev_df.shape[0]))
# Churn rate: 0.066976

# Churn rate is on the lower side, so this data is not "balanced", meaning that the binary outcome variable is not split close to 50-50

# Look at the mean value of the variables by churn indication
for col in ev_df.columns[:-1]:
    print(col + " churn mean: " + str(round(ev_df[col][ev_df["churn"] == 1].mean(), 2)))
    print(col + " not churn mean: " + str(round(ev_df[col][ev_df["churn"] == 0].mean(), 2)))
    print("")

# Take a look at the distribution of the variables by churn indication
for col in ev_df.columns[:-1]:
    plt.hist(ev_df[col][ev_df["churn"] == 0], alpha = 0.35, color = "red", edgecolor = "darkred", bins = 50)
    plt.hist(ev_df[col][ev_df["churn"] == 1], alpha = 0.35, color = "blue", edgecolor = "darkblue", bins = 50)
    plt.grid()
    plt.title(col)
    plt.legend(["Not Churn", "Churn"])
    plt.show()

# Not much difference in the mean values of the variables when split between churn and not churn
# Hopefully the model will find some signal, but at this point my expectations for an accurate model are low

# A question that should be asked: is it possible to engineer any features from this data that would be predictive?
# Answer: nothing stands out, so move along without engineered features


# ------------------------------------ #
# ------ Model creation process ------ #
# ------------------------------------ #

"""
Steps to creating a model:
1) Split into train and test sets
- The Train Set
 * Get a balanced train set, so start by sampling from the data where churn = 1, get 80% of those observations
 * Get a random sample of the data where churn = 0 that is the same size as the sample where churn = 1
- The Test Set
 * Test set will be very imbalanced and consist of all the data not used in the train set
2) Model selection
- Use GridSearchCV from sklearn to get optimal hyperparameters for each model
- Test various machine learning models - Random Forest, GBM, Catboost, KNN
- Compare the models using Cohen's kappa
"""

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import cohen_kappa_score, make_scorer
kappa_scorer = make_scorer(cohen_kappa_score)

train_samp = ev_df[ev_df["churn"] == 1].sample(frac = 0.75, random_state = 10).index
train_samp = train_samp.append(ev_df[ev_df["churn"] == 0].sample(n = len(train_samp)).index)
train_data = ev_df.iloc[train_samp, :]
test_data = ev_df.iloc[ev_df.index.isin(train_samp) == False, :]
X_train = train_data.iloc[:, :-1]
y_train = train_data.iloc[:, -1]
X_test = test_data.iloc[:, :-1]
y_test = test_data.iloc[:, -1]

# Function to get models with optimized parameters and accuracy results based on Cohen's Kappa
# For Cohen's Kappa, the closer to 1 the better, the closer to 0 the worse
def model_grid_search(model, params):
    param_grid = params
    grid_search = GridSearchCV(estimator = model,
                               param_grid = param_grid,
                               scoring = kappa_scorer,
                               n_jobs = -1,
                               cv = 5)
    grid_search.fit(X_train, y_train)
    grid_search.score(X_test, y_test)
    y_hat = grid_search.predict(X_test)
    cm = confusion_matrix(y_true = y_test, y_pred = y_hat)
    print("")
    print(type(model).__name__)
    print("Confusion Matrix:")
    print(cm)
    disp = ConfusionMatrixDisplay(confusion_matrix = cm)
    disp.plot()
    plt.title(type(model).__name__)
    plt.show()

    try:
        print("Best parameters:")
        print(grid_search.best_params_)
    except:
        print("")
    try:
        print("Kappa: " + str(round(grid_search.best_score_, 4)))
    except:
        print("")
    try:
        feat_imp = pd.Series(grid_search.best_estimator_.feature_importances_ , ev_df.columns[:-1].to_list()).sort_values(ascending = False)
        feat_imp.plot(kind = "bar", title = "Feature Importance")
        plt.ylabel("Feature Importance Score")
        plt.title(type(model).__name__)
        plt.show()
    except:
        print("")


# KNN model
params = {"n_neighbors": [10, 20, 30, 40, 50, 100, 200]}
model_grid_search(model = KNeighborsClassifier(), params = params)

# GradientBoostingClassifier
params = {"n_estimators": [4, 6, 8, 10],
          "max_depth": [6, 10],
          "learning_rate": [0.01, 0.1, 1]}
model_grid_search(model = GradientBoostingClassifier(), params = params)

# RandomForestClassifier
params = {"n_estimators": [50, 100, 200, 500],
          "max_depth": [6, 10],
          "criterion": ["entropy", "gini"]}
model_grid_search(model = RandomForestClassifier(), params = params)

# catboost
params = {"iterations": [1, 5, 10],
          "learning_rate": [0.01, 0.1, 1],
          "depth": [3, 6, 9]}
model_grid_search(model = CatBoostClassifier(), params = params)

"""
Kappa results:
KNN - 0.0092
GradientBoostingClassifier - 0.2324
RandomForestClassifier - 0.2371
CatBoostClassifier - 0.2328

Although none of these are stellar, they could still be somewhat useful
The Random Forest had the highest Kappa, implying that it is the best model, but the GBM and Catbooster were very nearly as good as the Random Forest

Interestingly enough, the feature with the greatest importance in all of the models was number_of_reviews
This is insightful and perhaps an effort to increase the number of reviews that property owners are given would help retain them
"""


