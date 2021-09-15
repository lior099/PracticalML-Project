import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.utils.fixes import loguniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, BaggingClassifier, AdaBoostClassifier
import xgboost as xgb

class ClassificationModels:
    def __init__(self, X_train, y_train, X_test, y_test, cv, scoring, n_jobs, model_selection="random", n_iter=10):
        self.cv = cv
        self.scoring = scoring
        self.X_train = X_train
        self.y_train = y_train
        # self.X_test = X_test
        # self.y_test = y_test
        self.model_selection = model_selection
        self.n_jobs = n_jobs
        self.n_iter = n_iter
        self.best_models = {}

    def logistic_regression(self):
        print("Logistic Regression")
        clf = LogisticRegression(max_iter=1000)
        params = {"solver": ['newton-cg', 'lbfgs', 'liblinear'],
                  "penalty": ['l2'],
                  "C": loguniform(1e-6, 1e6)}
        gcv = self.run_search_parameters_strategy(clf, params)
        return gcv

    def decision_tree(self):
        print("Decision Tree")
        clf = DecisionTreeClassifier()
        params = {
            'max_depth': [2, 3, 5, 10, 20],
            'min_samples_leaf': [5, 10, 20, 50, 100],
            'criterion': ["gini", "entropy"]
        }
        gcv = self.run_search_parameters_strategy(clf, params)
        return gcv

    def random_forest(self):
        print("Random Forest")
        clf = RandomForestClassifier()
        params = {
            'max_depth': [None, 20, 30, 40, 50],
            'min_samples_leaf': [2, 5, 10, 20],
            'criterion': ["gini", "entropy"]
        }
        gcv = self.run_search_parameters_strategy(clf, params)
        return gcv

    def naive_bayes(self):
        clf = GaussianNB()
        clf.fit(self.X_train, self.y_train)
        return clf

    def kmeans(self):
        print("K-Means")
        clf = KNeighborsClassifier()
        params = {"n_neighbors": range(2, 21, 2),
                  "weights": ['uniform', 'distance'],
                  "metric": ['euclidean', 'manhattan', 'minkowski']
        }
        gcv = self.run_search_parameters_strategy(clf, params)
        return gcv

    def svm(self):
        print("SVM")
        clf = SVC()
        params = {"kernel": ['poly', 'rbf', 'sigmoid'],
                  "C": loguniform(1e-6, 1e6),
                  "gamma": ['scale']
        }
        gcv = self.run_search_parameters_strategy(clf, params)
        return gcv

    def xgboost(self):
        print("Gradient Boosting")
        clf = GradientBoostingClassifier()
        params = {"n_estimators": [10, 100],
                  "learning_rate": loguniform(1e-6, 9e-1),
                  "subsample": [0.5, 0.7, 1.0],
                  "max_depth": [10, 20, 30]
        }
        gcv = self.run_search_parameters_strategy(clf, params)
        return gcv

    def real_xgboost(self):
        print("XGBoost")
        clf = xgb.XGBClassifier(use_label_encoder=False)
        params = {"n_estimators": [10, 100],
                  "learning_rate": loguniform(1e-6, 9e-1),
                  "subsample": [0.5, 0.7, 1.0],
                  "max_depth": [10, 20, 30]
                  }
        gcv = self.run_search_parameters_strategy(clf, params)
        return gcv

    def adaboost(self):
        print("AdaBoost")
        clf = AdaBoostClassifier()
        params = {"n_estimators": [10, 100],
                  "learning_rate": loguniform(1e-6, 9e-1),
                  "subsample": [0.5, 0.7, 1.0],
                  "max_depth": [10, 20, 30]
                  }
        gcv = self.run_search_parameters_strategy(clf, params)
        return gcv

    def bagging(self):
        print("Bagging")
        clf = BaggingClassifier()
        params = {"n_estimators": [10, 100],
                  "learning_rate": loguniform(1e-6, 9e-1),
                  "subsample": [0.5, 0.7, 1.0],
                  "max_depth": [10, 20, 30]
                  }
        gcv = self.run_search_parameters_strategy(clf, params)
        return gcv

    def run_search_parameters_strategy(self, clf, params):
        if self.model_selection == "random":
            gcv = RandomizedSearchCV(clf, params, cv=self.cv, scoring=self.scoring,
                                     return_train_score=True, verbose=2, n_iter=self.n_iter)
            gcv.fit(self.X_train, self.y_train)
        else:
            gcv = GridSearchCV(clf, params, cv=self.cv, scoring=self.scoring,
                               return_train_score=True, verbose=2, n_iter=self.n_iter)
            gcv.fit(self.X_train, self.y_train)
        return gcv

    def algorithms_options(self, algos):
        func_dict = {"logistic_regression": self.logistic_regression,
                     "svm": self.svm,
                     "decision_tree": self.decision_tree,
                     "random_forest": self.random_forest,
                     "xgboost": self.xgboost,
                     "kmeans": self.kmeans,
                     "real_xgboost": self.real_xgboost,
                     "adaboost": self.adaboost,
                     "bagging": self.bagging}
        algos = list(set(algos))
        for algo in algos:
            cur_algo_gcv = func_dict[algo]()
            self.best_models[algo] = {}
            self.best_models[algo]["best_estimator"] = cur_algo_gcv.best_estimator_
            self.best_models[algo]["best_params"] = cur_algo_gcv.best_params_
            self.best_models[algo]["best_score"] = cur_algo_gcv.best_score_


def load_dataset(path="dataset_phishing.csv"):
    df = pd.read_csv(path)
    df["status"] = df["status"].map({"legitimate": 0, "phishing": 1})
    del df["url"]
    y = np.array(df['status'].tolist()).ravel()
    X = df.loc[:, df.columns != 'status']
    return X, y


if __name__ == '__main__':
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
    scaler = preprocessing.StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    clf_models = ClassificationModels(X_train, y_train, X_test, y_test, 5, "f1", 1, n_iter=200)
    clf_models.algorithms_options(["logistic_regression", "svm", "decision_tree", "random_forest", "xgboost", "kmeans"])
    clf_models.algorithms_options(["real_xgboost", "adaboost", "bagging"])
    print(clf_models.best_models)
