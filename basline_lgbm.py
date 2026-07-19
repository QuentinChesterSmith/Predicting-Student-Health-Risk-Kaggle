from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score

import lightgbm as lgb

import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")

training_data = pd.read_csv(r'/kaggle/input/competitions/playground-series-s6e7/train.csv')

X = training_data.drop(columns=["health_condition", "id"])
y = training_data["health_condition"]

label_encoder = LabelEncoder()
#y_encoded = label_encoder.fit_transform(y)

numerical_columns = X.select_dtypes("number").columns
categorical_columns = X.select_dtypes("object").columns

numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median", add_indicator=True))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("onehotencoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numerical", numerical_pipeline, numerical_columns),
    ("categorical", categorical_pipeline, categorical_columns)
])

clf = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, num_leaves=63, class_weight="balanced", verbosity=-1)

model = Pipeline([
    ("pre", preprocessor),
    ("clf", clf)
])

kfold = StratifiedKFold(n_splits=5)
accs = []

for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    print(f"Fold {fold}, Balanced Accuracy: {balanced_acc}")
    accs.append(balanced_acc)
 
print(f"Mean Accuracy: {np.mean(accs)} +/- {np.std(accs)}")
