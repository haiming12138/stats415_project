import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, \
    StandardScaler, RobustScaler, PowerTransformer
from sklearn.decomposition import PCA
from read_data_util import NUM_COLS, CAT_COLS, get_data
import time

# Prepare data for full model
X, y = get_data('./datasets/data.csv')

transformer = ColumnTransformer(
    [('encode', OneHotEncoder(drop='first'), CAT_COLS),
     ('standardize', RobustScaler(), NUM_COLS)],
    remainder='passthrough',
    n_jobs=-1,
    verbose_feature_names_out=False
)

pipe = Pipeline(
    [('transform', transformer),
     ('clf', SVC(kernel='rbf', 
                 gamma='scale', 
                 class_weight='balanced'))
    ],
    memory='./.cache'
)

params= {
    'clf__C': [1], 
    'clf__gamma': [10]
}


grid = GridSearchCV(
    estimator=pipe, 
    param_grid=params,
    scoring='accuracy',
    cv=5,
    n_jobs=-1,
)

start = time.time()
grid.fit(X, y)
end = time.time()
print(end - start)
print(grid.best_score_)

