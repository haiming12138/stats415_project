import numpy as np
import pandas as pd
import joblib
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

model = joblib.load('./models/model_full.sav')
print(type(model))