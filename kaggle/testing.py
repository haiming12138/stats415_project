import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer
from sklearn.compose import TransformedTargetRegressor

trans = PowerTransformer()

arr = np.random.rand(100).reshape(-1, 1)
trans_arr = trans.fit_transform(arr)
inverse_arr = trans.inverse_transform(trans_arr)
idx = np.where(arr != inverse_arr)[0]

df = pd.DataFrame({
    'original': arr.flatten(),
    'inverted': inverse_arr.flatten()
})

df.iloc[idx, :].to_csv('./temp.csv', index=False)