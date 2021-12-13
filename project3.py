import pickle
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from category_encoders import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

df_20 = pd.read_csv('./jeju_rent_car.csv', encoding='utf-8')


target = ['COST_D']
features = df_20.drop(columns=target).columns

X_train, X_test, y_train, y_test = train_test_split(df_20[features], df_20[target], test_size=0.2, random_state=4)

Linear_pipeline = make_pipeline(
    OneHotEncoder(),
    SimpleImputer(),
    StandardScaler(),
    LinearRegression(n_jobs=-1)
)

Linear_pipeline.fit(X_train, y_train)

with open('model.pkl', 'wb') as pickle_file:
    pickle.dump(Linear_pipeline, pickle_file)
