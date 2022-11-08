# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# ## 2. Multiple Linear Regression

# ### 원-핫 인코딩

import pandas as pd

dataset = pd.read_csv('MultipleLinearRegressionData.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

X

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(drop='first'), [2])], remainder='passthrough')
X = ct.fit_transform(X)
X

# ### 데이터 세트 분리

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# ### 학습 (다중 선형 회귀)

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train)

# ### 예측 값과 실제 값 비교(테스트 세트)

y_pred = reg.predict(X_test)
y_pred

y_test

reg.coef_

reg.intercept_

# ### 모델 평가

reg.score(X_train, y_train) # 훈련 세트

reg.score(X_test, y_test) # 테스트 세트

# ### 다양한 평가 지표 (회귀 모델)

#  1. MAE (Mean Absolute Error) : (실제 값과 예측 값) 차이의 절대값
#  2. MSE (Mean Squared Error) : 차이의 제곱
#  3. RMSE (Root Mean Squared Error) : 차이의 제곱에 루트
#  4. R2: 결정 계수
#  
#  > R2는 1에 가까울수록, 나머지는 0에 가까울수록 좋음

from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test, y_pred) # 실제 값, 예측값을 넣었을 때 MAE

from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, y_pred) # 실제 값, 예측값을 넣었을 때 MSE

from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, y_pred, squared=False) # 실제 값, 예측값을 넣었을 때 RMAE

from sklearn.metrics import r2_score
r2_score(y_test, y_pred) #R2 - 모델 평가 데스트 세트 훈련 결과와 같음 => R2와 선형모델 중 택 1
