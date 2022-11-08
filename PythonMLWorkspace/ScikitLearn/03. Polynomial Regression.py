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

# # 3. Polynomial Regression

# ### 공부 시간에 따른 시험 점수 (우등생)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('PolynomialRegressionData.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# # 3-1 단순 선형 회귀 (Simple Linear Regression)

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X, y) # 전체 데이터로 학습

# ### 데이터 시각화 (전체)

plt.scatter(X, y, color='blue') #산점도
plt.plot(X, reg.predict(X), color='green' ) #x에 대한 예측값이 y값으로 대입
plt.title('Score by hours (genius)') #제목
plt.xlabel('hours') # X 축 이름
plt.ylabel('score') # Y 축 이름
plt.show()

reg.score(X, y) # 전체 데이터를 통한 모델 평가

# ## 3-2. 다항 회귀 (Polynomial Regression)

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4) # 2차
X_poly = poly_reg.fit_transform(X) # fit: 새롭게 만들 features 들의 조합을 찾음, transform: 실제로 데이터를 변환 => 따로따로 호출 가능하지만 한 번에 호출 가능
X_poly[:5] # Features 확장
      #x^0    x^1    x^2 (← degree 2차, 4차면 5개(x^0 ~ x^4 확장)

X[:5]

poly_reg.get_feature_names_out() # feature가 한 개라서 단순

lin_reg = LinearRegression()
lin_reg.fit(X_poly, y) # 변환된 x와 y를 가지고 모델 생성 [학습]

# ### 데이터 시각화 (변환된 X와 y)

plt.scatter(X, y, color='blue')
plt.plot(X, lin_reg.predict(poly_reg.fit_transform(X)), color='green')
plt.title('Score by hours (genius)') #제목
plt.xlabel('hours') # X 축 이름
plt.ylabel('score') # Y 축 이름
plt.show()

# x의 최소값에서 최대값까지의 범위를 0.1 단위로 잘라서 데이터 생성
X_range = np.arange(min(X), max(X), 0.1) # numpy 함수 이용 → 선을 유연하게
X_range

X_range.shape # X_range의 데이터, 1차원 형식의 데이터가 46개 존재

X[:5] # 2차원 → X_range의 1차원 데이터 형식을 맞출 필요성

X.shape

X_range = X_range.reshape(-1, 1) # -1 (low 개수 자동 계산) = len(X_renge), 1 (column 개수)
X_range.shape

X_range[:5]

plt.scatter(X, y, color='blue')
plt.plot(X_range, lin_reg.predict(poly_reg.fit_transform(X_range)), color='green')
plt.title('Score by hours (genius)') #제목
plt.xlabel('hours') # X 축 이름
plt.ylabel('score') # Y 축 이름
plt.show()

# ### 공부 시간에 따른 시험 성적 예측

reg.predict([[2]]) # 2시간을 공부했을 때 선형 회귀 모델의 예측

lin_reg.predict(poly_reg.fit_transform([[2]])) # 2시간을 공부했을 때 다항 회귀 모델의 예측

lin_reg.score(X_poly, y)
