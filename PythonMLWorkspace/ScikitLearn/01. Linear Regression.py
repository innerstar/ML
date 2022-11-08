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

# #1.Linear Regression
# ### 공부 시간에 따른 시험 점수

import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('LinearRegressionData.csv')

dataset.head()

X = dataset.iloc[:, :-1].values # 처음부터 마지막 컬럼 직전까지 데이터 [독립 변수 - 원인]
Y = dataset.iloc[:, -1].values # 마지막 컬럼 데이터 [종속 변수 - 결과]

X, Y

from sklearn.linear_model import LinearRegression
reg = LinearRegression() # 객체 생성
reg.fit(X, Y) # fit -> 훈련시킨다는 뜻, 학습 [모델 생성]

y_pred = reg.predict(X) # x에 대한 예측 값 출력
y_pred

plt.scatter(X, Y, color='blue') # 산점도 그래프
plt.plot(X, y_pred, color='green') # 선 그래프
plt.title('Score by hours') # 제목
plt.xlabel('hours')
plt.ylabel('score')
plt.show()

print('9시간 공부했을 때 예상 점수: ', reg.predict([[9]]))

reg.coef_ # 기울기 [m]

reg.intercept_ # y 절편 [b]

# y = mx + b -> y = 10.4436x - 0.2184

# ### 데이터 세트 분리

import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('LinearRegressionData.csv') # 입력 변수가 공부시간 1개인 데이터 set
dataset

X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0) # testset의 크기 20 훈련 80 : 테스트 20으로 분리

X, len(X) # 전체 데이터 X, 개수

X_train, len(X_train) # 훈련 세트 X, 개수

X_test, len(X_test) # 테스트 세트 X, 개수

Y, len(Y)

Y_train, len(Y_train)

Y_test, len(Y_test)

# ### 분리된 데이터를 통한 모델링

from sklearn.linear_model import LinearRegression
reg = LinearRegression()

reg.fit(X_train, Y_train) # 훈련 세트로 학습

# ### 데이터 시각화 (훈련 세트)

plt.scatter(X_train, Y_train, color='blue') # 산점도 그래프
plt.plot(X_train, reg.predict(X_train), color='green') # 선 그래프
plt.title('Score by hours(train data)') # 제목
plt.xlabel('hours')
plt.ylabel('score')
plt.show()

# ### 데이터 시각화 (테스트 세트)

plt.scatter(X_test, Y_test, color='blue') # 산점도 그래프
plt.plot(X_train, reg.predict(X_train), color='green') # 선 그래프, 모델을 만들었던 훈련 세트는 그대로 둠
plt.title('Score by hours(test data)') # 제목
plt.xlabel('hours')
plt.ylabel('score')
plt.show()

reg.coef_

reg.intercept_

# ### 모델 평가

reg.score(X_test, Y_test) # test set를 통한 모델 평가 -> (0~1) 1에 가까울 수록 점수 높음

reg.score(X_train, Y_train) # train set을 통한 모델 평가

# ### 경사 하강법 (Gradient Descent)

# max_iter: 훈련 세트 반복 횟수(Epoch 횟수)
#
# eta0: 학습률(learning rate)

# +
from sklearn.linear_model import SGDRegressor #SGD: Stochastic Gradient Descent 확률적 경사 하강법

# 지수표기법
# 1e-3: 0.001 [10^-3]
# 1e-4: 0.0001 [10^-4]

# sr = SGDRegressor(max_iter=100, eta0=1e-4, random_state=0, verbose=1)
sr = SGDRegressor()
sr.fit(X_train, Y_train)
# -

plt.scatter(X_train, Y_train, color='blue') # 산점도 그래프
plt.plot(X_train, sr.predict(X_train), color='green') # 선 그래프, 모델을 만들었던 훈련 세트는 그대로 둠
plt.title('Score by hours(test data, SGD)') # 제목
plt.xlabel('hours')
plt.ylabel('score')
plt.show()

sr.coef_, sr.intercept_ # 절편 확인

sr.score(X_test, Y_test) # 테스트 세트를 통한 모델 평가

sr.score(X_train, Y_train) # 훈련 세트를 통한 모델 평가 (원래 점수는 훈련세트 > 테스트 세트가 이상적)

 


