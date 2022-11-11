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

# # 4. Logistic Regression #

# ### 공부 시간에 따른 자격증 시험 합격 가능성 ###

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("LogisticRegressionData.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# ### 데이터 분리 ###

from sklearn.model_selection import train_test_split # split할 경우 4개의 데이터로 분리 됨
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) # test size = 20%

# ### 학습 (로지스틱 희귀 모델)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# ### 6시간 공부했을 때 예측 값

classifier.predict([[6]]) #classifier의 predict 메소드 사용 - 인자 (2차원 배열)
# 결과 1 = 합격 예측

classifier.predict_proba([[6]]) #합격할 확률 정보 출력
# 불합격 확률 14%, 합격 확률 88%

# ### 4시간 공부했을 때 예측 값

classifier.predict([[4]]) #classifier의 predict 메소드 사용 - 인자 (2차원 배열)
# 결과 0 = 불합격 예측

classifier.predict_proba([[4]]) #불합격할 확률 정보 출력
# 불합격 확률 62%, 합격 확률 38%

# ### 분류 결과 예측(테스트 세트)

# +
y_pred = classifier.predict(X_test)
y_pred # 예측 값

# 결과 [합격, 불합격, 합격, 합격] : testset 4개의 데이터 예측값 출력
# -

y_test # 실제 값 [테스트 세트]

X_test # 공부 시간 [테스트 세트]

classifier.score(X_test, y_test) # 모델 평가
# 전체 테스트 세트 4개 중에서 분류 예측을 올바르게 맞힌 개수 3개 -> 3/4 = 0.75

# ### 데이터 시각화 (훈련 세트)

X_range = np.arange(min(X), max(X), 0.1) # X의 최솟값에서 최대값까지를 0.1단위로 잘라서 데이터 생성
X_range

# y = ma + b
p = 1 / ( 1 + np.exp(-(classifier.coef_ * X_range + classifier.intercept_)) ) # exp 지수 의미 (exponential)
p

p.shape # 2차원 배열

X_range.shape # 1차원 배열

p = p.reshape(-1) # 1차원 배열 형태로 변경 -1의 의미 = len(p)
p.shape

plt.scatter(X_train, y_train, color='blue')
plt.plot(X_range, p, color='green') # X_range에 따라 변하는 p값 대입
plt.plot(X_range, np.full(len(X_range), 0.5), color='red') # X_range 개수만큼 0.5로 가득찬 배열 만들기
plt.title('Probability by hours')
plt.xlabel('hpurs')
plt.ylabel('P')
plt.show()

# ### 데이터 시각화 (테스트 세트)

plt.scatter(X_test, y_test, color='blue')
plt.plot(X_range, p, color='green') # X_range에 따라 변하는 p값 대입
plt.plot(X_range, np.full(len(X_range), 0.5), color='red') # X_range 개수만큼 0.5로 가득찬 배열 만들기
plt.title('Probability by hours(test)')
plt.xlabel('hpurs')
plt.ylabel('P')
plt.show()

# 4.5 시간 공부 했을 때 확률
# [모델에서는 51% 확률로 합격 예측, 실제로는 불합격]
classifier.predict_proba([[4.5]])

# ### 혼돈 행렬 (Confusion Matrix)

# +
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm

# TRUE MEGATIE [TN]      FALSE POSITIVE [FP]
# 불합격 예측            합격 예측
# 실제로 불합격          실제로 불합격

# FALSE MEGATIVE [FN]    TRUE POSITIVE [TP]
# 불합격 예측            합격 예측
# 실제로 합격            실제로 합격

# => 대각선을 살펴보면서 올바르게 예측한 데이터 개수와 틀리게 예측한 데이터 개수 확인 가능
