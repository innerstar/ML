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

# # 5. K-Means

import os # 경고 해결
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("KMeansData.csv")
dataset [:5]

X = dataset.iloc[:, :].values # == X = dataset.values or X = dataset.to_numpy[] ← 공식 홈페이지 권장
X[:5]

# ### 데이터 시각화 (전체 데이터 분포 확인)

plt.scatter(X[:, 0], X[:, 1]) # x축: hour, y축: score
plt.title('Score by hours')
plt.xlabel('hours')
plt.ylabel('score')
plt.show()

# ### 데이터 시각화 (축 범위 통일)

plt.scatter(X[:, 0], X[:, 1]) # x축: hour, y축: score
plt.title('Score by hours')
plt.xlabel('hours')
plt.xlim(0, 100)
plt.ylabel('score')
plt.ylim(0, 100)
plt.show()

# ### 피처 스케일링 (Feature Scaling)

# x축과 y축 범주를 동일화
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
X[:5]

# ### 데이터 시각화 (스케일링된 데이터)

plt.figure(figsize=(5,5)) # 정사각형 틀
plt.scatter(X[:, 0], X[:, 1])
plt.title('Score by hours')
plt.xlabel('hours')
plt.ylabel('score')
plt.show()

# ### 최적의 K값 찾기 (엘보우 방식 Elbow Method)

# +
from sklearn.cluster import KMeans
inertia_list = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=0)
    kmeans.fit(X) # 지도학습과 다르게 y값이 없음
    inertia_list.append(kmeans.inertia_) # 각 지점으로부터 클러스터 중심[centroid]까지의 거리 제곱의 합
    
plt.plot(range(1,11), inertia_list)
plt.title('Elbow Method')
plt.xlabel('n_clusters')
plt.ylabel('inertia')
plt.show()
# -

# ### 최적의 K (4) 값으로 KMeans 학습

K = 4 # 최적의 k 값

kmeans = KMeans(n_clusters=K, random_state=0)
# kmeans.fit(X)
# 어떤 cluster에 속하는지 나타내기 위한 값을 반환
y_kmeans = kmeans.fit_predict(X)

y_kmeans

# ### 데이터 시각화 (최적의 K)

centers = kmeans.cluster_centers_ # 클러스터의 중심점[centroid]의 좌표 값들을 가짐
centers

           # x좌표, y좌표

# +
for cluster in range(K):
    plt.scatter(X[y_kmeans == cluster, 0], X[y_kmeans == cluster, 1], s=100, edgecolor='black') # 0은 공부 시간 정보, 1은 score 정보 → 각 데이터 출력
    plt.scatter(centers[cluster, 0], centers[cluster, 1], s=300, edgecolor='black', color='yellow', marker='s') # 중심점 네모 출력
    plt.text(centers[cluster, 0], centers[cluster, 1], cluster, va='center', ha='center') # 클러스터 네모 출력
    
plt.title('Score by hours')
plt.xlabel('hours')
plt.ylabel('score')
plt.show()
# -

# ### 데이터 시각화 (스케일링 원복)

X_org = sc.inverse_transform(X) # Feature Scaling 된 데이터를 원복
X_org[:5]

centers_org = sc.inverse_transform(centers)
centers_org

# +
for cluster in range(K):
    plt.scatter(X_org[y_kmeans == cluster, 0], X_org[y_kmeans == cluster, 1], s=100, edgecolor='black') # 0은 공부 시간 정보, 1은 score 정보 → 각 데이터 출력
    plt.scatter(centers_org[cluster, 0], centers_org[cluster, 1], s=300, edgecolor='black', color='yellow', marker='s') # 중심점 네모 출력
    plt.text(centers_org[cluster, 0], centers_org[cluster, 1], cluster, va='center', ha='center') # 클러스터 네모 출력
    
plt.title('Score by hours')
plt.xlabel('hours')
plt.ylabel('score')
plt.show()
