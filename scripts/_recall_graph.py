#! /usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sys

# 입력받은 CSV 파일 경로
csv_file = sys.argv[1]

# CSV 파일 읽기
data = pd.read_csv(csv_file)

# 특정 ef 값을 가진 행만 필터링 (예: ef 값이 64인 경우)
ef_value = 64
filtered_data = data[data['ef'] == ef_value]

# 특정 열을 x, y, z축으로 설정
x = filtered_data['efConstruction'].values
y = filtered_data['maxConnections'].values
z = filtered_data['qps'].values

# X, Y, Z를 2차원으로 변환 (그리드 형식)
X, Y = np.meshgrid(np.unique(x), np.unique(y))
Z = np.zeros_like(X, dtype=float)

# 각 (x, y)에 해당하는 z 값을 Z에 채우기
for i in range(len(x)):
    xi = np.where(np.unique(x) == x[i])[0][0]
    yi = np.where(np.unique(y) == y[i])[0][0]
    Z[yi, xi] = z[i]

# 3D 그래프 그리기
fig = plt.figure(figsize=(12, 6))

# 첫 번째 그래프: 표면 그래프
ax1 = fig.add_subplot(121, projection='3d')
surf = ax1.plot_surface(X, Y, Z, cmap='viridis')

# 축 라벨 설정
ax1.set_xlabel('efConstruction')
ax1.set_ylabel('maxConnections')
ax1.set_zlabel('qps')

# 축의 범위 설정
ax1.set_xlim(X.min(), X.max())
ax1.set_ylim(Y.min(), Y.max())
ax1.set_zlim(Z.min(), Z.max())

# 두 번째 그래프: 산점도 그래프
ax2 = fig.add_subplot(122, projection='3d')
scatter = ax2.scatter(x, y, z, c=z, cmap='viridis')

# 축 라벨 설정
ax2.set_xlabel('efConstruction')
ax2.set_ylabel('maxConnections')
ax2.set_zlabel('qps')

# 축의 범위 설정
ax2.set_xlim(X.min(), X.max())
ax2.set_ylim(Y.min(), Y.max())
ax2.set_zlim(Z.min(), Z.max())

# 그래프 보여주기
plt.show()
