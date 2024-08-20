#! /usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sys

def load_and_filter_data(csv_file, ef_value):
    """CSV 파일을 읽고, 특정 ef 값을 가진 행만 필터링합니다."""
    data = pd.read_csv(csv_file)
    filtered_data = data[data['ef'] == ef_value]
    return filtered_data

def prepare_grid_data(filtered_data):
    """필터링된 데이터를 사용하여 X, Y, Z 그리드를 준비합니다."""
    x = filtered_data['efConstruction'].values
    y = filtered_data['maxConnections'].values
    recall_z = filtered_data['recall'].values 
    qps_z = filtered_data['qps'].values

    X, Y = np.meshgrid(np.unique(x), np.unique(y))
    Z_recall = np.zeros_like(X, dtype=float)
    Z_qps = np.zeros_like(X, dtype=float)

    for i in range(len(x)):
        xi = np.where(np.unique(x) == x[i])[0][0]
        yi = np.where(np.unique(y) == y[i])[0][0]
        Z_recall[yi, xi] = recall_z[i]
        Z_qps[yi, xi] = qps_z[i]

    return X, Y, Z_recall, Z_qps, x, y, recall_z, qps_z

def plot_3d_graphs(X, Y, Z_recall, Z_qps, x, y, recall_z, qps_z):
    """Recall 및 QPS에 대한 3D 그래프를 플로팅합니다."""
    fig = plt.figure(figsize=(16, 16))

    # 첫 번째 줄: recall에 대한 그래프
    ax1 = fig.add_subplot(221, projection='3d')
    surf_recall = ax1.plot_surface(X, Y, Z_recall, cmap='viridis')
    ax1.set_title('Recall (surface)')
    ax1.set_xlabel('efConstruction')
    ax1.set_ylabel('maxConnections')
    ax1.set_zlabel('Recall')
    ax1.set_xlim(X.min(), X.max())
    ax1.set_ylim(Y.min(), Y.max())
    ax1.set_zlim(Z_recall.min(), Z_recall.max())

    ax2 = fig.add_subplot(222, projection='3d')
    scatter_recall = ax2.scatter(x, y, recall_z, c=recall_z, cmap='viridis')
    ax2.set_title('Recall (scatter)')
    ax2.set_xlabel('efConstruction')
    ax2.set_ylabel('maxConnections')
    ax2.set_zlabel('Recall')
    ax2.set_xlim(X.min(), X.max())
    ax2.set_ylim(Y.min(), Y.max())
    ax2.set_zlim(recall_z.min(), recall_z.max())

    # 두 번째 줄: qps에 대한 그래프
    ax3 = fig.add_subplot(223, projection='3d')
    surf_qps = ax3.plot_surface(X, Y, Z_qps, cmap='viridis')
    ax3.set_title('QPS (surface)')
    ax3.set_xlabel('efConstruction')
    ax3.set_ylabel('maxConnections')
    ax3.set_zlabel('QPS')
    ax3.set_xlim(X.min(), X.max())
    ax3.set_ylim(Y.min(), Y.max())
    ax3.set_zlim(Z_qps.min(), Z_qps.max())

    ax4 = fig.add_subplot(224, projection='3d')
    scatter_qps = ax4.scatter(x, y, qps_z, c=qps_z, cmap='viridis')
    ax4.set_title('QPS (scatter)')
    ax4.set_xlabel('efConstruction')
    ax4.set_ylabel('maxConnections')
    ax4.set_zlabel('QPS')
    ax4.set_xlim(X.min(), X.max())
    ax4.set_ylim(Y.min(), Y.max())
    ax4.set_zlim(qps_z.min(), qps_z.max())

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    csv_file = sys.argv[1]
    EF_VALUE = 64  # 원하는 ef 값을 설정 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    filtered_data = load_and_filter_data(csv_file, EF_VALUE)
    X, Y, Z_recall, Z_qps, x, y, recall_z, qps_z = prepare_grid_data(filtered_data)
    plot_3d_graphs(X, Y, Z_recall, Z_qps, x, y, recall_z, qps_z)
