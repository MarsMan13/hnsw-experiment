#! /usr/bin/env python3

import os
import sys
import json
import numpy as np
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 유추할 recall 값들
desired_recalls = np.arange(0.80, 1.00, 0.01)  # 0.80부터 0.99까지의 정수

def interpolate_qps(recall_points, qps_points, target_recalls):
    """ 주어진 recall 포인트에서의 QPS를 이용해 대상 recall 지점의 QPS를 유추합니다. """
    # recall 100에 대한 qps 값이 없을 경우, recall 100을 추가하고 qps는 0으로 설정
    if 1.0 not in recall_points:
        recall_points = np.append(recall_points, 1.0)
        qps_points = np.append(qps_points, 0.0)

    # 대상 recall 값들에서의 QPS를 유추
    interpolated_qps = np.interp(target_recalls, recall_points, qps_points)
    return interpolated_qps

def score_json_file(data):
    """ 주어진 JSON 파일에서 recall과 qps 값을 사용해 점수를 계산합니다. """
    recall_points = np.array([entry['recall'] for entry in data])
    qps_points = np.array([entry['qps'] for entry in data])

    # 대상 recall 값들에서의 QPS를 유추
    interpolated_qps = interpolate_qps(recall_points, qps_points, desired_recalls)

    # 유추된 QPS 값들의 합계를 점수로 계산
    score = np.sum(interpolated_qps)
    return score

def read_and_score_json_files(directory):
    """ 주어진 디렉토리 내 모든 JSON 파일을 읽어 점수를 부여합니다. """
    if not os.path.isdir(directory):
        print(f"Error: The directory '{directory}' does not exist.")
        sys.exit(1)
    
    scores = {"efConstruction": [], "maxConnections": [], "score": []}
    
    csv_file = os.path.join(directory, "score.csv")
    with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["filename", "efConstruction", "maxConnections", "score"])  # 헤더 작성

        for filename in os.listdir(directory):
            if filename.endswith(".json"):
                filepath = os.path.join(directory, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as file:
                        data = json.load(file)
                        score = score_json_file(data)
                        csvwriter.writerow([filename, data[0]["efConstruction"], data[0]["maxConnections"], score])

                        # 점수 정보를 저장
                        scores["efConstruction"].append(data[0]["efConstruction"])
                        scores["maxConnections"].append(data[0]["maxConnections"])
                        scores["score"].append(score)

                except Exception as e:
                    print(f"Failed to read {filename}: {e}")
                    
    return scores

def plot_scores(scores):
    """ 저장된 점수를 이용해 scatter와 surface 두 개의 3D 그래프를 그립니다. """
    fig = plt.figure(figsize=(24, 12))
    
    # Scatter plot
    ax1 = fig.add_subplot(121, projection='3d')
    x = np.array(scores["efConstruction"])
    y = np.array(scores["maxConnections"])
    z = np.array(scores["score"])
    ax1.scatter(x, y, z, c='r', marker='o')
    ax1.set_xlabel('efConstruction')
    ax1.set_ylabel('maxConnections')
    ax1.set_zlabel('Score')
    ax1.set_title('Scatter Plot')

    # Surface plot
    ax2 = fig.add_subplot(122, projection='3d')
    X, Y = np.meshgrid(np.unique(x), np.unique(y))
    Z = np.zeros_like(X, dtype=float)

    for i in range(len(x)):
        xi = np.where(np.unique(x) == x[i])[0][0]
        yi = np.where(np.unique(y) == y[i])[0][0]
        Z[yi, xi] = z[i]

    ax2.plot_surface(X, Y, Z, cmap='viridis')
    ax2.set_xlabel('efConstruction')
    ax2.set_ylabel('maxConnections')
    ax2.set_zlabel('Score')
    ax2.set_title('Surface Plot')

    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <directory>")
        sys.exit(1)

    directory = sys.argv[1]
    scores = read_and_score_json_files(directory)

    # 전체 파일 점수 출력
    print("\nOverall Scores:")
    for filename, score in zip(scores["efConstruction"], scores["score"]):
        print(f"{filename}: {score}")

    # 3D 그래프 그리기
    plot_scores(scores)
