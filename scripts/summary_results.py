#! /usr/bin/env python3

import os
import sys
import json
import csv

def read_json_files(directory):
    if not os.path.isdir(directory):
        print(f"Error: The directory '{directory}' does not exist.")
        sys.exit(1)

    csv_file = os.path.join(directory, "summary.csv")

    with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["run_id", "efConstruction", "maxConnections", "ef", "recall", "qps", "meanLatency", "p99Latency", "importTime"])  # 헤더 작성

        for filename in os.listdir(directory):
            if filename.endswith(".json"):
                filepath = os.path.join(directory, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as file:
                        data = json.load(file)
                        if isinstance(data, list):
                            for item in data:
                                csvwriter.writerow([item["run_id"], item["efConstruction"], item["maxConnections"], item["ef"], item["recall"], item["qps"], item["meanLatency"], item["p99Latency"], item["importTime"]])

                except Exception as e:
                    print(f"Failed to read {filename}: {e}")

    print(f"Data successfully written to {csv_file}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <directory>")
        sys.exit(1)

    directory = sys.argv[1]
    read_json_files(directory)

