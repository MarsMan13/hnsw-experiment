#!/usr/bin/env python3
import argparse
import subprocess
import random

class GetGroundTruth:
    # hardcorded working directory
    BENCHMARK_DIR = "../weaviate-benchmarking/benchmarker"

    def __init__(self, vectors, distance, maxConnectionsRange, efConstructionRange):
        for maxConnections in maxConnectionsRange:
            for efConstruction in efConstructionRange:
                self.get_ground_truth(vectors, distance, maxConnections, efConstruction)
        

    def get_ground_truth(self, vectors, distance, maxConnections, efConstruction):
        command = ["go", "run", ".", "ann-benchmark", "-v", vectors, "-d", distance, "--maxConnections", str(maxConnections), "--efConstruction", str(efConstruction), \
                    "--output", f"{efConstruction}_{maxConnections}_{''.join(random.choices('abcdef', k=4))}.json"]
        print(f"Running command in directory {self.BENCHMARK_DIR}: {command}")
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True, cwd=self.BENCHMARK_DIR)
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(e.stderr)
            exit(1)


if __name__ == '__main__':
    # get variables from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vectors", dest="vectors", action="store", default=None, help="Vectors to get ground truth for")
    parser.add_argument("-d", "--distance", dest="distance", action="store", default=None, help="Distance to get ground truth for")
    parser.add_argument("-M", "--maxConnectionsRange", dest="maxConnectionsRange", action="store", default=None, help="maxConnectionsRange to get ground truth for")
    parser.add_argument("-efc", "--efConstructionRange", dest="efConstructionRange", action="store", default=None, help="efConstructionRange to get ground truth for")

    args = parser.parse_args()
    if(args.vectors == None or args.distance == None or args.maxConnectionsRange == None or args.efConstructionRange == None):
        print("Please provide all the required arguments")
        exit(0)

    ## get the variables
    vectors = args.vectors
    distance = args.distance
    # maxConnectionsRange ====
    slice_values = args.maxConnectionsRange.strip('[]').split(':')
    start, stop, step = map(int, slice_values)
    maxConnectionsRange = list(range(start, stop, step))
    # efConstructionRange ====
    slice_values = args.efConstructionRange.strip('[]').split(':')
    start, stop, step = map(int, slice_values)
    efConstructionRange = list(range(start, stop, step))

    GetGroundTruth(vectors, distance, maxConnectionsRange, efConstructionRange)
