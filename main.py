import numpy as np
import pandas as pd
import data
import clustring
import sys


def main(argv):
    london_data = data.load_data("london_sample_500.csv")
    features = ["hum", "cnt"]
    london_data_np = clustring.transform_data(london_data, features)
    print(london_data_np)
    labels, centroids = clustring.kmeans(london_data_np, 2)
    print(labels)
    print(centroids)


if __name__ == '__main__':
    main(sys.argv)