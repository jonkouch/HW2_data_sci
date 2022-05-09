import numpy as np
import pandas as pd
import data
import clustring

london_data = np.array(data.load_data("london_sample_500.csv"))

labels, centroids = clustring.kmeans(london_data, 2)
print(centroids)