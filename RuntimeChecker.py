import numpy as np
import pandas as pd
import sys

test = pd.read_csv(sys.argv[1], header=None)
print("Durchschnitt: " + str(test.mean(axis=1)))
print("Median: " + str(test.median(axis=1)))
print("Max: " + str(test.max(axis=1)))
print("Min: " + str(test.min(axis=1)))
print("Verteilung:")
test = test.T
print(test.value_counts())
print(test.shape)