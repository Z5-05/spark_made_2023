import os
from pathlib import Path
from sklearn.datasets import make_blobs, make_classification
import pandas as pd


Path(os.getcwd() + "/src/test/data").mkdir(parents=True, exist_ok=True)

# dataset №1
X, y = make_classification(
    n_samples=500, 
    n_features=4,
    n_classes=2)
dataset = pd.DataFrame(X)
dataset.columns = ['X1', 'X2', 'X3', 'X4']
dataset['label'] = y
dataset.to_csv("/home/ilya/MADE/spark_made_2023/src/test/data/dataset_noisy.csv", index=False)

# dataset №2
X, y = make_blobs(
    n_samples=500, 
    centers=2, 
    n_features=4)
dataset = pd.DataFrame(X)
dataset.columns =  ['X1', 'X2', 'X3', 'X4']
dataset['label'] = y
dataset.to_csv("/home/ilya/MADE/spark_made_2023/src/test/data/dataset_clear.csv", index=False)

