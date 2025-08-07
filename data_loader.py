import csv
import numpy as np
from sklearn.model_selection import train_test_split

def load_dataset(filepath):
    x, y = [], []
    diseases = {}
    disease_index = 0

    with open(filepath, 'r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)
        for row in csv_reader:
            disease = row[0]
            if disease not in diseases:
                diseases[disease] = disease_index
                disease_index += 1
            y.append(disease)
            x.append([int(val) for val in row[1:]])

    return np.array(x), y, diseases

def encode_labels(y, diseases):
    return np.array([diseases[label] for label in y])
