import pandas as pd
import os

def check_class_distribution(directory):
    csv_path = os.path.join(directory, '_classes.csv')
    df = pd.read_csv(csv_path)
    print(f"Class distribution in {directory}:")
    print(df.drop('filename', axis=1).sum())

check_class_distribution('data/train')
