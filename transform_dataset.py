import pandas as pd
import sys
import os

def transform_dataset(dataset_path, rating_file, user_file, item_file):
    print(f'transforming {dataset_path}, {rating_file}, {user_file}, {item_file}')

    dataset_path = os.path.join(os.path.dirname(__file__), dataset_path)
    if not os.path.exists(dataset_path):
        print("dataset is not exist!!!!")
        return

if __name__ == '__main__':
    if len(sys.argv) < 5:
        print("Usage: python transform_dataset.py dataset_path rating_file user_file item_file")
        sys.exit(1)
    
    transform_dataset(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])