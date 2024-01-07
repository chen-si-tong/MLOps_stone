import sys
import os
import pickle
project_root = "/Users/chensitong/MLOps/project/MLOps_test"  
sys.path.append(project_root)

from tests import _PATH_DATA 
path_data= _PATH_DATA 

def test_data():
    with open(os.path.join(path_data, 'processed/train_dataset.pkl'), 'rb') as f:
        dataset = pickle.load(f)
    assert dataset[0][0].shape == (1, 28, 28)
    print('test_data passed!')

if __name__ == '__main__':
    test_data()