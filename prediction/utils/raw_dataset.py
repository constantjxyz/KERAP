import numpy as np
import pandas as pd
import numpy
import os
from scipy.sparse import lil_matrix
from scipy.sparse import save_npz


def read_raw_hyperedges(file_dir, keep_sample_indices='all'):
    # get the hyperedges in files like 'hyperedges-mimic3.txt'
    with open(file_dir, 'r') as f:
        read_lines = f.readlines()
        if keep_sample_indices != 'all' and type(keep_sample_indices) == list:
            valid_indices = [i for i in keep_sample_indices if 0 <= i <= (len(read_lines) - 1)]
            read_lines = [read_lines[i] for i in valid_indices]  # cut the specific length of samples
        hyperedges = [list(map(int, text.strip().split(','))) for text in read_lines]
    return hyperedges



def read_raw_edge_labels(file_dir, keep_sample_indices='all', keep_label_indices='all'):
    # get the edge labels in files like 'edge-labels-mimic3.txt'
    with open(file_dir, 'r') as f:
        read_lines = f.readlines()
        if keep_sample_indices != 'all' and type(keep_sample_indices) == list:
            valid_indices = [i for i in keep_sample_indices if 0 <= i <= (len(read_lines) - 1)]
            read_lines = [read_lines[i] for i in valid_indices]  # cut the specific length of samples
        labels = np.array([list(map(int, text.strip().split(','))) for text in read_lines])
        if keep_label_indices != 'all' and type(keep_label_indices) == list:
            valid_indices = [i for i in keep_label_indices if 0 <= i <= (len(labels[0]) - 1)]
            labels = labels[:, valid_indices]
    return labels


def save_hyperedges(file_dir, list_of_list):
    # input: list of list
    with open(file_dir, "w") as file:
        for i in range(len(list_of_list)):
            hyperedge = list_of_list[i]
            file.write(",".join(map(str, hyperedge)) + '\n')

    
class EHRDataset:
    def __init__(self, **kwargs):
        # input: there need to be three files: feature_names.csv, features.txt, labels.csv
        # data stored in dataset: mostly numpy array; features as 
        self.dataset_dir = kwargs.get('dataset_dir', '/dataset/small/mimic3')
        required_files = ['feature_names.csv', 'features.txt', 'labels.csv']
        for file in required_files:
            assert file in os.listdir(self.dataset_dir), f"Required file '{file}' is missing in {self.dataset_dir}"
        self.features_exist = read_raw_hyperedges(os.path.join(self.dataset_dir, 'features.txt'))
        self.feature_names = np.array(pd.read_csv(os.path.join(self.dataset_dir, 'feature_names.csv'))['feature_name'])
        labels = pd.read_csv(os.path.join(self.dataset_dir, 'labels.csv'))
        self.label_names = np.array(labels.columns)
        self.labels = np.array(labels.values)
        self.sample_num, self.label_num, self.feature_num = len(self.features_exist), len(self.label_names), len(self.feature_names)  
             
        
        # get the split indices
        self.rand_seed = kwargs.get('rand_seed', 0)
        self.train_num = kwargs.get('train_num', 0)
        self.valid_num = kwargs.get('valid_num', 0)
        self.test_num = self.sample_num - self.train_num - self.valid_num
        assert self.test_num >= 0, f"Test set number '{self.test_num}' smaller than 0"  
        indices = np.arange(self.sample_num)
        np.random.seed(self.rand_seed)
        np.random.shuffle(indices)
        self.train_indices = indices[:self.train_num]
        self.valid_indices = indices[self.train_num:self.train_num + self.valid_num]
        self.test_indices = indices[self.train_num + self.valid_num:]        
        
    def print_info(self):
        print(f'dataset directory: {self.dataset_dir}')
        print(f'sample number: {self.sample_num}')
        print(f'feature number: {self.feature_num}')
        print(f'labels: {self.label_names}; total number: {self.label_num}')
        print(f'training set number: {self.train_num}, validation set number: {self.valid_num}, test_set_number: {self.test_num}')

    def get_feature_names(self):
        return self.feature_names
    def get_label_names(self):
        return self.label_names
    def get_train_valid_test_num(self):
        return self.train_num, self.valid_num, self.test_num
    def get_sample_data(self, idx):   # return sample feature (in string format) sample feature (int order format) and labels
        return [self.feature_names[f] for f in self.features_exist[idx]], self.features_exist[idx], list(self.labels[idx])
    def get_train_indices(self):
        return self.train_indices
    def get_val_indices(self):
        return self.val_indices
    def get_test_indices(self):
        return self.test_indices
  
    
    