import numpy as np
import pandas as pd
import os
from utils import *
import time
import igraph as ig

start_time = time.time()

mode = 'neigbhor'

if __name__ == '__main__':
    # prepare for the data
    linking = pd.read_csv('')
    feature_linking = pd.read_csv('')
    
    dataset_dir = ''
    dataset = np.load(dataset_dir, allow_pickle=True)
    ori_names = dataset['prediction_names']
    kg_names = dataset['concept_names']
    relation_path = ''
    dfr = pd.read_csv(relation_path)
    dfr = dfr[~dfr['column2'].str.contains('Not', na=False)]
    print('End loading data')
    print_running_time(start_time=start_time)

    label = ori_names
    label_linked_names, feature_linked_names = [], [], 
    label_pg_values, feature_pg_values = [], []
    all_relations = []
    
    
    for i in range(0, 1):
        print(f'Start fixing on index: {i}')
        label_entity = label[i]
        label_linked_name = linking[linking['entity']==label_entity]['ibkh_name'].iloc[0]
        print(label_linked_name)
        nodes_1, relations_1, all_1= get_neighbors_relation(dfr, label_linked_name)
        ## need to be modified below
        
    np.save("neighbor_21_neg.npy", relations_1, allow_pickle=True)
    
    
    