from utils.others import *
from utils.distances import *
from utils.metrics import *
import re
import os
import numpy as np
import torch
import time


# -----------------   modify here ---------------------------
print('Start matching embeddings')
os.chdir('')
file_dir = ''
save_prediction = ''

# -----------------   load dataset ---------------------------
dataset = np.load(file_dir, allow_pickle=True)
prediction_embeddings, concept_embeddings = dataset['prediction_embeddings'], dataset['concept_embeddings'], 
prediction_names, concept_names = dataset['prediction_names'], dataset['concept_names'], 
print(f'File_dir: {file_dir}')
array_names = dataset.files
for name in array_names:
    print(f"{name}: {dataset[name].shape}")

    
# device = torch.device('cuda')
device = torch.device('cpu')
slice_num = 1

    

# -----------------   no need to modify ---------------------------
start_time = time.time()
# print(f'Experiment ids: start{experiment_ids[0]}, end:{experiment_ids[-1]}')
# mention_predictions = generate_predictions_cosine_torch_slice(mention_embeddings, concept_embeddings, concepts, query_idxs=experiment_ids, k=100, slice_num=slice_num, device=device)
print(f'Matching embeddings for prediction')
prediction_candidates, prediction_values = generate_predictions_cosine_torch_cpu(prediction_embeddings, concept_embeddings, concept_names, query_idxs=range(len(prediction_names)), k=10,)
print_running_time(start_time=start_time)

    
    
if save_prediction != '':
    prediction_npy(prediction_candidates, output_dir=save_prediction)
    
end_time = time.time()
print_running_time(start_time=start_time)
print('End matching embeddings')