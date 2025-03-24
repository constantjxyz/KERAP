from utils.others import *
from utils.metrics import *
from utils.prompts import *
from utils.retrieval import *
import re
import os
import numpy as np
import pandas as pd
import time


# -----------------   modify here ---------------------------
print('Start matching embeddings')
response_save_dir = ''
responses = np.load(response_save_dir, allow_pickle=True)
file_dir = ''
candidate_dir = ''
candidates = load_candidates(file_dir = candidate_dir, candidate_num=10)
save_candidates = ''
save_excel = ''

dataset_dir = file_dir
dataset = np.load(dataset_dir, allow_pickle=True)
prediction_names, concept_names, = dataset['prediction_names'], dataset['concept_names']




# mode = 'prompt0'
mode = 'prompt1'
# mode = 'prompt0_filter'
    

# -----------------   no need to modify ---------------------------
start_time = time.time()


if mode == 'prompt0':
    new_predictions = retrieve_prediction_prompt0(responses, response_idx=range(responses.shape[0]), answer_number=responses.shape[1])
    new_predictions = new_predictions.reshape(int(new_predictions.shape[0]/candidates.shape[1]), candidates.shape[1], responses.shape[1])
    retrieved_answer = pick_prompt0(new_predictions, candidates, prediction_idx=range(new_predictions.shape[0]), candidate_num=new_predictions.shape[1], answer_num=new_predictions.shape[2])
    # top_1_acc, right_idxs, wrong_idxs = top_k_accuracy(retrieved_answer, labels, query_idxs=range(len(retrieved_answer)), k=1, queries=mentions)

elif mode == 'prompt0_filter':
    new_predictions = retrieve_prediction_prompt0(responses, response_idx=range(responses.shape[0]), answer_number=responses.shape[1])
    new_predictions = new_predictions.reshape(int(new_predictions.shape[0]/candidates.shape[1]), candidates.shape[1], responses.shape[1])
    new_candidates = filter_candidates(new_predictions, candidates, prediction_idx=range(new_predictions.shape[0]), candidate_num=new_predictions.shape[1], answer_num=new_predictions.shape[2], low=0, high=new_predictions.shape[2]*0.8)


elif mode == 'prompt1':
    new_predictions = retrieve_prediction_prompt1(responses, response_idx=range(responses.shape[0]), answer_number=responses.shape[1])
    retrieved_answer = pick_prompt1(new_predictions, candidates, prediction_idx=range(new_predictions.shape[0]), candidate_num=candidates.shape[1], answer_num=new_predictions.shape[1])


if save_candidates != '':
    np.save(save_candidates, new_candidates, allow_pickle=True)
if save_excel != '':
    df = pd.DataFrame(columns=['entity', 'linking'])
    df['entity'] = prediction_names
    df['linking'] = retrieved_answer
    df.to_csv(save_excel, index=False)
    
end_time = time.time()
print_running_time(start_time=start_time)
print('End matching embeddings')