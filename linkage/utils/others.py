import numpy as np
import time
import os
import pandas as pd
import string


def load_candidates(file_dir='', candidate_num=5):
    candidates = np.load(file_dir, allow_pickle=True)
    candidates = candidates[:, :candidate_num]
    return candidates

def preprocess_np(np_array):
    # input a numpy array, lowercase the array and remove punctuation
    np_array = np.char.mod('%s', np_array)
    arr_lower = np.char.lower(np_array)
    translator = str.maketrans("", "", string.punctuation)
    arr_no_punctuation = np.char.translate(arr_lower, translator)
    return arr_no_punctuation
    
def print_running_time(start_time=0):
    # pass
    current = time.time()
    print(f'Running time: {current-start_time} seconds')
    
def prediction_excel(preds, labs, query_idxs=range(3), k=5, queries=[], output_dir=''):
    df = pd.DataFrame(columns=['mentions', 'labels']+['prediction_'+str(i) for i in range(k)])
    for query_idx in query_idxs:
        df.loc[len(df)] = [queries[query_idx], labs[query_idx]] + list(preds[query_idx][:k])
    df.to_excel(output_dir, index=False)
    print(f'save prediction excel to {output_dir}')

def prediction_npy(predictions, output_dir=''):
    np.save(output_dir, np.array(predictions, dtype=object))
    print(f'save prediction npy to {output_dir}')