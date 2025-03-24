'''
    generate gpt responses and store in numpy arrays
'''

import os
import numpy as np
import pandas as pd
import openai
import re
from collections import Counter
import string
import time
from utils.others import *
from utils.metrics import top_k_accuracy
from utils.prompts import generate_responses_prompt0, generate_responses_prompt1

# gpt
openai.api_key = ''
openai.api_base = '' 
openai.api_type = ''
openai.api_version = '' # this may change in the future
deployment_name=''
gpt_setting = {'api_key':openai.api_key, 'api_base':openai.api_base, 'api_type':openai.api_type, 'api_version':openai.api_version, 'deployment_name':deployment_name}


# load the datase
candidates_dir = ''
candidates = load_candidates(file_dir=candidates_dir, candidate_num=10)
candidates = preprocess_np(candidates) # lowercase the array and remove punctuation

dataset_dir = ''
dataset = np.load(dataset_dir, allow_pickle=True)
prediction_names, concept_names, = dataset['prediction_names'], dataset['concept_names']
mentions = prediction_names
print(f'length of mentions {len(mentions)}')

# generate responses
repeat_times = 5
start = 0
end = 1
prompt = 'prompt1'
print(f'Using prompt {prompt}')
if prompt == 'prompt1':
    responses, input_list = generate_responses_prompt1(mentions, candidates, query_idx=range(start, end), candidate_number=candidates.shape[1], repeat_times=repeat_times, gpt_setting=gpt_setting)
    # print(responses)
    # print(input_list)
elif prompt == 'prompt0':
    responses, input_list = generate_responses_prompt0(mentions, candidates, query_idx=range(start, end), candidate_number=candidates.shape[1], repeat_times=repeat_times, gpt_setting=gpt_setting)

print(responses)

# save the responses
response_save_dir = '' + str(start) + '_' + str(end) + '.npy'
np.save(response_save_dir, np.array(responses, dtype=object), allow_pickle=True)
print(f'Responses saved in: {response_save_dir}')
print('End')
