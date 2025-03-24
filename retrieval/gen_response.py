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

# gpt
openai.api_key = ''
openai.api_base = '' 
openai.api_type = ''
openai.api_version = '' # this may change in the future
deployment_name=''
gpt_setting = {'api_key':openai.api_key, 'api_base':openai.api_base, 'api_type':openai.api_type, 'api_version':openai.api_version, 'deployment_name':deployment_name}


# load the datase
dataset_dir = ''
dataset = np.load(dataset_dir, allow_pickle=True)

# generate responses
start = 0
end = 1
prompt = 'prompt1'
print(f'Using prompt {prompt}')
if prompt == 'prompt1':
    responses, input_list = generate_responses_prompt1(dataset, gpt_setting=gpt_setting)
    # print(responses)
    # print(input_list)
elif prompt == 'prompt0':
    responses, input_list = generate_responses_prompt0(dataset, gpt_setting=gpt_setting)

print(responses)

# save the responses
response_save_dir = '' + str(start) + '_' + str(end) + '.npy'
np.save(response_save_dir, np.array(responses, dtype=object), allow_pickle=True)
print(f'Responses saved in: {response_save_dir}')
print('End')
