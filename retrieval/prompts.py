import numpy as np
import time
import openai

'''generate responses for different kinds of prompts 
    input: dataset, gpt_engine
    output: response
'''

def generate_responses_prompt0(dataset, gpt_setting=dict()):
    # set gpt
    openai.api_key = gpt_setting['api_key']
    openai.api_base = gpt_setting['api_base'] 
    openai.api_type = gpt_setting['api_type']
    openai.api_version = gpt_setting['api_version']
    deployment_name= gpt_setting['deployment_name']
    
    # set containers to store variables which need to be returned or printed finally
    prompt_input_list = []
    all_responses = []
    all_prompt_tokens = []
    all_completion_tokens = []
    all_gpt_times = []
    
    prompt_tokens = 0
    completion_tokens = 0
    gpt_times = 0
            
    # generate prompt
    prompt_input =   [{"role":"user", "content":f"'Please summarize key related factors for diagnosing PSCI in a paragraph based on your knowledge base and the extracted KG information: {dataset}"}]
            
    # generate response
    response = openai.ChatCompletion.create(messages=prompt_input, engine=deployment_name,)
            
    
    # time.sleep(5)  # avoid load limit
        
    # return the responses and prompts
    return np.char.mod('%s', np.array(all_responses)), 

def generate_responses_prompt1(dataset, gpt_setting=dict()):
    # set gpt
    openai.api_key = gpt_setting['api_key']
    openai.api_base = gpt_setting['api_base'] 
    openai.api_type = gpt_setting['api_type']
    openai.api_version = gpt_setting['api_version']
    deployment_name= gpt_setting['deployment_name']
    
    # set containers to store variables which need to be returned or printed finally
    prompt_input_list = []
    all_responses = []
    all_prompt_tokens = []
    all_completion_tokens = []
    all_gpt_times = []
    
    prompt_tokens = 0
    completion_tokens = 0
    gpt_times = 0
            
    # generate prompt
    prompt_input =   [{"role":"user", "content":f"'Please summarize unrelated factors for diagnosing PSCI based on your knowledge base and the extracted KG information: {dataset}"}]
            
    # generate response
    response = openai.ChatCompletion.create(messages=prompt_input, engine=deployment_name,)
            
    
    # time.sleep(5)  # avoid load limit
        
    # return the responses and prompts
    return np.char.mod('%s', np.array(all_responses)), 

