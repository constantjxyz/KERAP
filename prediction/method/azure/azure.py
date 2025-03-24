from openai import AzureOpenAI
from utils.utils import *
from utils.raw_dataset import *
import numpy as np
import time
from method.azure.chatbot_handlers.chatbot_handler_base import BaseChatBotHandler
import importlib.util
import sys

def gpt_setting_dict(model):
    api_key = ''
    api_endpoint = ''
    api_type = 'azure'
    api_version = '2024-06-01' 
    client = AzureOpenAI(azure_endpoint=api_endpoint, api_key=api_key, api_version=api_version)
    gpt_setting = {'api_key':api_key, 'api_endpoint':api_endpoint, 'api_type':api_type, 'api_version':api_version, 'model':model, 'client':client}
    return gpt_setting

def gpt_chat(ehrdataset, **kwargs):
    train_num, valid_num, test_num = ehrdataset.get_train_valid_test_num()
    feature_names, label_names = ehrdataset.get_feature_names(), ehrdataset.get_label_names()
       
    # test_dataset
    all_responses = []
    all_predictions = []
    all_process_predictions = []
    all_prompt_tokens = []
    all_completion_tokens = []
    all_labels = []
    sample_indices = ehrdataset.get_test_indices()
    if kwargs['gen_mode']:
        run_sample_num = test_num
        run_sample_indices = sample_indices
    else:
        run_sample_num = 3  # only consider three samples
        run_sample_indices = sample_indices[:3]


    # load chatbot handler
    handler_file = kwargs.get('handler_file')
    if handler_file:
        if not os.path.isfile(handler_file):
            raise FileNotFoundError(f"Handler file '{handler_file}' does not exist.")

        module_name = os.path.splitext(os.path.basename(handler_file))[0]  # 提取模块名称
        spec = importlib.util.spec_from_file_location(module_name, handler_file)
        if spec is None:
            raise ImportError(f"Could not load module from '{handler_file}'")
        
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        handler_class_name = "ChatBotHandler"  # 你的 handler 里应该有这个类
        if hasattr(module, handler_class_name):
            handler_class = getattr(module, handler_class_name)
            handler = handler_class(**kwargs)  # 实例化
        else:
            raise ImportError(f"Module '{handler_file}' does not have class '{handler_class_name}'")
    else:
        raise ValueError("handler_file must be specified in kwargs")
    
    # run chats
    for i in range(run_sample_num):
        sample_idx = run_sample_indices[i]  
        # print('sample_idx', sample_idx)
        if i < 100:
            process_responses, process_predictions, final_prediction, patient_labels, prompt_tokens, completion_tokens = handler.prompt_chat(ehrdataset, sample_idx, print_conversation=True)
        else:
            process_responses, process_predictions, final_prediction, patient_labels, prompt_tokens, completion_tokens = handler.prompt_chat(ehrdataset, sample_idx, print_conversation=False)
        all_labels.append(patient_labels)
        all_responses.append(process_responses)
        all_process_predictions.append(process_predictions)
        all_predictions.append(final_prediction)
        all_prompt_tokens.append(prompt_tokens)
        all_completion_tokens.append(completion_tokens)
        if i%50 == 0:
            print(f'Current idx: {i}; Last idx: {run_sample_num - 1}')
            print_running_time(kwargs['start_time'])
    print(f'All chats are finished now')
    all_running_time = print_running_time(kwargs['start_time'])
    return np.array(all_responses), np.array(all_process_predictions), np.array(all_predictions), np.array(all_labels), np.array(all_prompt_tokens).sum(), np.array(all_completion_tokens).sum(), all_running_time


