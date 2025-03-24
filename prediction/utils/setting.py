import os
import argparse
import time


def parse_args_to_dict():
    # transform settings from command line into dictionary
    parser = argparse.ArgumentParser(description="Parse arguments into a dictionary.")
    parser.add_argument("--dataset_dir", type=str, default="./dataset/middle/promote", help="Folder directory of the utilized dataset.")
    parser.add_argument("--rand_seed", type=int, default=0, help="Seed of the random state.")
    parser.add_argument("--train_num", type=int, default=0, help="Number of samples in the training set.")
    parser.add_argument("--valid_num", type=int, default=0, help="Number of samples in the validation set.")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Use what model to finish the task")
    parser.add_argument("--prompt", type=str, default="prompt0", help="Which prompt to use")
    parser.add_argument("--gen_mode", action="store_true", help="Use the generating mode consider all the samples, otherwise only 3 samples are considered.")
    parser.add_argument("--handler_file", type=str, default="./method/azure/chatbot_handlers/promote_psci/kg_llm_cot.py", help="Which file contains the chatbot handler")
    args = parser.parse_args()
    args_dict = vars(args)
    
    
    # add more arguments
    ml_model_lists = ['logistic_regression']
    if args_dict['model'] in ml_model_lists:
        args_dict['package'] = 'sklearn'
    gpt_model_lists = ['gpt-4o-mini', 'gpt-4o', 'gpt-4-turbo', 'gpt-4-32k', 'gpt-35-turbo-0301']
    if args_dict['model'] in ml_model_lists:
        args_dict['package'] = 'azure'
    args_dict['start_time'] = time.time()

    # validate the arguments in the args.dict
    assert os.path.exists(args_dict['dataset_dir']), f"Dataset dir '{args_dict['dataset_dir']}' does not exist"
    assert os.path.exists(args_dict['output_dir']), f"Output dir '{args_dict['output_dir']}' does not exist"
    assert args_dict['train_num'] >= 0, f"Training set number '{args_dict['train_num']}' smaller than 0"
    assert args_dict['valid_num'] >= 0, f"Validation set number '{args_dict['valid_num']}' smaller than 0"  
    valid_model_lists = ml_model_lists + gpt_model_lists
    assert args_dict['model'] in valid_model_lists, f"Invalid model '{args_dict['model']}' are specificed"
    assert args_dict['prompt'] in ['prompt0', 'prompt1', 'prompt2'], f"Invalid prompt setting '{args_dict['prompt']}'"
    assert os.path.exists(args_dict['handler_file']), f"Chatbot hanlder file '{args_dict['handler_file']}' does not exist"
    
    return args_dict



    