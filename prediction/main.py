import os
import sys
project_path = ''
os.chdir(project_path)
sys.path.append(project_path)
from utils.raw_dataset import *
from utils.setting import *
from utils.utils import *
from method.azure.azure import *
from utils.evaluation import *

# import debugpy
# debugpy.listen(('0.0.0.0', 58370))


def main():
    # get settings from the command line
    param_setting = parse_args_to_dict()
    print('-'*10 + 'All arguments: '+ '-'*10 )
    print_dict(param_setting)
    
    # get the input dataset
    print('-'*10 + 'Input dataset: '+ '-'*10 )
    dataset = EHRDataset(**param_setting)
    dataset.print_info()
    
    # initiate different pipelines
    package = param_setting.get('package', 'azure')
    assert package in ['sklearn', 'azure'], f"Invalid package specificed: {package}"
    print('-'*10 + 'Pipeline: '+ '-'*10 )
    if package == 'azure':
        model = param_setting.get('model', 'gpt-4o-mini')
        gpt_setting = gpt_setting_dict(model=model)
        print_dict(gpt_setting)       
        responses, process_predictions, predictions, labels, prompt_token_num, completion_token_num, time_cost = gpt_chat(dataset, **{**gpt_setting, **param_setting})
        # np.save(os.path.join(param_setting.get('output_dir', './dataset/small/promote/response'), 'predictions.npy'), predictions)
        print(f"time cost: {time_cost}s, prompt token number: {prompt_token_num}, completion token number: {completion_token_num}")
        pass
    else:
        pass

    # evaluation
    print('-'*10 + 'Evaluation: '+ '-'*10 )
    evaluation(predictions, labels)
    
    
    # end
    print('-'*10 + 'End'+ '-'*10 )
    print_running_time(param_setting['start_time'])
    

if __name__ == "__main__":
    main()