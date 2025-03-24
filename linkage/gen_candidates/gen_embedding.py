import os
import numpy as np
import pandas as pd
import torch
# from transformers import *
from transformers import AutoTokenizer
from transformers import pipeline
import time
from transformers import AutoModel
from utils.others import *
print('Start generating embedding')
start_time = time.time()

# '''  ---------------  modify the configuration here  ------------------------'''
tokenizer = AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")  
model = AutoModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
npz_save_dir = ''
# mode = 'try'
mode = 'generate'


# '''  ---------------  load dataset here  ------------------------'''
prediction = np.array(pd.read_csv('/local/scratch/yxie289/kg_mining/mimic_promote_label/data/labels.csv').columns)
prediction = np.append(prediction, "post-stroke cognitive impairment")
prediction = preprocess_np(prediction)
print(f'Shape of the unlinked predition entities: {prediction.shape}')

if mode == 'try':
        experiment_idxs = range(2)
else:
    experiment_idxs = range(len(prediction))

# '''  ---------------  modify the configuration here  ------------------------'''

pipeline = pipeline('feature-extraction', model=model, tokenizer=tokenizer, device=device)
feature_extraction = pipeline
embeddings = feature_extraction(['Hello, world!', 'This is a test sentence.'])
sentence_embeddings = [np.mean(embedding, axis=0) for embedding in embeddings]
print(sentence_embeddings[1].shape)
print(f'Running time {time.time()-start_time} seconds')

if mode == 'try':
    prediction_mid_embeddings = feature_extraction(list(prediction[:2]))
    
elif mode == 'generate':
    prediction_mid_embeddings = feature_extraction(list(prediction))
prediction_embeddings = [np.mean(embedding, axis=0).mean(axis=0) for embedding in prediction_mid_embeddings]
print(f'Running time {time.time()-start_time} seconds')

# load dataset for ibkh
keep_index = np.load('')
refer_dataset = np.load('', allow_pickle=True)
print(refer_dataset.files)
print(f'Shape of concept embeddings: {refer_dataset["concept_embeddings"][keep_index].shape}, concept names:{refer_dataset["concept_names"][keep_index].shape}')


np.savez(npz_save_dir, 
         prediction_embeddings=np.array(prediction_embeddings), prediction_names =prediction, concept_embeddings=refer_dataset['concept_embeddings'][keep_index], concept_names=refer_dataset['concept_names'][keep_index])

print(f'Running time {time.time()-start_time} seconds')
print('End generating embedding')


