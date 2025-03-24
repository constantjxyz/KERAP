import numpy as np
import pandas as pd
import os
import string
import ast
import time
import igraph as ig
import re

    
def preprocess_np(np_array):
    # input a numpy array, lowercase the array and remove punctuation
    np_array = np.char.mod('%s', np_array)
    arr_lower = np.char.lower(np_array)
    arr_lower = np.vectorize(lambda text: re.sub(r'\s*\(.*?\)', '', text).strip())(arr_lower)
    translator = str.maketrans("", "", string.punctuation)
    arr_no_punctuation = np.char.translate(arr_lower, translator)
    arr_no_punctuation = np.char.replace(arr_no_punctuation, '\n', '')
    arr_no_punctuation = np.char.replace(arr_no_punctuation, '\t', '')
    return arr_no_punctuation

def print_running_time(start_time=0):
    current = time.time()
    print(f'Running time: {current-start_time} seconds')

def get_neighbors_pd(df, entity_name):
    EA = df[df['column1'] == entity_name]
    EB = df[df['column3'] == entity_name] 
    filtered_df_3 = EA['column3']
    filtered_df_1 = EB['column1']
    cat = np.concatenate((filtered_df_3, filtered_df_1))
    return np.array(cat, dtype=str), pd.concat([EA, EB], axis=0)

def get_neighbors_relation(df, entity_name):
    EA = df[df['column1'] == entity_name]
    EB = df[df['column3'] == entity_name] 
    EA['merged'] = EA[['column1', 'column2', 'column3']].apply(lambda row: '(' + ','.join(row.values.astype(str))+')', axis=1)
    EB['merged'] = EB[['column1', 'column2', 'column3']].apply(lambda row: '(' + ','.join(row.values.astype(str))+')', axis=1)
    relation_cat = np.concatenate((EA['merged'], EB['merged']))
    filtered_df_3 = EA['column3']
    filtered_df_1 = EB['column1']
    node_cat = np.concatenate((filtered_df_3, filtered_df_1))
    return np.array(node_cat, dtype=str), np.array(relation_cat, dtype=str), pd.concat([EA, EB], axis=0)

def construct_2hopgraph(df, entity_name):
    # get 1-hop neighbors
    # print(entity_name)
    nodes_1, edges_1= get_neighbors_pd(df, entity_name)
    # print(nodes_1)
    # print(edges_1)
    
    # get 2-hop neighbors
    EA = df[df['column1'].isin(nodes_1)]
    EB = df[df['column3'].isin(nodes_1)]
    edges_2 = pd.concat([EA, EB], axis=0)
    filtered_df_1 = EB['column1']
    filtered_df_3 = EA['column3']
    nodes_2 = np.array(np.concatenate((filtered_df_3, filtered_df_1)), dtype=str)
    # print(nodes_2)
    # print(edges_2)
    
    # concatenate 1-2-hop neighbors
    points = np.concatenate((nodes_1, nodes_2))
    edges = pd.concat([edges_1, edges_2], axis=0)
    
    
    # construct graph
    g = ig.Graph(directed=False)
    g.add_vertices(list(set(points)))
    edges = edges.astype('str')
    adding_edges = list(zip(edges['column1'], edges['column3']))
    g.add_edges(adding_edges)
    g.es['relation'] = edges['column2'].tolist()
    
    return g

def construct_1hopgraph(df, entity_name):
   # get 1-hop neighbors
    # print(entity_name)
    nodes_1, edges_1= get_neighbors_pd(df, entity_name)
    
    nodes_1 = np.concatenate((nodes_1, [entity_name]))
    # print(nodes_1)
    # print(edges_1)
    
    
    # concatenate 1-2-hop neighbors
    # points = np.concatenate((nodes_1, nodes_2))
    # edges = pd.concat([edges_1, edges_2], axis=0)
    
    
    # construct graph
    g = ig.Graph(directed=False)
    g.add_vertices(list(set(nodes_1)))
    edges_1 = edges_1.astype('str')
    adding_edges = list(zip(edges_1['column1'], edges_1['column3']))
    g.add_edges(adding_edges)
    g.es['relation'] = edges_1['column2'].tolist()
    
    return g

    