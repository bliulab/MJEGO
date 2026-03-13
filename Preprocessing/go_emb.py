import numpy as np
from collections import deque
import csv
import Utils.Constants as Constants
from Utils.utils import pickle_save, pickle_load, save_ckp, load_ckp, class_distribution_counter, \
    draw_architecture, compute_roc
from collections import deque, Counter
import warnings
import pandas as pd
import numpy as np
from xml.etree import ElementTree as ET
import math
import torch
import torch.nn as nn
import torch.optim as optim
import Utils.net_utils as net_utils

class Ontology(object):

    def __init__(self, filename='data/go.obo', with_rels=True):
        self.ont = self.load(filename, with_rels)
        self.ic = None 

    def has_term(self, term_id):
        return term_id in self.ont

    def get_term(self, term_id):
        if self.has_term(term_id):
            return self.ont[term_id]
        return None

    def calculate_ic(self, annots):
        cnt = Counter()
        for x in annots:
            cnt.update(x)
        self.ic = {}
        for go_id, n in cnt.items():
            parents = self.get_parents(go_id)
            if len(parents) == 0:
                min_n = n
            else:
                min_n = min([cnt[x] for x in parents])
            self.ic[go_id] = math.log(min_n / n, 2)
    
    def get_ic(self, go_id):
        if self.ic is None:
            raise Exception('Not yet calculated')
        if go_id not in self.ic:
            return 0.0
        return self.ic[go_id]

    def load(self, filename, with_rels):
        ont = dict()
        obj = None
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line == '[Term]':
                    if obj is not None:
                        ont[obj['id']] = obj
                    obj = dict()
                    obj['is_a'] = list()
                    obj['part_of'] = list()
                    obj['regulates'] = list()
                    obj['alt_ids'] = list()
                    obj['is_obsolete'] = False
                    continue
                elif line == '[Typedef]':
                    if obj is not None:
                        ont[obj['id']] = obj
                    obj = None
                else:
                    if obj is None:
                        continue
                    l = line.split(": ")
                    if l[0] == 'id':
                        obj['id'] = l[1]
                    elif l[0] == 'alt_id':
                        obj['alt_ids'].append(l[1])
                    elif l[0] == 'namespace':
                        obj['namespace'] = l[1]
                    elif l[0] == 'is_a':
                        obj['is_a'].append(l[1].split(' ! ')[0])
                    elif with_rels and l[0] == 'relationship':
                        it = l[1].split()
                        # add all types of relationships
                        obj['is_a'].append(it[1])
                    elif l[0] == 'name':
                        obj['name'] = l[1]
                    elif l[0] == 'is_obsolete' and l[1] == 'true':
                        obj['is_obsolete'] = True
            if obj is not None:
                ont[obj['id']] = obj
        for term_id in list(ont.keys()):
            for t_id in ont[term_id]['alt_ids']:
                ont[t_id] = ont[term_id]
            if ont[term_id]['is_obsolete']:
                del ont[term_id]
        for term_id, val in ont.items():
            if 'children' not in val:
                val['children'] = set()
            for p_id in val['is_a']:
                if p_id in ont:
                    if 'children' not in ont[p_id]:
                        ont[p_id]['children'] = set()
                    ont[p_id]['children'].add(term_id)
        return ont


    def get_anchestors(self, term_id):
        if term_id not in self.ont:
            return set()
        term_set = set()
        q = deque()
        q.append(term_id)
        while(len(q) > 0):
            t_id = q.popleft()
            if t_id not in term_set:
                term_set.add(t_id)
                for parent_id in self.ont[t_id]['is_a']:
                    if parent_id in self.ont:
                        q.append(parent_id)
        return term_set


    def get_parents(self, term_id):
        if term_id not in self.ont:
            return set()
        term_set = set()
        for parent_id in self.ont[term_id]['is_a']:
            if parent_id in self.ont:
                term_set.add(parent_id)
        return term_set


    def get_namespace_terms(self, namespace):
        terms = set()
        for go_id, obj in self.ont.items():
            if obj['namespace'] == namespace:
                terms.add(go_id)
        return terms

    def get_namespace(self, term_id):
        return self.ont[term_id]['namespace']
    
    def get_term_set(self, term_id):
        if term_id not in self.ont:
            return set()
        term_set = set()
        q = deque()
        q.append(term_id)
        while len(q) > 0:
            t_id = q.popleft()
            if t_id not in term_set:
                term_set.add(t_id)
                for ch_id in self.ont[t_id]['children']:
                    q.append(ch_id)
        return term_set
    

def create_go_matrix(go_ids, Ontology):
    term_to_index = {term_id: idx for idx, term_id in enumerate(go_ids)}
    num_terms = len(go_ids)
    go_matrix = np.zeros((num_terms, num_terms), dtype=int)

    for term_id in go_ids:
        idx = term_to_index[term_id]
        ancestors = Ontology.get_anchestors(term_id)
        for ancestor in ancestors:
            if ancestor in term_to_index:
                go_matrix[idx, term_to_index[ancestor]] = 1

    return go_matrix

def main(go_filename, go_ids):
    ontology = Ontology(go_filename)
    go_matrix = create_go_matrix(go_ids, ontology)
    
    # transfer to runable
    go_matrix_pt = torch.tensor(go_matrix, dtype=torch.float32, requires_grad=False)
    
    # print("go_matrix_pt:", go_matrix_pt)
    # print("Shape:", go_matrix_pt.shape)
    # print("Requires Grad:", go_matrix_pt.requires_grad)
    
    return go_matrix_pt

# 

class goEmbed(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        
        self.fc1 = net_utils.FC(input_size, output_size, relu=False, bnorm=False)
        # self.fc2 = net_utils.FC(hidden_size, output_size, relu=False, bnorm=True)
        
    def forward(self, x):
        
        x = self.fc1(x)
        # x = self.fc2(x)
                 
        return x

def load_embeddings(go_ids, embedding_file):
    """
    从文件加载GO嵌入,并返回一个可训练的嵌入矩阵
    """
    # 加载.npy文件并获取GO字典
    data = np.load(embedding_file, allow_pickle=True)
    go_dict = data.item()  # 获取字典
        
    # 如果GO ID不在文件中，则用零填充
    embeddings = []
    for go_id in go_ids:
        if go_id in go_dict:
            embeddings.append(go_dict[go_id])
        else:
            print(f"Warning: GO ID {go_id} not found in embeddings file. Using zero vector.")
            embeddings.append(np.zeros_like(next(iter(go_dict.values()))))  # 使用零向量

    # 将嵌入矩阵转为 PyTorch 张量，并设置为可训练参数
    embedding_matrix = torch.FloatTensor(np.vstack(embeddings))
    # embedding_matrix_param = torch.nn.Parameter(embedding_matrix_tensor, requires_grad=True)
        
    return embedding_matrix

def go_emb(ont, output_dim):
    # init Pararm
    ont = ont
    go_ids = pickle_load(Constants.ROOT + 'go_terms')[f'GO-terms-{ont}'] 
    # embedding_file = '/data0/wzw/ProtPre/data/label-embedding-200.npy'
    embedding_file = '/data0/wzw/ProtPre/data/label-embedding-200.npy'
    
    # get matrix dim:(go_ids.shape[0], go_ids.shape[0])
    embedding_matrix = load_embeddings(go_ids, embedding_file)
    
    # print(f"embedding_matrix requires_grad: {embedding_matrix.requires_grad}")
    # print(embedding_matrix.shape)

    model = goEmbed(embedding_matrix.shape[1], output_dim)

    output = model(embedding_matrix)

    # print(output)
    # print(output.shape)
    # print(f"Output requires_grad: {output.requires_grad}")
    
    return output
    # return embedding_matrix