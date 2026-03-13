import math
import os, subprocess
import sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import shutil
import pandas as pd
import torch
import itertools
from Bio import SeqIO
import pickle
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from biopandas.pdb import PandasPdb
from collections import deque, Counter
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
import csv
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from torchviz import make_dot
from Utils.Constants import *
import re
import subprocess
from pathlib import Path
import numpy as np
import torch.nn.functional as F
from keras_preprocessing.sequence import pad_sequences
from joblib import Parallel, delayed
import torch, functools
from torch import nn
import torch.nn.functional as F
import torch, math
import torch.utils.data as data
import torch_geometric
import torch_cluster
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_add
import torch.utils.checkpoint as checkpoint
import Utils.Constants as Constants
from Utils.Constants import residues, amino_acids
from keras.utils import to_categorical
from typing import List, Tuple
from Bio import SeqIO
import string
import json
import tqdm, random
import Utils.net_utils as net_utils
import numpy as np
from timm import optim
from Utils.cosine_annealing_warmup import CosineAnnealingWarmupRestarts
import torch
from torch.utils.tensorboard import SummaryWriter
import logging as log
from box import Box
import shutil
from esm import pretrained
from torch.nn.functional import normalize
from pathlib import Path
import torch.nn.functional as F
import datetime
import random
from accelerate import Accelerator
from sklearn.metrics import precision_score, recall_score

###### Dataset Utils #####

def get_input_data():
    # test data
    input = ["1a0b", "1a0c", "1a0d", "1a0e", "1a0f", "1a0g", "1a0h", "1a0i", "1a0j", "1a0l"]
    raw = [s + ".pdb" for s in input]
    processed = [s + ".pt" for s in input]

    with open('../Dataset/raw.pickle', 'wb') as handle:
        pickle.dump(raw, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('../Dataset/proceesed.pickle', 'wb') as handle:
        pickle.dump(processed, handle, protocol=pickle.HIGHEST_PROTOCOL)


patterns = {
    'pdb': r'pdb[0-9]*$',
    'pdb.gz': r'pdb[0-9]*\.gz$',
    'mmcif': r'(mm)?cif$',
    'sdf': r'sdf[0-9]*$',
    'xyz': r'xyz[0-9]*$',
    'xyz-gdb': r'xyz[0-9]*$',
    'silent': r'out$',
    'sharded': r'@[0-9]+',
}

_regexes = {k: re.compile(v) for k, v in patterns.items()}


def is_type(f, filetype):
    if filetype in _regexes:
        return _regexes[filetype].search(str(f))
    else:
        return re.compile(filetype + r'$').search(str(f))


def find_files(path, suffix, relative=None, type="Path"):
    """
    Find all files in path with given suffix. =

    :param path: Directory in which to find files.
    :type path: Union[str, Path]
    :param suffix: Suffix determining file type to search for.
    :type suffix: str
    :param relative: Flag to indicate whether to return absolute or relative path.

    :return: list of paths to all files with suffix sorted by their names.
    :rtype: list[str]
    """
    if not relative:
        find_cmd = r"find {:} -regex '.*\.{:}' | sort".format(path, suffix)
    else:
        find_cmd = r"cd {:}; find . -regex '.*\.{:}' | cut -d '/' -f 2- | sort" \
            .format(path, suffix)
    out = subprocess.Popen(
        find_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd=os.getcwd(), shell=True)
    (stdout, stderr) = out.communicate()
    name_list = stdout.decode().split()
    name_list.sort()
    if type == "Path":
        return sorted([Path(x) for x in name_list])
    elif type == "Name":
        return sorted([Path(x).name for x in name_list])

"""
def process_pdbpandas(raw_path, chain_id):
    pdb_to_pandas = PandasPdb().read_pdb(raw_path)

    pdb_df = pdb_to_pandas.df['ATOM']
    assert (len(set(pdb_df['chain_id'])) == 1) & (list(set(pdb_df['chain_id']))[0] == chain_id)

    pdb_df = pdb_df[(pdb_df['atom_name'] == 'CA') & (pdb_df['chain_id'] == chain_id)]
    pdb_df = pdb_df.drop_duplicates()

    _residues = pdb_df['residue_name'].to_list()
    _residues = [amino_acids[i] for i in _residues if i != "UNK"]

    sequence_features = [[residues[residue] for residue in _residues]]

    sequence_features = pad_sequences(sequence_features, maxlen=1024, truncating='post', padding='post')

    # sequences + padding
    sequence_features = torch.tensor(to_categorical(sequence_features, num_classes=len(residues) + 1))
    # sequence_features = F.one_hot(sequence_features, num_classes=len(residues) + 1).to(dtype=torch.int64)

    node_coords = torch.tensor(pdb_df[['x_coord', 'y_coord', 'z_coord']].values, dtype=torch.float32)

    return node_coords, sequence_features, ''.join(_residues)

    return residues
"""

def process_pdbpandas(raw_path, chain_id):
    pdb_to_pandas = PandasPdb().read_pdb(raw_path)

    pdb_df = pdb_to_pandas.df['ATOM']
    assert (len(set(pdb_df['chain_id'])) == 1) & (list(set(pdb_df['chain_id']))[0] == chain_id)

    pdb_df = pdb_df[(pdb_df['atom_name'] == 'CA') & (pdb_df['chain_id'] == chain_id)]
    pdb_df = pdb_df.drop_duplicates()

    # **显式按 residue_number 排序，确保 node_coords 按残基顺序排列**
    pdb_df = pdb_df.sort_values(by=['residue_number']).reset_index(drop=True)

    _residues = pdb_df['residue_name'].to_list()
    _residues = [amino_acids[i] for i in _residues if i != "UNK"]

    sequence_features = [[residues[residue] for residue in _residues]]
    sequence_features = pad_sequences(sequence_features, maxlen=1024, truncating='post', padding='post')
    sequence_features = torch.tensor(to_categorical(sequence_features, num_classes=len(residues) + 1))

    node_coords = torch.tensor(pdb_df[['x_coord', 'y_coord', 'z_coord']].values, dtype=torch.float32)

    return node_coords, sequence_features, ''.join(_residues)


def generate_Identity_Matrix(shape, sequence):

    node_coords = torch.from_numpy(np.zeros(shape=(shape[0], 3)))
    _residues = sequence[3]

    # _residues = [amino_acids[i] for i in _residues if i != "UNK"]

    sequence_features = [[residues[residue] for residue in list(_residues) if residue not in Constants.INVALID_ACIDS]]
    sequence_features = pad_sequences(sequence_features, maxlen=1024, truncating='post', padding='post')
    # sequences + padding
    sequence_features = torch.tensor(to_categorical(sequence_features, num_classes=len(residues) + 1))
    # sequence_features = F.one_hot(sequence_features, num_classes=len(residues) + 1).to(dtype=torch.int64)
    return node_coords, sequence_features, str(_residues)


def get_cbrt(a):
    return a**(1./3.)


def get_knn(**kwargs):
    mode = kwargs["mode"]
    seq_length = kwargs["sequence_length"]
    if mode == "sqrt":
        x = int(math.sqrt(seq_length))
        if x % 2 == 0:
            return x + 1
        return x
    elif mode == "cbrt":
        x = int(get_cbrt(seq_length))
        if x % 2 == 0:
            return x + 1
        return x
    else:
        return seq_length
    

def process_pdbpandas2(raw_path, chain_id):
    pdb_to_pandas = PandasPdb().read_pdb(raw_path)

    pdb_df = pdb_to_pandas.df['ATOM']
    assert (len(set(pdb_df['chain_id'])) == 1) & (list(set(pdb_df['chain_id']))[0] == chain_id)

    pdb_df = pdb_df[(pdb_df['atom_name'] == 'CA') & (pdb_df['chain_id'] == chain_id)]
    pdb_df = pdb_df.drop_duplicates()

    _residues = pdb_df['residue_name'].to_list()
    _residues = [amino_acids[i] for i in _residues if i != "UNK"]

    sequence_features = [[residues[residue] for residue in _residues]]

    sequence_features = pad_sequences(sequence_features, maxlen=1024, truncating='post', padding='post')

    # sequences + padding
    sequence_features = torch.tensor(to_categorical(sequence_features, num_classes=len(residues) + 1))
    # sequence_features = F.one_hot(sequence_features, num_classes=len(residues) + 1).to(dtype=torch.int64)

    node_coords = torch.tensor(pdb_df[['x_coord', 'y_coord', 'z_coord']].values, dtype=torch.float32)

    return node_coords, sequence_features, ''.join(_residues)


def process_pdbpandas4(raw_path, chain_id):
    pdb_to_pandas = PandasPdb().read_pdb(raw_path)
    pdb_df = pdb_to_pandas.df['ATOM']
    
    assert (len(set(pdb_df['chain_id'])) == 1) & (list(set(pdb_df['chain_id']))[0] == chain_id)

    # 先提取 CA 原子，并按 residue_number 排序，作为基准
    ca_df = pdb_df[(pdb_df['atom_name'] == 'CA') & (pdb_df['chain_id'] == chain_id)]
    ca_df = ca_df.sort_values(by=['residue_number']).reset_index(drop=True)
    
    # 记录所有残基编号（确保顺序）
    residue_numbers = ca_df['residue_number'].values

    # 获取 CA 代表的氨基酸序列
    _residues = ca_df['residue_name'].to_list()
    _residues = [amino_acids[i] for i in _residues if i != "UNK"]

    # 转换成 one-hot 特征
    sequence_features = [[residues[residue] for residue in _residues]]
    sequence_features = pad_sequences(sequence_features, maxlen=1024, truncating='post', padding='post')
    sequence_features = torch.tensor(to_categorical(sequence_features, num_classes=len(residues) + 1))

    # 构建 N, CA, C, O 的坐标数据
    coords = []
    inf_fill = np.array([np.inf, np.inf, np.inf])  # 用 inf 处理缺失原子

    # 预先筛选出 N, C, O 原子
    other_atoms = pdb_df[(pdb_df['atom_name'].isin(['N', 'C', 'O'])) & (pdb_df['chain_id'] == chain_id)]
    
    for residue_number in residue_numbers:
        # 找到该残基的 CA 原子坐标
        ca_coord = ca_df[ca_df['residue_number'] == residue_number][['x_coord', 'y_coord', 'z_coord']].values[0]

        # 依次查找 N, C, O 原子的坐标
        atom_dict = {'N': inf_fill, 'CA': ca_coord, 'C': inf_fill, 'O': inf_fill}
        
        residue_atoms = other_atoms[other_atoms['residue_number'] == residue_number]
        for _, row in residue_atoms.iterrows():
            atom_dict[row['atom_name']] = row[['x_coord', 'y_coord', 'z_coord']].values

        # 统一存储 N, CA, C, O 坐标
        coords.append([atom_dict['N'], atom_dict['CA'], atom_dict['C'], atom_dict['O']])

    # 转换为 PyTorch Tensor
    node_coords = torch.tensor(coords, dtype=torch.float32)

    return node_coords, sequence_features, ''.join(_residues)


##### PreProcessing Utils #####

def extract_id(header):
    return header.split('|')[1]


def count_proteins(fasta_file):
    num = len([1 for line in open(fasta_file) if line.startswith(">")])
    return num


def read_dictionary(file):
    reader = csv.reader(open(file, 'r'), delimiter='\t')
    d = {}
    for row in reader:
        k, v = row[0], row[1]
        d[k] = v
    return d


def create_seqrecord(id="", name="", description="", seq=""):
    record = SeqRecord(Seq(seq), id=id, name=name, description=description)
    return record


# Count the number of protein sequences in a fasta file with biopython -- slower.
def count_proteins_biopython(fasta_file):
    num = len(list(SeqIO.parse(fasta_file, "fasta")))
    return num


def get_proteins_from_fasta(fasta_file):
    proteins = list(SeqIO.parse(fasta_file, "fasta"))
    # proteins = [i.id.split("|")[1] for i in proteins]
    proteins = [i.id for i in proteins]
    return proteins


def fasta_to_dictionary(fasta_file, identifier='protein_id'):
    if identifier == 'protein_id':
        loc = 1
    elif identifier == 'protein_name':
        loc = 2
    data = {}
    for seq_record in SeqIO.parse(fasta_file, "fasta"):
        if "|" in seq_record.id:
            data[seq_record.id.split("|")[loc]] = (
            seq_record.id, seq_record.name, seq_record.description, seq_record.seq)
        else:
            data[seq_record.id] = (seq_record.id, seq_record.name, seq_record.description, seq_record.seq)
    return data


def cafa_fasta_to_dictionary(fasta_file):
    data = {}
    for seq_record in SeqIO.parse(fasta_file, "fasta"):
        data[seq_record.description.split(" ")[0]] = (
        seq_record.id, seq_record.name, seq_record.description, seq_record.seq)
    return data


def alpha_seq_fasta_to_dictionary(fasta_file):
    data = {}
    for seq_record in SeqIO.parse(fasta_file, "fasta"):
        _protein = seq_record.id.split(":")[1].split("-")[1]
        data[_protein] = (seq_record.id, seq_record.name, seq_record.description, seq_record.seq)
    return data


def pickle_save(data, filename):
    with open('{}.pickle'.format(filename), 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load(filename):
    with open('{}.pickle'.format(filename), 'rb') as handle:
        return pickle.load(handle)


def download_msa_database(url, name):
    database_path = "./msa/hh_suite_database/{}".format(name)
    if not os.path.isdir(database_path):
        os.mkdir(database_path)
        # download database, note downloading a ~1Gb file can take a minute
        database_file = "{}/{}.tar.gz".format(database_path, name)
        subprocess.call('wget -O {} {}'.format(database_file, url), shell=True)
        # unzip the database
        subprocess.call('tar xzvf {}.tar.gz'.format(name), shell=True, cwd="{}".format(database_path))


def search_database(file, database):
    base_path = "./msa/{}"
    output_path = base_path.format("outputs/{}.hhr".format(file))
    input_path = base_path.format("inputs/{}.fasta".format(file))
    oa3m_path = base_path.format("oa3ms/{}.03m".format(file))
    database_path = base_path.format("hh_suite_database/{}/{}".format(database, database))
    if not os.path.isfile(oa3m_path):
        subprocess.call(
            'hhblits -i {} -o {} -oa3m {} -d {} -cpu 4 -n 1'.format(input_path, output_path, oa3m_path, database_path),
            shell=True)


# Just used to group msas to keep track of generation
def partition_files(group):
    from glob import glob
    dirs = glob("/data_bp/fasta_files/{}/*/".format(group), recursive=False)
    for i in enumerate(dirs):
        prt = i[1].split('/')[4]
        if int(i[0]) % 100 == 0:
            current = "/data_bp/fasta_files/{}/{}".format(group, (int(i[0]) // 100))
            if not os.path.isdir(current):
                os.mkdir(current)
        old = "/data_bp/fasta_files/{}/{}".format(group, prt)
        new = current + "/{}".format(prt)
        if old != current:
            os.rename(old, new)


# Just used to group msas to keep track of generation
def fasta_for_msas(proteins, fasta_file):
    root_dir = '/data0/wzw/PFP-Pred-Proj/Preprocessing/data_transfer/uniprot/'
    input_seq_iterator = SeqIO.parse(fasta_file, "fasta")
    num_protein = 0
    for record in input_seq_iterator:
        if num_protein % 200 == 0:
            parent_dir = root_dir + str(int(num_protein / 200))
            print(parent_dir)
            if not os.path.exists(parent_dir):
                os.mkdir(parent_dir)
        protein = extract_id(record.id)
        if protein in proteins:
            protein_dir = parent_dir + '/' + protein
            if not os.path.exists(protein_dir):
                os.mkdir(protein_dir)
            SeqIO.write(record, protein_dir + "/{}.fasta".format(protein), "fasta")


# Files to generate esm embedding for.
def fasta_for_esm(proteins, fasta_file):
    protein_path = Constants.PAESERROOT + "uniprot/{}.fasta".format("filtered")
    input_seq_iterator = SeqIO.parse(fasta_file, "fasta")

    filtered_seqs = [record for record in input_seq_iterator if extract_id(record.id) in proteins]

    if not os.path.exists(protein_path):
        SeqIO.write(filtered_seqs, protein_path, "fasta")


def get_sequence_from_pdb(pdb_file, chain_id):
    pdb_to_pandas = PandasPdb().read_pdb(pdb_file)

    pdb_df = pdb_to_pandas.df['ATOM']

    assert (len(set(pdb_df['chain_id'])) == 1) & (list(set(pdb_df['chain_id']))[0] == chain_id)

    pdb_df = pdb_df[(pdb_df['atom_name'] == 'CA') & ((pdb_df['chain_id'])[0] == chain_id)]
    pdb_df = pdb_df.drop_duplicates()

    residues = pdb_df['residue_name'].to_list()
    residues = ''.join([amino_acids[i] for i in residues if i != "UNK"])
    return residues


def is_ok(seq, MINLEN=49, MAXLEN=1022):
    """
           Checks if sequence is of good quality
           :param MAXLEN:
           :param MINLEN:
           :param seq:
           :return: None
           """
    if len(seq) < MINLEN or len(seq) >= MAXLEN:
        return False
    for c in seq:
        if c in INVALID_ACIDS:
            return False
    return True


def is_cafa_target(org):
    return org in Constants.CAFA_TARGETS


def is_exp_code(code):
    return code in Constants.exp_evidence_codes


def read_test_set(file_name):
    with open(file_name) as file:
        lines = file.readlines()
    lines = [line.rstrip('\n').split("\t")[0] for line in lines]
    return lines


def read_test_set_x(file_name):
    with open(file_name) as file:
        lines = file.readlines()
    lines = [line.rstrip('\n').split("\t") for line in lines]
    return lines


def read_test(file_name):
    with open(file_name) as file:
        lines = file.readlines()
    lines = [line.rstrip('\n') for line in lines]
    return lines


def collect_test():
    cafa3 = pickle_load(Constants.PAESERROOT + "test/test_proteins_list")
    cafa3 = set([i[0] for i in cafa3])

    new_test = set()
    for ts in Constants.TEST_GROUPS:
        # tmp = read_test_set(Constants.ROOT + "test/195-200/{}".format(ts))
        # total_test.update(set([i[0] for i in tmp]))
        tmp = read_test_set(Constants.PAESERROOT + "test/205-now/{}".format(ts))
        new_test.update(set([i[0] for i in tmp]))

    return cafa3, new_test


def test_annotation():
    # Add annotations for test set
    data = {}
    for ts in Constants.TEST_GROUPS:
        tmp = read_test_set("/data_bp/pycharm/TransFunData/data_bp/195-200/{}".format(ts))
        for i in tmp:
            if i[0] in data:
                data[i[0]][ts].add(i[1])
            else:
                data[i[0]] = {'LK_bpo': set(), 'LK_mfo': set(), 'LK_cco': set(), 'NK_bpo': set(), 'NK_mfo': set(),
                              'NK_cco': set()}
                data[i[0]][ts].add(i[1])

        tmp = read_test_set("/data_bp/pycharm/TransFunData/data_bp/205-now/{}".format(ts))
        for i in tmp:
            if i[0] in data:
                data[i[0]][ts].add(i[1])
            else:
                data[i[0]] = {'LK_bpo': set(), 'LK_mfo': set(), 'LK_cco': set(), 'NK_bpo': set(), 'NK_mfo': set(),
                              'NK_cco': set()}
                data[i[0]][ts].add(i[1])

    return data


# GO terms for test set.
def get_test_classes():
    data = set()
    for ts in Constants.TEST_GROUPS:
        tmp = read_test_set("/data_bp/pycharm/TransFunData/data_bp/195-200/{}".format(ts))
        for i in tmp:
            data.add(i[1])

        tmp = read_test_set("/data_bp/pycharm/TransFunData/data_bp/205-now/{}".format(ts))
        for i in tmp:
            data.add(i[1])

    return data


def create_cluster(seq_identity=None):
    def get_position(row, pos, column, split):
        primary = row[column].split(split)[pos]
        return primary

    computed = pd.read_pickle(Constants.ROOT + 'uniprot/set1/swissprot.pkl')
    computed['primary_accession'] = computed.apply(lambda row: get_position(row, 0, 'accessions', ';'), axis=1)
    annotated = pickle_load(Constants.ROOT + "uniprot/anotated")

    def max_go_terms(row):
        members = row['cluster'].split('\t')
        largest = 0
        max = 0
        for index, value in enumerate(members):
            x = computed.loc[computed['primary_accession'] == value]['prop_annotations'].values  # .tolist()
            if len(x) > 0:
                if len(x[0]) > largest:
                    largest = len(x[0])
                    max = index
        return members[max]

    if seq_identity is not None:
        src = "/data_bp/pycharm/TransFunData/data_bp/uniprot/set1/mm2seq_{}/max_term".format(seq_identity)
        if os.path.isfile(src):
            cluster = pd.read_pickle(src)
        else:
            cluster = pd.read_csv("/data_bp/pycharm/TransFunData/data_bp/uniprot/set1/mm2seq_{}/final_clusters.tsv"
                                  .format(seq_identity), names=['cluster'], header=None)

            cluster['rep'] = cluster.apply(lambda row: get_position(row, 0, 'cluster', '\t'), axis=1)
            cluster['max'] = cluster.apply(lambda row: max_go_terms(row), axis=1)
            cluster.to_pickle("/data_bp/pycharm/TransFunData/data_bp/uniprot/set1/mm2seq_{}/max_term".format(seq_identity))

        cluster = cluster['max'].to_list()
        computed = computed[computed['primary_accession'].isin(cluster)]

    return computed

def class_distribution_counter2(**kwargs):
    data = pickle_load(Constants.ROOT + "{}/{}/{}".format(kwargs['seq_id'], kwargs['ont'], kwargs['session']))

    all_proteins = []
    if isinstance(data, dict):
        for i in data:
            all_proteins.extend(data[i])
    elif isinstance(data, (set, list)):
        all_proteins.extend(data)
    else:
        raise TypeError(f"Unexpected data type: {type(data)}")

    annot = pd.read_csv(Constants.ROOT + 'annot.tsv', delimiter='\t')
    annot = annot.where(pd.notnull(annot), None)
    annot = annot[annot['Protein'].isin(all_proteins)]
    annot = pd.Series(annot[kwargs['ont']].values, index=annot['Protein']).to_dict()

    terms = []
    for i in annot:
        terms.extend(annot[i].split(","))

    counter = Counter(terms)

    return counter

def class_distribution_counter(**kwargs):
    """
        Count the number of proteins for each GO term in training set.
    """
    data = pickle_load(Constants.ROOT + "{}/{}/{}".format(kwargs['seq_id'], kwargs['ont'], kwargs['session']))

    all_proteins = []
    for i in data:
        all_proteins.extend(data[i])

    annot = pd.read_csv(Constants.ROOT + 'annot.tsv', delimiter='\t')
    annot = annot.where(pd.notnull(annot), None)
    annot = annot[annot['Protein'].isin(all_proteins)]
    annot = pd.Series(annot[kwargs['ont']].values, index=annot['Protein']).to_dict()

    terms = []
    for i in annot:
        terms.extend(annot[i].split(","))

    counter = Counter(terms)

    # for i in counter.most_common():
    #     print(i)
    # print("# of ontologies is {}".format(len(counter)))

    return counter


def save_ckp(state, is_best, checkpoint_path, best_model_path):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    f_path = checkpoint_path
    # save checkpoint data_bp to the path given, checkpoint_path
    torch.save(state, f_path)
    # if it is a best model, min validation loss
    if is_best:
        best_fpath = best_model_path
        # copy that checkpoint file to best path given, best_model_path
        shutil.copyfile(f_path, best_fpath)


def load_ckp(checkpoint_fpath, model, optimizer, device):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into
    optimizer: optimizer we defined in previous training
    """
    # load check point
    checkpoint = torch.load(checkpoint_fpath, map_location=torch.device(device))
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])
    # initialize valid_loss_min from checkpoint to valid_loss_min
    valid_loss_min = checkpoint['valid_loss_min']
    # return model, optimizer, epoch value, min validation loss
    return model, optimizer, checkpoint['epoch'], valid_loss_min


def draw_architecture(model, data_batch):
    '''
    Draw the network architecture.
    '''
    output = model(data_batch)
    make_dot(output, params=dict(model.named_parameters())).render("rnn_lstm_torchviz", format="png")


def compute_roc(labels, preds):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)
    return roc_auc


def generate_bulk_embedding(path_to_extract_file, fasta_file, output_dir):
    subprocess.call('python {} esm1b_t33_650M_UR50S {} {} --repr_layers 0 32 33 '
                    '--include mean per_tok --truncate'.format(path_to_extract_file,
                                                               "{}".format(fasta_file),
                                                               "{}".format(output_dir)),
                    shell=True)
    


##### MSA Utils #####

def remove_insertions(sequence: str):
    """Removes any insertions into the sequence. Needed to load aligned sequences in an MSA."""
    deletekeys = dict.fromkeys(string.ascii_lowercase)
    deletekeys["."] = None
    deletekeys["*"] = None
    translation = str.maketrans(deletekeys)
    return sequence.translate(translation)


def read_msa(filename: str) -> List[Tuple[str, str]]:
    """Reads the first nseq sequences from an MSA file, automatically removes insertions."""
    return [
        (record.description, remove_insertions(str(record.seq)))
        for record in SeqIO.parse(filename, "fasta")
    ]
    
    
def pad_sequence(seq: str, target_length: int, pad_char: str = "-") -> str:
    """填充序列到指定长度"""
    return seq.ljust(target_length, pad_char)

def read_msa2data(filename: str, max_depth: int = 1024) -> List[Tuple[str, str]]:
    """Reads MSA file, removes insertions, and ensures all sequences have the same length."""
    msa_sequences = [
        (record.description, remove_insertions(str(record.seq)))
        for record in SeqIO.parse(filename, "fasta")
    ]

    if not msa_sequences:
        return []

    # 获取参考长度（第一条序列）
    ref_length = len(msa_sequences[0][1])

    # **对短序列进行填充**
    padded_msa = [(desc, pad_sequence(seq, ref_length)) for desc, seq in msa_sequences]

    return padded_msa[:max_depth]  # 确保不超过 max_depth


def random_window(msa_data, max_len=1024):
    """
    对 MSA 数据进行随机裁剪，如果 MSA 中的某些序列长度超过 max_len。
    :param msa_data: 输入的 MSA 数据（list of tuples），每个元组是 (description, sequence)
    :param max_len: 最大序列长度
    :return: 裁剪后的 MSA 数据
    """
    # 对每个序列进行裁剪
    for i, (desc, seq) in enumerate(msa_data):
        if len(seq) > max_len:
            start_idx = random.randint(0, len(seq) - max_len)
            msa_data[i] = (desc, seq[start_idx:start_idx + max_len])
    
    return msa_data



def msa_to_tensor(msa):
    """
    将 MSA 数据从字符串形式转换为 Tensor。
    :param msa: MSA 数据，包含多个氨基酸序列（每个序列是字符串形式）
    :return: 转换后的 Tensor（每个字符映射为一个数字）
    """
    # 氨基酸字母到数字的映射
    residues = {
        "A": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7, "I": 8, "K": 9, "L": 10, "M": 11,
        "N": 12, "P": 13, "Q": 14, "R": 15, "S": 16, "T": 17, "V": 18, "W": 19, "Y": 20
    }

    # 将每个氨基酸字母转换为数字
    msa_numeric = [[residues.get(aa, 0) for aa in seq] for seq in msa]  # '0' 用于处理未知的氨基酸

    # 转换为 PyTorch tensor
    msa_tensor = torch.tensor(msa_numeric, dtype=torch.int64)

    return msa_tensor


def mask_seq_and_msa(
    seq,
    msa_batch_tokens,
    coord_mask,
    device,
    mask_size=0.15,
    mask_tok=0.60,
    mask_col=0.20,
    mask_rand=0.10,
    mask_same=0.10,
):
    # Get masked positions
    assert mask_tok + mask_col + mask_rand + mask_same == 1.00
    indices = torch.arange(len(seq), device=device)
    indices_mask = indices[coord_mask]  # Only consider indices within coord mask
    indices_mask = indices_mask[torch.randperm(indices_mask.size(0))]
    mask_pos_all = indices_mask[: int(len(indices_mask) * mask_size)]

    mask_pos_tok = mask_pos_all[: int(len(mask_pos_all) * mask_tok)]
    mask_pos_col = mask_pos_all[
        int(len(mask_pos_all) * (mask_tok)) : int(
            len(mask_pos_all) * (mask_tok + mask_col)
        )
    ]
    mask_pos_rand = mask_pos_all[
        int(len(mask_pos_all) * (mask_tok + mask_col)) : int(
            len(mask_pos_all) * (mask_tok + mask_col + mask_rand)
        )
    ]

    # Do masking - MSA level
    msa_batch_tokens_masked = msa_batch_tokens.clone()
    msa_batch_tokens_masked[:, 0, mask_pos_tok + 1] = 32  # Correct for <cls> token
    msa_batch_tokens_masked[:, :, mask_pos_col + 1] = 32  # Correct for <cls> token
    msa_batch_tokens_masked[:, 0, mask_pos_rand + 1] = torch.randint(
        low=4, high=24, size=(len(mask_pos_rand),), device=device
    )  # Correct for <cls> token, draw random standard amino acids

    # Do masking - seq level
    seq_masked = seq.clone()
    mask_pos_tok_all = torch.cat((mask_pos_tok, mask_pos_col))
    seq_masked[mask_pos_tok_all] = 20
    seq_masked[mask_pos_rand] = torch.randint(
        low=0, high=20, size=(len(mask_pos_rand),), device=device
    )

    return seq_masked, msa_batch_tokens_masked, mask_pos_all


def process_msa(batch, msa_batch_converter, rank, max_msa=16):
    """
        处理 MSA（多序列比对），包括子采样、tokenization 和掩码处理
        # model_msa_pre, msa_alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
        # msa_batch_converter = msa_alphabet.get_batch_converter()
    """
    # Subsample MSA
    msa_sub = [batch.msa[0][0]]  # 总是包含查询序列
    k = min(len(batch.msa[0]) - 1, max_msa - 1)
    msa_sub += [batch.msa[0][j] for j in sorted(random.sample(range(1, len(batch.msa[0])), k))]

    # Tokenize MSA
    _, _, msa_batch_tokens = msa_batch_converter(msa_sub)
    msa_batch_tokens = msa_batch_tokens.to(rank)

    # Mask sequence
    seq_masked, msa_batch_tokens_masked, mask_pos = mask_seq_and_msa(
        batch.seq, msa_batch_tokens, batch.mask, rank
    )

    return seq_masked, msa_batch_tokens_masked, mask_pos
    


##### GVP Utils #####

def _normalize(tensor, dim=-1):
    '''
    Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
    '''
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))


def _rbf(D, D_min=0., D_max=20., D_count=16, device='cpu'):
    '''
    From https://github.com/jingraham/neurips19-graph-protein-design
    
    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    '''
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF


class CATHDataset:
    '''
    Loader and container class for the CATH 4.2 dataset downloaded
    from http://people.csail.mit.edu/ingraham/graph-protein-design/data/cath/.
    
    Has attributes `self.train`, `self.val`, `self.test`, each of which are
    JSON/dictionary-type datasets as described in README.md.
    
    :param path: path to chain_set.jsonl
    :param splits_path: path to chain_set_splits.json or equivalent.
    '''
    def __init__(self, path, splits_path):
        with open(splits_path) as f:
            dataset_splits = json.load(f)
        train_list, val_list, test_list = dataset_splits['train'], \
            dataset_splits['validation'], dataset_splits['test']
        
        self.train, self.val, self.test = [], [], []
        
        with open(path) as f:
            lines = f.readlines()
        
        for line in tqdm.tqdm(lines):
            entry = json.loads(line)
            name = entry['name']
            coords = entry['coords']
            
            entry['coords'] = list(zip(
                coords['N'], coords['CA'], coords['C'], coords['O']
            ))
            
            if name in train_list:
                self.train.append(entry)
            elif name in val_list:
                self.val.append(entry)
            elif name in test_list:
                self.test.append(entry)
                                
class BatchSampler(data.Sampler):
    '''
    From https://github.com/jingraham/neurips19-graph-protein-design.
    
    A `torch.utils.data.Sampler` which samples batches according to a
    maximum number of graph nodes.
    
    :param node_counts: array of node counts in the dataset to sample from
    :param max_nodes: the maximum number of nodes in any batch,
                      including batches of a single element
    :param shuffle: if `True`, batches in shuffled order
    '''
    def __init__(self, node_counts, max_nodes=3000, shuffle=True):
        
        self.node_counts = node_counts
        self.idx = [i for i in range(len(node_counts))  
                        if node_counts[i] <= max_nodes]
        self.shuffle = shuffle
        self.max_nodes = max_nodes
        self._form_batches()
    
    def _form_batches(self):
        self.batches = []
        if self.shuffle: random.shuffle(self.idx)
        idx = self.idx
        while idx:
            batch = []
            n_nodes = 0
            while idx and n_nodes + self.node_counts[idx[0]] <= self.max_nodes:
                next_idx, idx = idx[0], idx[1:]
                n_nodes += self.node_counts[next_idx]
                batch.append(next_idx)
            self.batches.append(batch)
    
    def __len__(self): 
        if not self.batches: self._form_batches()
        return len(self.batches)
    
    def __iter__(self):
        if not self.batches: self._form_batches()
        for batch in self.batches: yield batch

class ProteinGraphDataset(data.Dataset):
    '''
    A map-syle `torch.utils.data.Dataset` which transforms JSON/dictionary-style
    protein structures into featurized protein graphs as described in the 
    manuscript.
    
    Returned graphs are of type `torch_geometric.data.Data` with attributes
    -x          alpha carbon coordinates, shape [n_nodes, 3]
    -seq        sequence converted to int tensor according to `self.letter_to_num`, shape [n_nodes]
    -name       name of the protein structure, string
    -node_s     node scalar features, shape [n_nodes, 6] 
    -node_v     node vector features, shape [n_nodes, 3, 3]
    -edge_s     edge scalar features, shape [n_edges, 32]
    -edge_v     edge scalar features, shape [n_edges, 1, 3]
    -edge_index edge indices, shape [2, n_edges]
    -mask       node mask, `False` for nodes with missing data that are excluded from message passing
    
    Portions from https://github.com/jingraham/neurips19-graph-protein-design.
    
    :param data_list: JSON/dictionary-style protein dataset as described in README.md.
    :param num_positional_embeddings: number of positional embeddings
    :param top_k: number of edges to draw per node (as destination node)
    :param device: if "cuda", will do preprocessing on the GPU
    '''
    def __init__(self, data_list, 
                 num_positional_embeddings=16,
                 top_k=30, num_rbf=16, device="cpu"):
        
        super(ProteinGraphDataset, self).__init__()
        
        self.data_list = data_list
        self.top_k = top_k
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings
        self.device = device
        self.node_counts = [len(e['seq']) for e in data_list]
        
        self.letter_to_num = {'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9,
                       'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8,
                       'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19, 
                       'N': 2, 'Y': 18, 'M': 12}
        self.num_to_letter = {v:k for k, v in self.letter_to_num.items()}
        
    def __len__(self): return len(self.data_list)
    
    def __getitem__(self, i): return self._featurize_as_graph(self.data_list[i])
    
    def _featurize_as_graph(self, protein):
        name = protein['name']
        with torch.no_grad():
            coords = torch.as_tensor(protein['coords'], 
                                     device=self.device, dtype=torch.float32)   
            seq = torch.as_tensor([self.letter_to_num[a] for a in protein['seq']],
                                  device=self.device, dtype=torch.long)
            
            mask = torch.isfinite(coords.sum(dim=(1,2)))
            coords[~mask] = np.inf
            
            X_ca = coords[:, 1]
            edge_index = torch_cluster.knn_graph(X_ca, k=self.top_k)
            
            pos_embeddings = self._positional_embeddings(edge_index)
            E_vectors = X_ca[edge_index[0]] - X_ca[edge_index[1]]
            rbf = _rbf(E_vectors.norm(dim=-1), D_count=self.num_rbf, device=self.device)
            
            dihedrals = self._dihedrals(coords)                     
            orientations = self._orientations(X_ca)
            sidechains = self._sidechains(coords)
            
            node_s = dihedrals
            node_v = torch.cat([orientations, sidechains.unsqueeze(-2)], dim=-2)
            edge_s = torch.cat([rbf, pos_embeddings], dim=-1)
            edge_v = _normalize(E_vectors).unsqueeze(-2)
            
            node_s, node_v, edge_s, edge_v = map(torch.nan_to_num,
                    (node_s, node_v, edge_s, edge_v))
            
        data = torch_geometric.data.Data(x=X_ca, seq=seq, name=name,
                                         node_s=node_s, node_v=node_v,
                                         edge_s=edge_s, edge_v=edge_v,
                                         edge_index=edge_index, mask=mask)
        return data
                                
    def _dihedrals(self, X, eps=1e-7):
        # From https://github.com/jingraham/neurips19-graph-protein-design
        
        X = torch.reshape(X[:, :3], [3*X.shape[0], 3])
        dX = X[1:] - X[:-1]
        U = _normalize(dX, dim=-1)
        u_2 = U[:-2]
        u_1 = U[1:-1]
        u_0 = U[2:]

        # Backbone normals
        n_2 = _normalize(torch.cross(u_2, u_1), dim=-1)
        n_1 = _normalize(torch.cross(u_1, u_0), dim=-1)

        # Angle between normals
        cosD = torch.sum(n_2 * n_1, -1)
        cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
        D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)

        # This scheme will remove phi[0], psi[-1], omega[-1]
        D = F.pad(D, [1, 2]) 
        D = torch.reshape(D, [-1, 3])
        # Lift angle representations to the circle
        D_features = torch.cat([torch.cos(D), torch.sin(D)], 1)
        return D_features
    
    
    def _positional_embeddings(self, edge_index, 
                               num_embeddings=None,
                               period_range=[2, 1000]):
        # From https://github.com/jingraham/neurips19-graph-protein-design
        num_embeddings = num_embeddings or self.num_positional_embeddings
        d = edge_index[0] - edge_index[1]
     
        frequency = torch.exp(
            torch.arange(0, num_embeddings, 2, dtype=torch.float32, device=self.device)
            * -(np.log(10000.0) / num_embeddings)
        )
        angles = d.unsqueeze(-1) * frequency
        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
        return E

    def _orientations(self, X):
        forward = _normalize(X[1:] - X[:-1])
        backward = _normalize(X[:-1] - X[1:])
        forward = F.pad(forward, [0, 0, 0, 1])
        backward = F.pad(backward, [0, 0, 1, 0])
        return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)

    def _sidechains(self, X):
        n, origin, c = X[:, 0], X[:, 1], X[:, 2]
        c, n = _normalize(c - origin), _normalize(n - origin)
        bisector = _normalize(c + n)
        perp = _normalize(torch.cross(c, n))
        vec = -bisector * math.sqrt(1 / 3) - perp * math.sqrt(2 / 3)
        return vec 



##### GO Embedding #####

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

def go_main(go_filename, go_ids):
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
    
    
##### Joint Embedding ##### 

class CrossModalAttention(nn.Module):
    def __init__(self, protein_dim=1229, go_dim=200, hidden_dim=512):
        super().__init__()
        self.proj_p = nn.Linear(protein_dim, hidden_dim)
        self.proj_g = nn.Linear(go_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=4)

    def forward(self, proteins, gos):
        P = self.proj_p(proteins)  # [batch, 512]
        G = self.proj_g(gos)       # [batch, 512]

        attn_output, _ = self.attn(P.unsqueeze(1), G.unsqueeze(1), G.unsqueeze(1))
        fused_feature = P + attn_output.squeeze(1)
        return fused_feature


##### S-PLM #####
def focal_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = 0.25,
        gamma: float = 2,
        reduction: str = "none", ) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs (Tensor): A float tensor of arbitrary shape.
                The predictions for each example.
        targets (Tensor): A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha (float): Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for ignore. Default: ``0.25``.
        gamma (float): Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples. Default: ``2``.
        reduction (string): ``'none'`` | ``'mean'`` | ``'sum'``
                ``'none'``: No reduction will be applied to the output.
                ``'mean'``: The output will be averaged.
                ``'sum'``: The output will be summed. Default: ``'none'``.
    Returns:
        Loss tensor with the reduction option applied.
    """
    # Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py
    # p = torch.sigmoid(inputs) #My pred is already from 0 to 1 no need to use binary_cross_entropy_with_sigmoid
    p = inputs
    ce_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    # Check reduction option and return loss accordingly
    if reduction == "none":
        pass
    elif reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    else:
        raise ValueError(
            f"Invalid Value for arg 'reduction': '{reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
        )
    return loss


def truncate_seq(seq, max_len):  # padding
    return [seqi[:max_len] for seqi in seq]


def f1_max(pred, target):
    """
    copied from https://torchdrug.ai/docs/_modules/torchdrug/metrics/metric.html#f1_max
    F1 score with the optimal threshold.

    This function first enumerates all possible thresholds for deciding positive and negative
    samples, and then pick the threshold with the maximal F1 score.

    Parameters:
        pred (Tensor): predictions of shape :math:`(B, N)`
        target (Tensor): binary targets of shape :math:`(B, N)`
    """

    order = pred.argsort(descending=True, dim=1)
    target = target.gather(1, order)
    precision = target.cumsum(1) / torch.ones_like(target).cumsum(1)
    recall = target.cumsum(1) / (target.sum(1, keepdim=True) + 1e-10)
    is_start = torch.zeros_like(target).bool()
    is_start[:, 0] = 1
    is_start = torch.scatter(is_start, 1, order, is_start)

    all_order = pred.flatten().argsort(descending=True)
    order = order + torch.arange(order.shape[0], device=order.device).unsqueeze(1) * order.shape[1]
    order = order.flatten()
    inv_order = torch.zeros_like(order)
    inv_order[order] = torch.arange(order.shape[0], device=order.device)
    is_start = is_start.flatten()[all_order]
    all_order = inv_order[all_order]
    precision = precision.flatten()
    recall = recall.flatten()
    all_precision = precision[all_order] - \
                    torch.where(is_start, torch.zeros_like(precision), precision[all_order - 1])
    all_precision = all_precision.cumsum(0) / is_start.cumsum(0)
    all_recall = recall[all_order] - \
                 torch.where(is_start, torch.zeros_like(recall), recall[all_order - 1])
    all_recall = all_recall.cumsum(0) / pred.shape[0]
    all_f1 = 2 * all_precision * all_recall / (all_precision + all_recall + 1e-10)
    return all_f1.max()


def Fmax_func(predictions, targets, binnum=10):
    custom_thresholds = np.linspace(0, 1, binnum)[1:-1]
    precision_at_thresholds = []
    recall_at_thresholds = []
    num_samples = len(predictions)
    for sample_index in range(num_samples):  # Loop through each class
        preds = predictions[sample_index, :]

        # Calculate precision and recall at custom thresholds
        precisions = [precision_score(targets[sample_index, :], (preds >= threshold), zero_division=0) for threshold in
                      custom_thresholds]
        recalls = [recall_score(targets[sample_index, :], (preds >= threshold), zero_division=0) for threshold in
                   custom_thresholds]

        precision_at_thresholds.append(precisions)
        recall_at_thresholds.append(recalls)

    # Calculate average precision and recall
    avg_precision = np.mean(precision_at_thresholds, 0)
    avg_recall = np.mean(recall_at_thresholds, 0)
    f_scores = [2 * (p * r) / (p + r + 1e-8) for (p, r) in zip(avg_precision, avg_recall)]
    # Calculate F1-score using average precision and recall
    return np.max(f_scores)


def print_gpu_memory_allocation(logging):
    """
    Prints the memory allocation and caching for each available GPU using PyTorch.
    """
    if not torch.cuda.is_available():
        print("No GPUs detected!")
        return

    num_gpus = torch.cuda.device_count()

    for i in range(num_gpus):
        total_mem = torch.cuda.get_device_properties(i).total_memory / 1e9  # in GB
        allocated_mem = torch.cuda.memory_allocated(i) / 1e9  # in GB
        cached_mem = torch.cuda.memory_reserved(i) / 1e9  # in GB

        logging.info(f"GPU {i}:")
        logging.info(f"\tTotal Memory: {total_mem:.2f} GB")
        logging.info(f"\tAllocated Memory: {allocated_mem:.2f} GB")
        logging.info(f"\tCached (Reserved) Memory: {cached_mem:.2f} GB")
        logging.info("-" * 50)


def calculate_class_weights(class_samples):
    total_samples = sum(class_samples.values())
    class_weights = {}

    # Calculate weights using the inverse of the class frequencies
    for class_name, samples in class_samples.items():
        class_weights[class_name] = total_samples / samples

    # Normalize the weights so that the largest class has a weight of 1
    min_weight = min(class_weights.values())
    for class_name, weight in class_weights.items():
        class_weights[class_name] = weight / min_weight

    return class_weights


def calculate_class_weights_normalized(samples_dict):
    """
    Calculate the weights for each class based on the number of samples.

    :param samples_dict: Dictionary containing classes as keys and number of samples as values.
    :return: Dictionary containing classes as keys and their respective weights as values.
    """
    min_samples = min(samples_dict.values())

    weights_dict = {}
    for key, value in samples_dict.items():
        weights_dict[key] = min_samples / value

    return weights_dict


def load_configs(config, args=None):
    """
        Load the configuration file and convert the necessary values to floats.

        Args:
            config (dict): The configuration dictionary.

        Returns:
            The updated configuration dictionary with float values.
        """

    # Convert the dictionary to a Box object for easier access to the values.
    tree_config = Box(config)

    # Convert the necessary values to floats.
    tree_config.optimizer.lr = float(tree_config.optimizer.lr)
    tree_config.optimizer.decay.min_lr = float(tree_config.optimizer.decay.min_lr)
    tree_config.optimizer.weight_decay = float(tree_config.optimizer.weight_decay)
    tree_config.optimizer.eps = float(tree_config.optimizer.eps)
    # overwrite parameters if set through commandline
    #print("num_end_adapter_layers!!!!!!!!")
    #print(tree_config.encoder.adapter_h.num_end_adapter_layers)
    if args is not None:
        if args.result_path:
            tree_config.result_path = args.result_path

        if args.resume_path:
            tree_config.resume.resume_path = args.resume_path
            #tree_config.resume.enable = True #if set by args, the resume enable will be overwrite as True
        
        if args.num_end_adapter_layers:
            if not isinstance(args.num_end_adapter_layers, list):
                if "-" in args.num_end_adapter_layers:
                   args.num_end_adapter_layers = args.num_end_adapter_layers.split("-")
                else:
                   args.num_end_adapter_layers = [args.num_end_adapter_layers]
            
            args.num_end_adapter_layers = [int(x) for x in args.num_end_adapter_layers]
            tree_config.encoder.adapter_h.num_end_adapter_layers = args.num_end_adapter_layers

        if args.module_type:
            tree_config.encoder.adapter_h.module_type = args.module_type

    # print("num_end_adapter_layers!!!!!!!!")
    # print(tree_config.encoder.adapter_h.num_end_adapter_layers)

    # print("freeze_adapter_layers!!!!!!!!")
    # print(tree_config.encoder.adapter_h.freeze_adapter_layers)
    return tree_config


def prepare_saving_dir(configs, config_file_path):
    """
    Prepare a directory for saving a training results.

    Args:
        configs: A python box object containing the configuration options.

    Returns:
        str: The path to the directory where the results will be saved.
    """
    # Create a unique identifier for the run based on the current time.
    run_id = datetime.datetime.now().strftime('%Y-%m-%d__%H-%M-%S')

    # Add '_evaluation' to the run_id if the 'evaluate' flag is True.
    # if configs.evaluate:
    #     run_id += '_evaluation'

    # Create the result directory and the checkpoint subdirectory.
    result_path = os.path.abspath(os.path.join(configs.result_path, run_id))
    checkpoint_path = os.path.join(result_path, 'checkpoints')
    Path(result_path).mkdir(parents=True, exist_ok=True)
    Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

    # Copy the config file to the result directory.
    shutil.copy(config_file_path, result_path)

    # Return the path to the result directory.
    return result_path, checkpoint_path


def prepare_optimizer(net, configs, num_train_samples, logging):
    optimizer, scheduler = load_opt(net, configs, logging)
    if scheduler is None:
        if configs.optimizer.decay.first_cycle_steps:
            first_cycle_steps = configs.optimizer.decay.first_cycle_steps
        else:
            whole_steps = np.ceil(
                num_train_samples / configs.train_settings.grad_accumulation
            ) * configs.train_settings.num_epochs
            first_cycle_steps = np.ceil(whole_steps / configs.optimizer.decay.num_restarts)

        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=first_cycle_steps,
            cycle_mult=1.0,
            max_lr=configs.optimizer.lr,
            min_lr=configs.optimizer.decay.min_lr,
            warmup_steps=configs.optimizer.decay.warmup,
            gamma=configs.optimizer.decay.gamma)

    return optimizer, scheduler


def load_opt(model, config, logging):
    scheduler = None
    if config.optimizer.name.lower() == 'adabelief':
        opt = optim.AdaBelief(model.parameters(), lr=config.optimizer.lr, eps=config.optimizer.eps,
                              decoupled_decay=True,
                              weight_decay=config.optimizer.weight_decay, rectify=False)
    elif config.optimizer.name.lower() == 'adam':
        # opt = eval('torch.optim.' + config.optimizer.name)(model.parameters(), lr=config.optimizer.lr, eps=eps,
        #                                       weight_decay=config.optimizer.weight_decay)
        if config.optimizer.use_8bit_adam:
            import bitsandbytes
            logging.info('use 8-bit adamw')
            opt = bitsandbytes.optim.AdamW8bit(
                model.parameters(), lr=float(config.optimizer.lr),
                betas=(config.optimizer.beta_1, config.optimizer.beta_2),
                weight_decay=float(config.optimizer.weight_decay),
                eps=float(config.optimizer.eps),
            )
        else:
            opt = torch.optim.AdamW(
                model.parameters(), lr=float(config.optimizer.lr),
                betas=(config.optimizer.beta_1, config.optimizer.beta_2),
                weight_decay=float(config.optimizer.weight_decay),
                eps=float(config.optimizer.eps)
            )

    else:
        raise ValueError('wrong optimizer')
    return opt, scheduler


def load_checkpoints_only(checkpoint_path,  model):
    model_checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'state_dict1' in model_checkpoint:
        #to load old checkpoints that saved adapter_layer_dict as adapter_layer. 
        from collections import OrderedDict
        if np.sum(["adapter_layer_dict" in key for key in model_checkpoint['state_dict1'].keys()])==0: #using old checkpoints, need to rename the adapter_layer into adapter_layer_dict.adapter_0
             new_ordered_dict = OrderedDict()
             for key, value in model_checkpoint['state_dict1'].items():
                 if "adapter_layer_dict" not in key:
                   new_key = key.replace('adapter_layer', 'adapter_layer_dict.adapter_0')
                   new_ordered_dict[new_key] = value
                 else:
                   new_ordered_dict[key] = value
             
             model.load_state_dict(new_ordered_dict,strict=False)
        else: #new checkpoints with new code, that can be loaded directly.
              model.load_state_dict(model_checkpoint['state_dict1'], strict=False)
    elif 'model_state_dict' in model_checkpoint:
          model.load_state_dict(model_checkpoint['model_state_dict'], strict=False)
    


def load_checkpoints(configs, optimizer, scheduler, logging, net):
    """
    Load saved checkpoints from a previous training session.

    Args:
        configs: A python box object containing the configuration options.
        optimizer (Optimizer): The optimizer to resume training with.
        scheduler (Scheduler): The learning rate scheduler to resume training with.
        logging (Logger): The logger to use for logging messages.
        net (nn.Module): The neural network model to load the saved checkpoints into.

    Returns:
        tuple: A tuple containing the loaded neural network model and the epoch to start training from.
    """
    start_epoch = 1

    # If the 'resume' flag is True, load the saved model checkpoints.
    if configs.resume.enable:
        model_checkpoint = torch.load(configs.resume.resume_path, map_location='cpu')
        #net.load_state_dict(model_checkpoint['state_dict1'], strict=False)
        if 'state_dict1' in model_checkpoint:
            #to load old checkpoints that saved adapter_layer_dict as adapter_layer. 
            from collections import OrderedDict
            if np.sum(["adapter_layer_dict" in key for key in model_checkpoint['state_dict1'].keys()])==0: #using old checkpoints, need to rename the adapter_layer into adapter_layer_dict.adapter_0
                 new_ordered_dict = OrderedDict()
                 for key, value in model_checkpoint['state_dict1'].items():
                     if "adapter_layer_dict" not in key:
                       new_key = key.replace('adapter_layer', 'adapter_layer_dict.adapter_0')
                       new_ordered_dict[new_key] = value
                     else:
                       new_ordered_dict[key] = value
                 
                 net.load_state_dict(new_ordered_dict,strict=False)
            else: #new checkpoints with new code, that can be loaded directly.
                  net.load_state_dict(model_checkpoint['state_dict1'], strict=False)
        elif 'model_state_dict' in model_checkpoint:
              net.load_state_dict(model_checkpoint['model_state_dict'], strict=False)
        
        logging.info(f'model checkpoint is loaded from: {configs.resume.resume_path}')
        # If the saved checkpoint contains the optimizer and scheduler states and the epoch number,
        # resume training from the last saved epoch.
        if 'optimizer_state_dict' in model_checkpoint and 'scheduler_state_dict' in model_checkpoint and 'epoch' in model_checkpoint:
            if not configs.resume.restart_optimizer:
                optimizer.load_state_dict(model_checkpoint['optimizer_state_dict'])
                logging.info('Optimizer is loaded to resume training!')

                scheduler.load_state_dict(model_checkpoint['scheduler_state_dict'])
                logging.info('Scheduler is loaded to resume training!')
                start_epoch = model_checkpoint['epoch'] + 1

    # Return the loaded model and the epoch to start training from.
    return net, start_epoch


def save_checkpoint(epoch: int, model_path: str, tools: dict, accelerator: Accelerator):
    """
    Save the model checkpoints during training.

    Args:
        epoch (int): The current epoch number.
        model_path (str): The path to save the model checkpoint.
        tools (dict): A dictionary containing the necessary tools for saving the model checkpoints.
        accelerator (Accelerator): Accelerator object.

    Returns:
        None
    """
    # # Set the path to save the model checkpoint.
    # model_path = os.path.join(tools['result_path'], 'checkpoints', f'checkpoint_{epoch}.pth')

    # Save the model checkpoint.
    torch.save({
        'epoch': epoch,
        'model_state_dict': accelerator.unwrap_model(tools['net'].state_dict()),
        'optimizer_state_dict': accelerator.unwrap_model(tools['optimizer'].state_dict()),
        'scheduler_state_dict': accelerator.unwrap_model(tools['scheduler'].state_dict()),
    }, model_path)


def test_gpu_cuda():
    print('Testing gpu and cuda:')
    print('\tcuda is available:', torch.cuda.is_available())
    print('\tdevice count:', torch.cuda.device_count())
    print('\tcurrent device:', torch.cuda.current_device())
    print(f'\tdevice:', torch.cuda.device(0))
    print('\tdevice name:', torch.cuda.get_device_name(), end='\n\n')


def prepare_tensorboard(result_path):
    train_path = os.path.join(result_path, 'train')
    val_path = os.path.join(result_path, 'val')
    Path(train_path).mkdir(parents=True, exist_ok=True)
    Path(val_path).mkdir(parents=True, exist_ok=True)

    train_log_path = os.path.join(train_path, 'tensorboard')
    train_writer = SummaryWriter(train_log_path)

    val_log_path = os.path.join(val_path, 'tensorboard')
    val_writer = SummaryWriter(val_log_path)

    return train_writer, val_writer


def get_dummy_logging():
    logger = log.getLogger(__name__)
    logger.addHandler(log.NullHandler())
    return logger


def get_logging_old(result_path):
    log.basicConfig(filename=os.path.join(result_path, "logs.txt"),
                    format='%(asctime)s - %(message)s',
                    filemode='a')
    log.getLogger().setLevel(log.INFO)
    log.getLogger().addHandler(log.StreamHandler())
    return log


def get_logging(result_path):
    logger = log.getLogger(result_path)
    logger.setLevel(log.INFO)

    fh = log.FileHandler(os.path.join(result_path, "logs.txt"))
    formatter = log.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = log.StreamHandler()
    logger.addHandler(sh)

    return logger


def save_model(epoch, model, opt, result_path, scheduler, description='best_model'):
    Path(os.path.join(result_path, 'checkpoints')).mkdir(parents=True, exist_ok=True)
    torch.save({
        'model': model.state_dict(),
        'optimizer': opt.state_dict(),
        'lr_scheduler': scheduler.state_dict(),
        'epoch': epoch
    }, os.path.join(result_path, 'checkpoints', description + '.pth'))


def random_pick(input_list, num_to_pick, seed):
    # Set the random seed
    random.seed(seed)

    # Check if num_to_pick is greater than the length of the input_list
    if num_to_pick > len(input_list):
        print("Number to pick is greater than the length of the input list")
        return input_list

    # Use random.sample to pick num_to_pick items from the input_list
    random_items = random.sample(input_list, num_to_pick)

    return random_items


if __name__ == '__main__':
    # For test utils modules
    print('done')



##### Metrix ##### 

def _micro_aupr(y_true, y_test):
    """
    Computes the micro AUPR

    Args:
        y_true: array with the GT observations.
        y_test: array with the predictions.
    Returns
        float representing the micro aupr score
    """
    return metrics.average_precision_score(y_true, y_test, average="micro")


def compute_f1_score_at_threshold(
    y_true: np.ndarray, y_pred: np.ndarray, t: float
):
    """Calculate protein-centric F1 score based on DeepFRI's description.
    ref: https://www.nature.com/articles/nmeth.2340
    Online method -> Evaluation metrics

    Args:
        y_true: [n_proteins, n_functions], binary matrix of ground truth labels
        y_pred: [n_proteins, n_functions], probabilities from model predictions after sigmoid.
        t: Float representing the threshold to use to compute the f1 score.

    Returns:
        float representing the f1 score
    """
    n_proteins = y_true.shape[0]
    y_pred_bin = y_pred >= t  # binarize predictions
    pr = []
    rc = []
    for i in range(n_proteins):
        if y_pred_bin[i].sum() > 0:
            pr_i = metrics.precision_score(y_true[i], y_pred_bin[i])
            pr.append(pr_i)

        rc_i = metrics.recall_score(y_true[i], y_pred_bin[i])
        rc.append(rc_i)

    pr = np.mean(pr)
    rc = np.mean(rc)
    return 2 * pr * rc / (pr + rc)


def evaluate_multilabel(
    y_true: np.ndarray, y_pred: np.ndarray, n_thresholds=100
):
    """Calculate protein-centric F_max and function-centric AUPR
    based on DeepFRI's description.
    ref: https://www.nature.com/articles/nmeth.2340
    Online method -> Evaluation metrics
    Args:
        y_true: [n_proteins, n_functions], binary matrix of ground truth labels
        y_pred: [n_proteins, n_functions], logits from model predictions
        n_thresholds (int): number of thresholds to estimate F_max

    Returns:
        Tuple where the first element is the F1 score and the second element is the micro AUPR
    """
    # function-centric AUPR
    micro_aupr = _micro_aupr(y_true, y_pred)

    # apply sigmoid to logits
    # y_pred = 1 / (1 + np.exp(-y_pred))

    thresholds = np.linspace(0.0, 1.0, n_thresholds, endpoint=False)
    f_scores = Parallel(n_jobs=-1, verbose=10)(
        delayed(compute_f1_score_at_threshold)(y_true, y_pred, thresholds[i])
        for i in range(n_thresholds)
    )
    
    return np.nanmax(f_scores), micro_aupr


def auprc(ytrue, ypred):
    p, r, t = precision_recall_curve(ytrue, ypred)
    return auc(r, p)


def CAFA_Metric(ytrue1, ypred1):
    fmax = 0
    prec = []
    rc = []
    ytrue = []
    ypred = []

    for i in range(len(ytrue1)):
        if np.sum(ytrue1[i]) > 0:
            ytrue.append(ytrue1[i])
            ypred.append(ypred1[i])

    for t in range(1, 101):
        thres = t / 100.

        thres_value = np.ones((len(ytrue), len(ytrue[0])), dtype=np.float32) * thres

        pred_values = np.greater(ypred, thres_value).astype(int)

        tp_matrix = pred_values * ytrue

        tp = np.sum(tp_matrix, axis=1, dtype=np.int32)
        tpfp = np.sum(pred_values, axis=1)
        tpfn = np.sum(ytrue, axis=1)

        avgprs = []

        for i in range(len(tp)):
            if tpfp[i] != 0:
                avgprs.append(tp[i] / float(tpfp[i]))

        if len(avgprs) == 0:
            continue
        avgpr = np.mean(avgprs)
        avgrc = np.mean(tp / tpfn)
        avgpr = float(avgpr)
        avgrc = float(avgrc)

        prec.append(avgpr)
        rc.append(avgrc)
        if avgrc == 0 and avgpr == 0:
            f1 = 0
        else:
            f1 = 2 * avgpr * avgrc / (avgpr + avgrc)

        fmax = max(fmax, f1)

    print('metric--------------------------')
    print('fmax:', fmax)
    aup = auprc(np.array(ytrue).flatten(), np.array(ypred).flatten())
    print('AUPR:', aup)
    print('--------------------------')
    return fmax,aup


##### stru #####

# set graph construct
def build_graph_from_sequence(self, sequence: str, method: str = 'full', threshold: float = 8.0):
    """
    主入口：根据序列生成图结构
    支持方式：'full', 'linear', 'contact_map', 'esmfold', 'alphafold', 'rosettafold'
    """
    # 确保 sequence 是一个字符串
    if isinstance(sequence, list):
        sequence = ''.join(sequence)  # 如果是列表，转换为字符串
    sequence = sequence.strip()
    length = len(sequence)
    if length <= 1:
        return torch.tensor([[0], [0]], dtype=torch.long, device=self.device)
    if method in ['full', 'linear']:
        return self._build_simple_graph(length, method)
    # 预测结构
    coords = self._predict_structure(sequence, method)
    if coords is None:
        raise ValueError(f"无法预测结构, method={method}")
    if method == 'contact_map':
        dist_mat = torch.cdist(coords, coords)
        edge_index = (dist_mat < threshold).nonzero(as_tuple=False).t().to(self.device)
        return edge_index
    else:
        # esmfold, alphafold, rosettafold 都用 coords + 距离阈值连边
        return self._build_contact_graph_from_coords(coords, threshold)
def _build_simple_graph(self, length, method):
    if method == 'full':
        edges = list(itertools.combinations(range(length), 2))
        src, dst = zip(*edges)
        edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long, device=self.device)
    elif method == 'linear':
        src = list(range(length - 1))
        dst = list(range(1, length))
        edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long, device=self.device)
    return edge_index
def _predict_structure(self, sequence, method):
    if method == 'esmfold':
        return self._run_esmfold(sequence)
    elif method == 'alphafold':
        return self._run_alphafold(sequence)
    elif method == 'rosettafold':
        return self._run_rosettafold(sequence)
    elif method == 'contact_map':
        return self._run_contact_map(sequence)  # Contact map 需要坐标
    else:
        return None
def _run_esmfold(self, sequence):
    if self.esmfold_model is None:
        print("加载 ESMFold 模型中...")
        self.esmfold_model = pretrained.esmfold_v1().eval().to(self.device)
    
    with torch.no_grad():
        output = self.esmfold_model.infer_pdb(sequence)
        coords = self.esmfold_model.output_coords.squeeze(0)  # [N, 3]
    return coords.to(self.device)

def _run_alphafold(self, sequence):
    if self.alphafold_model is None:
        print("加载 AlphaFold 模型中...")
        self.alphafold_model = pretrained.alphafold_v2().eval().to(self.device) 
    
    with torch.no_grad():
        output = self.alphafold_model.infer_pdb(sequence)
        coords = self.alphafold_model.output_coords.squeeze(0)  # [N, 3]
    return coords.to(self.device)

def _run_rosettafold(self, sequence):
    if self.rosettafold_model is None:
        print("加载 RosettaFold 模型中...")
        self.rosettafold_model = pretrained.rosettafold_v1().eval().to(self.device)  
    with torch.no_grad():
        output = self.rosettafold_model.infer_pdb(sequence)
        coords = self.rosettafold_model.output_coords.squeeze(0)  # [N, 3]
    return coords.to(self.device)

def _run_contact_map(self, sequence):
    if self.contact_map_model is None:
        print("加载 Contact Map 模型中...")
        self.contact_map_model = pretrained.contact_map_v1().eval().to(self.device) 
    
    with torch.no_grad():
        output = self.contact_map_model.infer_pdb(sequence)
        coords = self.contact_map_model.output_coords.squeeze(0)  # [N, 3]
    return coords.to(self.device)

def _build_contact_graph_from_coords(self, coords, threshold=8.0):
    dist_mat = torch.cdist(coords, coords)
    edge_index = (dist_mat < threshold).nonzero(as_tuple=False).t().to(self.device)
    return edge_index