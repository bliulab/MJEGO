import os
import math
import pickle
import subprocess
import torch
import os.path as osp
from torch_geometric.data import Dataset, download_url, HeteroData
import sys, gzip, csv, math
from pathlib import Path
from typing import List, Optional
import Utils.Constants
import requests
from Dataset.distanceTransform import myDistanceTransform
from Dataset.myKnn import myKNNGraph
from Dataset.myRadiusGraph import myRadiusGraph
from Utils.utils import find_files, process_pdbpandas, get_knn, generate_Identity_Matrix, process_pdbpandas4
import torch_geometric.transforms as T
from torch_geometric.data import Data
from Dataset.AdjacencyTransform import AdjacencyFeatures
from Utils.utils import pickle_load
import pandas as pd
import random
from Preprocessing.get_esm_msa1b_rep import get_esm_msa1b_rep
from Preprocessing.go_emb import *


class PDBDataset(Dataset):

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, **kwargs):

        self.root = root
        self.seq_id = kwargs.get('seq_id', None)
        self.ont = kwargs.get('ont', None)
        self.session = kwargs.get('session', None)
        self.prot_ids = kwargs.get('prot_ids', [])
        self.test_file = kwargs.get('test_file', None)
        self.pdb_pth = kwargs.get('pdb_path', self.root + "alphafold/")

        self.raw_file_list = []
        self.processed_file_list = []

        if self.session == "selected":
            self.data = self.prot_ids
            for i in self.data:
                self.raw_file_list.append('{}'.format(i))
                self.processed_file_list.append('{}.pt'.format(i))
        else:
            if self.session == "train":
                self.data = pickle_load(self.root + "/{}/{}/{}".format(self.seq_id, self.ont, self.session)) # 字典： 序号：蛋白质名称
                for i in self.data:
                    for j in self.data[i]:
                        self.raw_file_list.append('AF-{}-F1-model_v4.pdb.gz'.format(j))
                        self.processed_file_list.append('{}.pt'.format(j))
            elif self.session == "validation":
                self.data = list(pickle_load(self.root + "{}/{}".format(self.seq_id, self.session)))
                for i in self.data:
                    self.raw_file_list.append('AF-{}-F1-model_v4.pdb.gz'.format(i))
                    self.processed_file_list.append('{}.pt'.format(i))

        super().__init__(self.root, transform, pre_transform, pre_filter)

    @property
    def raw_dir(self) -> str:
        return self.pdb_pth

    @property
    def processed_dir(self) -> str:
        return self.root + "/esm2data/"

    @property
    def raw_file_names(self):
        return self.raw_file_list

    @property
    def processed_file_names(self):
        return self.processed_file_list
    
    def download_structure(name: str, save_dir: Path) -> Optional[Path]:
        """
        自动下载 AlphaFold 预测结构或 PDB 实验结构。
        返回保存路径；下载失败返回 None。
        """
        # 1) Frist Try RCSB PDB
        if len(name) == 4 and name[0].isdigit():
            url = f"https://files.rcsb.org/download/{name.upper()}.pdb"
            dst = save_dir / f"{name}.pdb"
        else:
            # 2) Default AlphaFold DB
            url = f"https://alphafold.ebi.ac.uk/files/AF-{name}-F1-model_v4.pdb"
            dst = save_dir / f"{name}.pdb"

        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            dst.write_bytes(r.content)
            print(f"  ↳ 已下载 {url} → {dst}")
            return dst
        except Exception as e:
            print(f"  ↳ 下载失败 ({e.__class__.__name__}): {url}")
            return None

    # Parser Protein Data
    def process(self):
        rem_files = set(self.processed_file_list) - set(find_files(self.processed_dir, suffix="pt", type="Name"))
        print("{} unprocessed proteins out of {}".format(len(rem_files), len(self.processed_file_list)))
        chain_id = 'A'

        for file in rem_files:
            protein = file.split(".")[0]
            print("Processing protein {}".format(protein))

            raw_path = self.raw_dir + '{}.pdb.gz'.format(protein)

            labels = {
                'molecular_function': [],
                'biological_process': [],
                'cellular_component': []
            }


            # Parser ESM Data
            emb = torch.load(self.root + "/esm2/{}.pt".format(protein))
            embedding_features_per_residue = emb['representations'][33]
            embedding_features_per_sequence = emb['mean_representations'][33].view(1, -1)

            # Paser MSA Data
            msa_file_path = f"/data0/wzw/msa2202/msas/{protein}.a3m"
            msa_rep = get_esm_msa1b_rep(a3m_path=msa_file_path, num_seqs=256, device='cuda')
            msa_rep = msa_rep.squeeze(0).to('cpu')


            if raw_path:
                node_coords, sequence_features, sequence_letters = process_pdbpandas(raw_path, chain_id)

            assert embedding_features_per_residue.shape[0] == node_coords.shape[0]
            assert embedding_features_per_residue.shape[1] == embedding_features_per_sequence.shape[1]

            node_size = node_coords.shape[0]
            names = torch.arange(0, node_size, dtype=torch.int8)

            # Set Hetero Graph
            data = HeteroData()
            data['atoms'].pos = node_coords

            data['atoms'].biological_process = torch.IntTensor(labels['biological_process'])
            data['atoms'].molecular_function = torch.IntTensor(labels['molecular_function'])
            data['atoms'].cellular_component = torch.IntTensor(labels['cellular_component'])

            data['atoms'].sequence_features = sequence_features
            data['atoms'].embedding_features_per_residue = embedding_features_per_residue
            data['atoms'].names = names
            data['atoms'].sequence_letters = sequence_letters
            data['atoms'].embedding_features_per_sequence = embedding_features_per_sequence
            data['atoms'].protein = protein

            # MSA
            data['atoms'].msa = msa_rep

            # GO
            # data['atoms'].go_bp = go_emb('biological_process', 200)
            # data['atoms'].go_mf = go_emb('molecular_function', 200)
            # data['atoms'].go_cc = go_emb('cellular_component', 200)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                _transforms = []
                for i in self.pre_transform:
                    if i[0] == "KNN":
                        kwargs = {'mode': i[1], 'sequence_length': node_size}
                        knn = get_knn(**kwargs)
                        _transforms.append(myKNNGraph(i[1], k=knn, force_undirected=True, ))
                    if i[0] == "DIST":
                        _transforms.append(myRadiusGraph(i[1], r=i[2], loop=False))
                _transforms.append(myDistanceTransform(edge_types=self.pre_transform, norm=True))
                _transforms.append(AdjacencyFeatures(edge_types=self.pre_transform))

                pre_transform = T.Compose(_transforms)
                data = pre_transform(data)

            torch.save(data, osp.join(self.root + "/esm2data/", f'{protein}.pt'))

    def len(self):
        return len(self.data)

    def get(self, idx):
        if self.session == "train":
            rep = random.sample(self.data[idx], 1)[0]
            return torch.load(osp.join(self.processed_dir, f'{rep}.pt'))
        elif self.session == "validation" or self.session == "selected" or self.session == "test":
            rep = self.data[idx]
            return torch.load(osp.join(self.processed_dir, f'{rep}.pt'))

# 
def load_dataset(root=None, **kwargs):
    
    if root == None:
        raise ValueError('Root path is empty, specify root directory')

    # Group; name; operation/cutoff; Description
    pre_transform = [("KNN", "sqrt", "sqrt", "K nearest neighbour with sqrt for neighbours"),
                     ("KNN", "cbrt", "cbrt", "K nearest neighbour with sqrt for neighbours"),
                     ("DIST", "dist_3", 3, "Distance of 2angs"),
                     ("DIST", "dist_4", 4, "Distance of 2angs"),
                     ("DIST", "dist_6", 6, "Distance of 2angs"),
                     ("DIST", "dist_10", 10, "Distance of 2angs"),
                     ("DIST", "dist_12", 12, "Distance of 2angs")]
    # PDB URL has 1 attached to it
    dataset = PDBDataset(root, pre_transform=pre_transform, **kwargs)
    return dataset
