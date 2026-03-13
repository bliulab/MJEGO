import sys
sys.path.append("/data0/wzw/PFP-Pred-Proj")
import itertools
import torch.nn as nn
import torch
from torch.nn import Sigmoid
from Models.egnn_clean import egnn_clean as eg
import Utils.net_utils as net_utils
from Preprocessing.go_emb import *
import yaml
from Utils.utils import load_configs, load_checkpoints_only
from Models.s_plm.model import SequenceRepresentation

"""
    SPLM & GO & MSA
"""

class CrossModalAttention(nn.Module):
    # mf 1350 cc 1229 bp 8491
    def __init__(self, protein_dim=1350, go_dim=200, hidden_dim=512):
        super().__init__()
        self.proj_p = nn.Linear(protein_dim, hidden_dim)
        self.proj_g = nn.Linear(go_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=4)

    def forward(self, proteins, gos):
        P = self.proj_p(proteins)  
        G = self.proj_g(gos)       

        attn_output, _ = self.attn(P.unsqueeze(1), G.unsqueeze(1), G.unsqueeze(1))
        fused_feature = P + attn_output.squeeze(1)
        return fused_feature

class GCN(torch.nn.Module):
    def __init__(self, **kwargs):
        super(GCN, self).__init__()

        input_features_size = kwargs['input_features_size']
        hidden_channels = kwargs['hidden']
        edge_features = kwargs['edge_features']
        num_classes = kwargs['num_classes']
        num_egnn_layers = kwargs['egnn_layers']

        self.edge_type = kwargs['edge_type']
        self.num_layers = kwargs['layers']
        self.device = kwargs['device']

        self.nc = num_classes
        self.esm_msa_linear = nn.Linear(768, 768)

        self.egnn_1 = eg.EGNN(in_node_nf=input_features_size,
                              hidden_nf=hidden_channels,
                              n_layers=num_egnn_layers,
                              out_node_nf=num_classes,
                              in_edge_nf=edge_features,
                              attention=True,
                              normalize=False,
                              tanh=True)

        self.egnn_2 = eg.EGNN(in_node_nf=num_classes,
                              hidden_nf=hidden_channels,
                              n_layers=num_egnn_layers,
                              out_node_nf=int(num_classes / 2),
                              in_edge_nf=edge_features,
                              attention=True,
                              normalize=False,
                              tanh=True)

        self.egnn_3 = eg.EGNN(in_node_nf=input_features_size,
                              hidden_nf=hidden_channels,
                              n_layers=num_egnn_layers,
                              out_node_nf=int(num_classes / 2),
                              in_edge_nf=edge_features,
                              attention=True,
                              normalize=False,
                              tanh=True)

        self.egnn_4 = eg.EGNN(in_node_nf=int(num_classes / 2),
                              hidden_nf=hidden_channels,
                              n_layers=num_egnn_layers,
                              out_node_nf=int(num_classes / 4),
                              in_edge_nf=edge_features,
                              attention=True,
                              normalize=False,
                              tanh=True)

        self.fc1 = net_utils.FC(num_classes + int(num_classes / 2) * 2 + int(num_classes / 4),
                                num_classes + 50, relu=False, bnorm=True)
        self.fc2 = net_utils.FC(num_classes + 50, num_classes, relu=False, bnorm=True)

        self.fc3 = net_utils.FC(num_classes + 512 + 768, num_classes, relu=False, bnorm=False)

        self.bnrelu1 = net_utils.BNormRelu(num_classes)
        self.bnrelu2 = net_utils.BNormRelu(int(num_classes / 2))
        self.bnrelu3 = net_utils.BNormRelu(int(num_classes / 4))
        self.sig = Sigmoid()

        # mf 1350 cc 1229 bp 8491
        self.cross_modal_attn = CrossModalAttention(protein_dim=1350, go_dim=128, hidden_dim=512)

    def get_seqembed(self, sequences):
    
        sequences = sequences
        # Load the configuration file
        config_path = "/data0/wzw/S-PLM/configs/representation_config.yaml"
        with open(config_path) as file:
                dict_config = yaml.full_load(file)
        configs = load_configs(dict_config)

        # Create the model using the configuration file
        model = SequenceRepresentation(logging=None, configs=configs).to(self.device)
        model.eval()  # Ensure the model is in eval mode

        # Load the checkpoint
        checkpoint_path = "/data0/wzw/S-PLM/model/checkpoint_0520000.pth"
        load_checkpoints_only(checkpoint_path, model)

        all_residue_embeddings = []
        all_protein_embeddings = []

        for i, seq in enumerate(sequences):
                # Prepare single sequence for batch_converter
                esm2_seq = [(0, str(seq))]
                batch_labels, batch_strs, batch_tokens = model.batch_converter(esm2_seq)

                # Get representations
                with torch.no_grad():
                        protein_representation, residue_representation, mask = model(batch_tokens.to(self.device))

                # Remove [CLS] and [EOS]
                residue_representation_seq = residue_representation[:, 1:-1, :].squeeze(0)

                all_residue_embeddings.append(residue_representation_seq)
                all_protein_embeddings.append(protein_representation.squeeze(0))

        # Concatenate all residue embeddings
        total_residue_embedding = torch.cat(all_residue_embeddings, dim=0)
        # Stack all protein embeddings (optional, in case you want all)
        all_protein_embeddings = torch.stack(all_protein_embeddings, dim=0)

        return all_protein_embeddings, total_residue_embedding

    def get_msa_batch_representation(self, data):
        msa_batch = data['atoms'].batch
        msa_rep = data['atoms'].msa
        pooled_msa = net_utils.get_pool(pool_type='mean')(msa_rep, msa_batch)  # -> (batch_size, 768)
        msa_out = self.esm_msa_linear(pooled_msa)  # -> (batch_size, out dim)
        return msa_out

    def forward_once(self, data):
        x_res, x_emb_seq, edge_index, edge_attr, x_batch, x_pos, x_seq = data['atoms'].embedding_features_per_residue, \
            data['atoms'].embedding_features_per_sequence, \
            data[self.edge_type].edge_index, \
            data[self.edge_type].edge_attr, \
            data['atoms'].batch, \
            data['atoms'].pos, \
            data['atoms'].sequence_letters
        
        protein_representation, residue_representation_seq = self.get_seqembed(x_seq)
        

        # print("protein_representation shape:", protein_representation.shape)
        # print("residue_representation_seq shape:", residue_representation_seq.shape)
        # print("x_res shape:", x_res.shape)
        # print("x_emb_seq shape:", x_emb_seq.shape)
        # 
        ppi_shape = x_emb_seq.shape[0]

        if ppi_shape > 1:
            edge_index_2 = list(zip(*list(itertools.combinations(range(ppi_shape), 2))))
            edge_index_2 = [torch.LongTensor(edge_index_2[0]).to(self.device),
                            torch.LongTensor(edge_index_2[1]).to(self.device)]
        else:
            edge_index_2 = tuple(range(ppi_shape))
            edge_index_2 = [torch.LongTensor(edge_index_2).to(self.device),
                            torch.LongTensor(edge_index_2).to(self.device)]

        output_res, pre_pos_res = self.egnn_1(h=residue_representation_seq,
                                              x=x_pos.float(),
                                              edges=edge_index,
                                              edge_attr=None)

        output_res_2, pre_pos_res_2 = self.egnn_2(h=output_res,
                                                  x=pre_pos_res.float(),
                                                  edges=edge_index,
                                                  edge_attr=None)

        output_seq, pre_pos_seq = self.egnn_3(h=protein_representation,
                                              x=net_utils.get_pool(pool_type='mean')(x_pos.float(), x_batch),
                                              edges=edge_index_2,
                                              edge_attr=None)

        output_res_4, pre_pos_seq_4 = self.egnn_4(h=output_res_2,
                                                  x=pre_pos_res_2.float(),
                                                  edges=edge_index,
                                                  edge_attr=None)

        output_res = net_utils.get_pool(pool_type='mean')(output_res, x_batch)
        output_res = self.bnrelu1(output_res)

        output_res_2 = net_utils.get_pool(pool_type='mean')(output_res_2, x_batch)
        output_res_2 = self.bnrelu2(output_res_2)

        output_seq = self.bnrelu2(output_seq)

        output_res_4 = net_utils.get_pool(pool_type='mean')(output_res_4, x_batch)
        output_res_4 = self.bnrelu3(output_res_4)

        output = torch.cat([output_res, output_seq, output_res_2, output_res_4], 1)

        return output
    
    
    def conb(self, data):
        output1 = self.forward_once(data)
        # linear = nn.Linear(output1.shape[1], self.nc).to(self.device)
        # protein_features = linear(output1)
        # print("output1:", output1.shape)

        output2 = go_emb('molecular_function', 128).to(self.device)
        # print("output2:", output2.shape)
        # go_features = torch.matmul(protein_features, output2)

        return output1, output2



    def forward(self, data):
        passes = []

        msa_out =  self.get_msa_batch_representation(data)

        for i in range(self.num_layers):
            passes.append(self.forward_once(data))

        x = torch.cat(passes, 1)

        x = self.fc1(x)
        x = self.fc2(x)

        protein_features, go_features = self.conb(data)
        fused_features = self.cross_modal_attn(protein_features, go_features)

        x = torch.cat([x, fused_features], 1)

        # cat msa emb
        x = torch.cat([x, msa_out], 1)

        x = self.fc3(x)
        x = self.sig(x)

        return x

