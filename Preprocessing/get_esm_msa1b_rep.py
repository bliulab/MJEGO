import torch
import sys
import esm
import string
import numpy as np
from Bio import SeqIO
from typing import List, Tuple, Optional, Dict, NamedTuple, Union, Callable
import copy
import torch.nn.functional as F
import math
# from colabfold.batch import run_mmseqs2
from scipy.spatial.distance import squareform, pdist, cdist

def save_model_as_pt(pt_path='/data0/wzw/ProteinPred/Models/msa_transformer/pt/pretrained.pt', device='cuda'):
    # 加载预训练模型
    msa_transformer, msa_transformer_alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
    
    # 保存模型为 .pt 文件
    torch.save(msa_transformer.state_dict(), pt_path)

def load_model_from_pt(pt_path='/data0/wzw/ProteinPred/Models/msa_transformer/pt/pretrained.pt', device='cuda'):
    # 加载模型定义
    msa_transformer, msa_transformer_alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
    
    # 加载 .pt 文件中的权重
    msa_transformer.load_state_dict(torch.load(pt_path, map_location=device))
    
    msa_transformer = msa_transformer.eval().to(device)
    return msa_transformer, msa_transformer_alphabet

# def compute_msa(sequence, output_dir="msa_output"):
#     """使用 ColabFold 计算 MSA，并返回 .a3m 文件路径"""
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
    
#     result = run_mmseqs2(sequence, output_dir, use_env=True)
#     return f"{output_dir}/query.a3m"

def pad_msa_with_self(sequence, num_copies=10):
    """如果 MSA 为空，则复制原始序列 num_copies 次，形成伪 MSA,并添加标签"""
    return [(f"seq_{i}", sequence) for i in range(num_copies)]

def get_esm_msa1b_rep(a3m_path='.a3m', num_seqs=256, device='cuda', delete_first_line=False, pt_path='/data0/wzw/ProteinPred/Models/msa_transformer/pt/pretrained.pt'):
    processed_alignment, position_converter, unprocessed_refseq = load_alignment(a3m_path)
    # assert len(processed_alignment) > 1, "Expected alignment, but received fasta"
    if len(processed_alignment) <= 1:
        print("MSA 为空，使用复制填充...")
        processed_alignment = pad_msa_with_self(unprocessed_refseq, num_copies=num_seqs)
    else:
        # 确保 processed_alignment 是 (label, sequence) 形式
        processed_alignment = [(f"seq_{i}", seq) if isinstance(seq, str) else seq for i, seq in enumerate(processed_alignment)]
    
    # 使用从 .pt 文件加载的模型
    msa_transformer, msa_transformer_alphabet = load_model_from_pt(pt_path, device)
    
    msa_transformer_batch_converter = msa_transformer_alphabet.get_batch_converter()
    
    inputs = greedy_select(processed_alignment, num_seqs=num_seqs) 
    msa_transformer_batch_labels, msa_transformer_batch_strs, msa_transformer_batch_tokens = msa_transformer_batch_converter([inputs])
    msa_transformer_batch_tokens = msa_transformer_batch_tokens.to(next(msa_transformer.parameters()).device)
    if delete_first_line:
        msa_transformer_batch_tokens = msa_transformer_batch_tokens[:, 1:, :]
    with torch.no_grad():
        results = msa_transformer(msa_transformer_batch_tokens, repr_layers=[12])
        all_temp_reprs = results["representations"][12][:, 0][:, 1:, :]
    
    seqlen = len(unprocessed_refseq)
    out_rep = torch.zeros(1, seqlen, 768)
    
    for key, value in position_converter.items():
        out_rep[:, key, :] = all_temp_reprs[:, value, :]
    
    return out_rep.to(device)

def greedy_select(msa: List[Tuple[str, str]], num_seqs: int, mode: str = "max") -> List[Tuple[str, str]]:
    assert mode in ("max", "min")
    if len(msa) <= num_seqs:
        return msa
    
    array = np.array([list(seq) for _, seq in msa], dtype=np.bytes_).view(np.uint8)

    optfunc = np.argmax if mode == "max" else np.argmin
    all_indices = np.arange(len(msa))
    indices = [0]
    pairwise_distances = np.zeros((0, len(msa)))
    for _ in range(num_seqs - 1):
        dist = cdist(array[indices[-1:]], array, "hamming")
        pairwise_distances = np.concatenate([pairwise_distances, dist])
        shifted_distance = np.delete(pairwise_distances, indices, axis=1).mean(0)
        shifted_index = optfunc(shifted_distance)
        index = np.delete(all_indices, indices)[shifted_index]
        indices.append(index)
    indices = sorted(indices)
    return [msa[idx] for idx in indices]

def load_alignment(input_filename):
    """
    Given the path to an alignment file, loads the alignment, then processes it
    to remove unaligned columns. The processed alignment is then ready to be 
    passed to the tokenization function of the MsaTransformer.
    
    Parameters
    ----------
    input_filename: str: Path to the alignment. 
    
    Returns
    -------
    processed_alignment: list of lists: Contents of an a3m alignment file
        with all unaligned columns removed. This is formatted for passage into
        the tokenization function of the MsaTransformer.
    old_to_new_pos: dict: A dictionary that relates the old index in the reference
        sequence to the new position in the processed reference sequence.
    """
    # Set up deletekeys to delete all lowercase letters and '*'
    deletekeys = dict.fromkeys(string.ascii_lowercase)
    deletekeys["*"] = None
    
    # Load the unprocessed alignment
    unprocessed_alignment = [(record.description, str(record.seq))
                             for record in SeqIO.parse(input_filename, "fasta")]

    # Save the original reference sequence
    unprocessed_refseq = unprocessed_alignment[0][1]

    # Get a dictionary linking old position to processed position
    position_converter = build_old_to_new(unprocessed_refseq, deletekeys)

    # Process the alignment (removing lowercase letters and '*')
    processed_alignment = process_alignment(unprocessed_alignment, deletekeys)
    
    return processed_alignment, position_converter, unprocessed_refseq


def build_old_to_new(unprocessed_refseq, deletekeys):
    """
    Build a mapping from old positions to new positions after removing unwanted characters.
    """
    n_capital_letters = sum((char.isalpha() and char.isupper()) 
                            for char in unprocessed_refseq)

    seq_ind = -1
    processed_ind = -1
    old_to_new_pos = {}
    for char in unprocessed_refseq:
        
        alpha_check = char.isalpha()
        delete_check = (char not in deletekeys)
        
        if alpha_check:
            seq_ind += 1
            
        if delete_check:
            processed_ind += 1
            
            if not alpha_check:
                assert char == "-", "Unexpected character in reference sequence"
        
        if alpha_check and delete_check:
            old_to_new_pos[seq_ind] = processed_ind 
                
    assert len(old_to_new_pos) == n_capital_letters
    return old_to_new_pos


def process_alignment(unprocessed_alignment, deletekeys):
    """
    Process the MSA by removing unaligned columns and duplicate sequences.
    """
    translation = str.maketrans(deletekeys)
    
    processed_alignment = []
    observed_seqs = []
    for desc, seq in unprocessed_alignment:
        processed_seq = seq.translate(translation)
        if processed_seq not in observed_seqs:
            observed_seqs.append(processed_seq)
            processed_alignment.append((desc, processed_seq))
            
    testlen = len(processed_alignment[0][1])
    assert all(len(seq) == testlen for _, seq in processed_alignment)
    
    return processed_alignment
