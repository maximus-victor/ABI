from typing import Dict, List, Tuple
import numpy as np
from itertools import product

AAs: str = 'ALGVSREDTIPKFQNYMHWCUZO'


def get_aa_composition(protein_seq: str) -> Dict[str, int]:
    if set(protein_seq) == set(AAs):
        return {aa:protein_seq.count(aa) for aa in AAs}  
    else: raise ValueError("Wrong chars in protein_seq")


def k_mers(alphabet: str, k: int) -> List[str]:
    return sorted([''.join(x) for x in product(alphabet, repeat=k)])


def get_kmer_composition(protein_seq: str, k: int) -> Dict[str, int]:
    return {kmer:protein_seq.count(kmer) for kmer in k_mers(protein_seq, k)}


def get_alignment(protein_seq_1: str, protein_seq_2: str,
                  gap_penalty: int, substitution_matrix: np.ndarray) -> Tuple[str, str]:
    m = [[0 for _ in range(len(protein_seq_2) + 1)] for _ in range(len(protein_seq_1) + 1)]

    max_i = 0
    max_j = 0
    max_score = 0
    for i in range(1, len(protein_seq_1) + 1):
        for j in range(1, len(protein_seq_2) + 1):
            score = max(
                m[i-1][j-1] + substitution_matrix[protein_seq_1[i-1]][protein_seq_2[j-1]],
                m[i-1][j] + gap_penalty, 
                m[i][j-1] + gap_penalty, 
                0)
            if score > max_score:
                max_score = score
                max_i = i
                max_j = j
                
            m[i][j] = score
            
    i = max_i
    j = max_j

    aligned_seq1 = ""
    aligned_seq2 = ""
    
    while i > 0 and j > 0:
        if m[i-1][j-1] == m[i][j] - substitution_matrix[protein_seq_1[i-1]][protein_seq_2[j-1]]:
            i-=1
            j-=1
            aligned_seq1 = protein_seq_1[i] + aligned_seq1
            aligned_seq2 = protein_seq_2[j] + aligned_seq2
        elif m[i-1][j] == m[i][j] - gap_penalty:
            i-=1
            aligned_seq1 = protein_seq_1[i] + aligned_seq1
            aligned_seq2 = "-" + aligned_seq2
        else:            
            j-=1
            aligned_seq1 = "-" + aligned_seq1
            aligned_seq2 = protein_seq_2[j] + aligned_seq2

    return aligned_seq1, aligned_seq2
