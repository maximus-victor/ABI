from torch.utils.data import DataLoader, Dataset
import torch
from transformers import BertTokenizer, BertModel
from pathlib import Path
from pyfaidx import Fasta
from typing import Dict, Tuple, List
import random


class TokenSeqDataset(Dataset):
    def __init__(self, fasta_path: Path, labels_dict: Dict[str, int], max_num_residues: int, protbert_cache: Path, device: torch.device):
        self.device = device
        self.max_num_residues = max_num_residues
        self.labels = labels_dict
        self.data = self.parse_fasta_input(fasta_path)
        self.protbert_cache = protbert_cache
        self.protbert = self.load_model()
        self.tokenizer = self.load_tokenizer()


    def load_tokenizer(self) -> BertTokenizer:
        return BertTokenizer.from_pretrained('Rostlab/prot_bert_bfd', do_lower_case=False, cache_dir=self.protbert_cache)

    def load_model(self) -> BertModel:
        model = BertModel.from_pretrained('Rostlab/prot_bert_bfd', cache_dir=self.protbert_cache)
        model = model.to(self.device)
        return model

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        key = self.keys[index]
        label = self.labels[key]
        seq = self.data[key][:self.max_num_residues]
        tokens = self.tokenize(seq)
        embedding = self.embedd(*tokens)

        return (embedding, label)

    def __len__(self) -> int:
        return len(self.data)

    def tokenize(self, seq: str) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = [" ".join(seq)]
        tokenized = self.tokenizer(text=seq, padding='max_length', max_length=self.max_num_residues+2, add_special_tokens=True, return_tensors='pt') #+2 seems a bit random but works

        return tokenized['input_ids'], tokenized['attention_mask']

    def embedd(self, tokens: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            embedding = self.protbert.forward(input_ids=tokens, attention_mask=attention_mask)[0]

        embedding = torch.mean(embedding, 1)
        embedding = torch.squeeze(embedding)

        return embedding


    def parse_fasta_input(self, input_file: Path) -> Dict[str, str]:
        fasta = Fasta(str(input_file))
        self.data = {key:str(fasta[key]) for key in fasta.keys()}
        self.keys = list(fasta.keys())
        return self.data


def collate_paired_sequences(data: List[Tuple[torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.tensor([x.tolist() for x in list(zip(*data))[0]]),
        torch.tensor([[x] for x in list(zip(*data))[1]]).float()
    )


def get_dataloader(fasta_path: Path, labels_dict: Dict[str, int], batch_size: int, max_num_residues: int,
                   protbert_cache: Path, device: torch.device, seed: int) -> DataLoader:
    random.seed(seed)
    dataset = TokenSeqDataset(
        fasta_path=fasta_path, 
        labels_dict=labels_dict, 
        max_num_residues=max_num_residues, 
        protbert_cache=protbert_cache, 
        device=device)
    
    return DataLoader(dataset=dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_paired_sequences)


if __name__ == '__main__':
    pass