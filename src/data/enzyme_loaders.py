"""
Data loading for de novo enzyme design.

Sources:
- KEGG Enzyme Database: reaction catalyzed, EC number, known structures
- PDB: solved enzyme structures with active site annotations
- EnzymeDB: curated enzyme sequences + EC classification
"""

import torch
import numpy as np
from typing import Tuple, List
from pathlib import Path
import requests
import json
from dataclasses import dataclass


@dataclass
class EnzymeData:
    """Single enzyme training example."""
    sequence: str  # Amino acid sequence
    structure_coords: np.ndarray  # (n_residues, 4, 3) backbone atoms
    ec_number: str  # E.C. classification (e.g., "1.1.1.1")
    active_site_residues: List[int]  # Indices of catalytic positions
    substrate_smiles: str  # What it catalyzes
    cofactor: str  # If any (NAD+, Mg2+, etc.)
    pdb_id: str  # Reference structure


def load_pdb_structure(pdb_id: str) -> np.ndarray:
    """
    Download PDB structure and extract backbone coordinates.
    
    Args:
        pdb_id: 4-character PDB ID (e.g., "1HBA")
    
    Returns:
        coords: (n_residues, 4, 3) array [N, CA, C, O atoms]
    """
    # Download from RCSB PDB
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    response = requests.get(url)
    
    if response.status_code != 200:
        raise ValueError(f"Could not download {pdb_id}")
    
    pdb_text = response.text
    coords = parse_pdb_backbone(pdb_text)
    return coords


def parse_pdb_backbone(pdb_text: str) -> np.ndarray:
    """
    Parse PDB file and extract backbone (N, CA, C, O) coordinates.
    
    Returns:
        (n_residues, 4, 3) array
    """
    atom_coords = {'N': [], 'CA': [], 'C': [], 'O': []}
    residue_id = None
    
    for line in pdb_text.split('\n'):
        if line.startswith('ATOM'):
            try:
                atom_name = line[12:16].strip()
                if atom_name not in atom_coords:
                    continue
                
                cur_res_id = int(line[22:26])
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                
                if residue_id is None:
                    residue_id = cur_res_id
                
                if cur_res_id != residue_id:
                    residue_id = cur_res_id
                
                atom_coords[atom_name].append([x, y, z])
            except (ValueError, IndexError):
                continue
    
    # Stack into (n_residues, 4, 3)
    n_residues = len(atom_coords['CA'])
    coords = np.zeros((n_residues, 4, 3))
    for i, atom in enumerate(['N', 'CA', 'C', 'O']):
        coords[:, i, :] = np.array(atom_coords[atom][:n_residues])
    
    return coords


def load_ec_database(ec_file: str = "data/raw/ec_database.json") -> List[EnzymeData]:
    """
    Load enzyme data from KEGG EC database.
    
    Format: JSON with enzyme entries
    """
    if not Path(ec_file).exists():
        print(f"Downloading EC database...")
        download_ec_database(ec_file)
    
    enzymes = []
    with open(ec_file, 'r') as f:
        data = json.load(f)
    
    for entry in data:
        try:
            pdb_id = entry.get('pdb_id')
            if not pdb_id:
                continue
            
            coords = load_pdb_structure(pdb_id)
            
            enzyme = EnzymeData(
                sequence=entry['sequence'],
                structure_coords=coords,
                ec_number=entry['ec_number'],
                active_site_residues=entry['active_site'],
                substrate_smiles=entry.get('substrate', ''),
                cofactor=entry.get('cofactor', 'None'),
                pdb_id=pdb_id,
            )
            enzymes.append(enzyme)
        except Exception as e:
            print(f"Skipping {entry.get('pdb_id')}: {e}")
            continue
    
    return enzymes


def download_ec_database(output_file: str):
    """
    Download EC database from KEGG.
    
    In real implementation, use KEGG REST API.
    For now, use local copy or smaller subset.
    """
    # Placeholder: in real code, fetch from KEGG
    sample_data = [
        {
            "pdb_id": "1HBA",
            "sequence": "MVLSPADKTNVIRAAQNCYSTEIN" + "A" * 100,
            "ec_number": "1.1.1.1",
            "active_site": [10, 15, 25],
            "substrate": "CCO",  # Ethanol
            "cofactor": "NAD+",
        },
    ]
    
    with open(output_file, 'w') as f:
        json.dump(sample_data, f)


class EnzymeDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for enzyme structures + sequences.
    """
    
    def __init__(self, enzymes: List[EnzymeData], max_length: int = 512):
        self.enzymes = enzymes
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.enzymes)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        enzyme = self.enzymes[idx]
        
        # Sequence encoding (0-19 for amino acids)
        from models.enzyme_design import AA_VOCAB
        seq_tokens = torch.tensor([AA_VOCAB[aa] for aa in enzyme.sequence], dtype=torch.long)
        
        # Pad or truncate
        if len(seq_tokens) < self.max_length:
            seq_tokens = torch.nn.functional.pad(seq_tokens, (0, self.max_length - len(seq_tokens)))
        else:
            seq_tokens = seq_tokens[:self.max_length]
        
        # Coordinates (normalized)
        coords = torch.from_numpy(enzyme.structure_coords).float()
        if coords.shape[0] < self.max_length:
            pad = (self.max_length - coords.shape[0], 0, 0, 0, 0, 0)
            coords = torch.nn.functional.pad(coords, pad)
        else:
            coords = coords[:self.max_length]
        
        # Active site mask
        as_mask = torch.zeros(self.max_length, dtype=torch.bool)
        for idx in enzyme.active_site_residues:
            if idx < self.max_length:
                as_mask[idx] = True
        
        return seq_tokens, coords, as_mask


def create_data_splits(
    enzymes: List[EnzymeData],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Tuple[List[EnzymeData], List[EnzymeData], List[EnzymeData]]:
    """
    Split into train/val/test.
    """
    n = len(enzymes)
    indices = np.random.permutation(n)
    
    train_idx = int(n * train_ratio)
    val_idx = int(n * (train_ratio + val_ratio))
    
    train = [enzymes[i] for i in indices[:train_idx]]
    val = [enzymes[i] for i in indices[train_idx:val_idx]]
    test = [enzymes[i] for i in indices[val_idx:]]
    
    return train, val, test
