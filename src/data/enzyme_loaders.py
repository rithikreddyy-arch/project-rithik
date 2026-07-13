"""
enzyme_loaders.py
=================
Data loading for de novo enzyme design training.

Sources:
- M-CSA (Mechanism and Catalytic Site Atlas): curated catalytic residues
- RCSB PDB: structures
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
from urllib.request import urlopen
from urllib.error import URLError, HTTPError

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    _TORCH = True
except ImportError:
    _TORCH = False
    Dataset = object

from enzyme_design import AA3_TO_IDX, ParsedStructure, Residue, parse_pdb

log = logging.getLogger(__name__)

ATOM_TYPES = [
    "N", "CA", "C", "O", "CB", "CG", "CG1", "CG2", "CD", "CD1", "CD2",
    "CE", "CE1", "CE2", "CE3", "CZ", "CZ2", "CZ3", "CH2", "ND1", "ND2",
    "NE", "NE1", "NE2", "NH1", "NH2", "NZ", "OD1", "OD2", "OE1", "OE2",
    "OG", "OG1", "OH", "SD", "SG", "LIG_C", "LIG_N", "LIG_O", "UNK",
]
ATOM_TO_IDX = {a: i for i, a in enumerate(ATOM_TYPES)}

ROLES = [
    "unknown", "general_base", "general_acid", "nucleophile", "electrophile",
    "metal_ligand", "oxyanion_hole", "scaffold",
]
ROLE_TO_IDX = {r: i for i, r in enumerate(ROLES)}


class Cache:
    def __init__(self, root: str | Path = "./data_cache"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def fetch(self, url: str, name: str) -> bytes | None:
        dest = self.root / name
        if dest.exists():
            return dest.read_bytes()
        try:
            with urlopen(url, timeout=60) as r:
                data = r.read()
        except (URLError, HTTPError, TimeoutError) as e:
            log.warning("fetch failed %s: %s", url, e)
            return None
        dest.write_bytes(data)
        return data


@dataclass
class CatalyticSite:
    mcsa_id: int
    pdb_id: str
    ec: str | None
    residues: list[tuple[str, str, int, str]]
    ligands: list[str]

    @property
    def uid(self) -> str:
        return f"mcsa{self.mcsa_id}_{self.pdb_id}"


def chemistry_aware_split(
    sites: Sequence[CatalyticSite], val_frac: float = 0.15, test_frac: float = 0.15, seed: int = 0
) -> dict[str, list[CatalyticSite]]:
    """Split by EC-3 group to prevent homolog leakage."""
    rng = random.Random(seed)
    groups: dict[str, list[CatalyticSite]] = {}
    for s in sites:
        key = ".".join((s.ec or "0.0.0").split(".")[:3])
        groups.setdefault(key, []).append(s)
    keys = sorted(groups)
    rng.shuffle(keys)
    n = len(keys)
    n_val, n_test = int(n * val_frac), int(n * test_frac)
    out = {
        "test": [s for k in keys[:n_test] for s in groups[k]],
        "val": [s for k in keys[n_test : n_test + n_val] for s in groups[k]],
        "train": [s for k in keys[n_test + n_val :] for s in groups[k]],
    }
    log.info("split: %d train / %d val / %d test", len(out["train"]), len(out["val"]), len(out["test"]))
    return out


if _TORCH:
    @dataclass
    class Pocket:
        coords: np.ndarray
        atom_type: np.ndarray
        res_type: np.ndarray
        role: np.ndarray
        label: int
        decoy_kind: str = "native"
        contact_dists: np.ndarray | None = None
        source: str = ""

    class ActiveSiteDataset(Dataset):
        """Catalytic sites + hard decoys."""

        def __init__(self, sites: Sequence[CatalyticSite], cache: Cache, seed: int = 0):
            self.cache = cache
            self.sites = list(sites)
            self.seed = seed
            self._cache: dict[str, Pocket] = {}

        def __len__(self) -> int:
            return len(self.sites) * 2  # native + 1 decoy per site

        def __getitem__(self, idx: int) -> dict:
            site_i, is_decoy = divmod(idx, 2)
            site = self.sites[site_i]
            
            return {
                "coords": torch.randn(256, 3).float(),  # Placeholder
                "atom_type": torch.randint(0, 40, (256,)),
                "res_type": torch.randint(0, 21, (256,)),
                "role": torch.randint(0, 8, (256,)),
                "label": torch.tensor(0.0 if is_decoy else 1.0),
                "dist": torch.zeros(8),
                "decoy_kind": "identity_swap" if is_decoy else "native",
            }

    def collate_pockets(batch: list[dict]) -> dict:
        n = max(b["coords"].shape[0] for b in batch)
        B = len(batch)
        out = {
            "coords": torch.zeros(B, n, 3),
            "atom_type": torch.zeros(B, n, dtype=torch.long),
            "res_type": torch.zeros(B, n, dtype=torch.long),
            "role": torch.zeros(B, n, dtype=torch.long),
            "mask": torch.zeros(B, n, dtype=torch.bool),
            "label": torch.stack([b["label"] for b in batch]),
            "dist": torch.stack([b["dist"] for b in batch]),
            "decoy_kind": [b["decoy_kind"] for b in batch],
        }
        for i, b in enumerate(batch):
            m = b["coords"].shape[0]
            out["coords"][i, :m] = b["coords"]
            out["atom_type"][i, :m] = b["atom_type"]
            out["res_type"][i, :m] = b["res_type"]
            out["role"][i, :m] = b["role"]
            out["mask"][i, :m] = True
        return out
