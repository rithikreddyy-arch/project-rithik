"""
enzyme_design.py
================
Core abstractions and models for de novo enzyme design.

The pipeline:
    theozyme (QM transition-state model + catalytic residues)
        -> backbone scaffolding      (RFdiffusion / RFdiffusion2, all-atom)
        -> sequence design           (LigandMPNN, catalytic residues fixed)
        -> structure prediction      (ESMFold / AF2)
        -> geometric + learned filtering
        -> ranked designs for synthesis
"""

from __future__ import annotations

import json
import logging
import numpy as np
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable, Sequence

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH = True
except ImportError:
    _TORCH = False
    nn = object  # type: ignore

log = logging.getLogger(__name__)

AA3 = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
]
AA3_TO_IDX = {a: i for i, a in enumerate(AA3)}
AA3_TO_AA1 = dict(zip(AA3, "ARNDCQEGHILKMFPSTWYV"))
AA1_TO_AA3 = {v: k for k, v in AA3_TO_AA1.items()}


@dataclass
class CatalyticResidue:
    """One member of a theozyme."""
    resname: str
    role: str
    coords: np.ndarray
    atom_names: list[str]
    contacts: list[tuple[str, str, float, float]] = field(default_factory=list)
    chain_hint: str | None = None

    def __post_init__(self) -> None:
        self.resname = self.resname.upper()
        if self.resname not in AA3_TO_IDX:
            raise ValueError(f"Unknown residue {self.resname!r}")
        self.coords = np.asarray(self.coords, dtype=np.float64).reshape(-1, 3)
        if len(self.atom_names) != len(self.coords):
            raise ValueError(
                f"{self.resname}: {len(self.atom_names)} atom names vs {len(self.coords)} coordinates"
            )

    def atom(self, name: str) -> np.ndarray:
        return self.coords[self.atom_names.index(name)]


@dataclass
class Ligand:
    """Transition-state analogue / substrate."""
    name: str
    coords: np.ndarray
    atom_names: list[str]
    elements: list[str]
    smiles: str | None = None
    is_transition_state: bool = True

    def __post_init__(self) -> None:
        self.coords = np.asarray(self.coords, dtype=np.float64).reshape(-1, 3)

    def atom(self, name: str) -> np.ndarray:
        return self.coords[self.atom_names.index(name)]


@dataclass
class Theozyme:
    """A quantum-mechanically idealised catalytic arrangement."""
    name: str
    residues: list[CatalyticResidue]
    ligand: Ligand
    reaction_ec: str | None = None
    reaction_smarts: str | None = None
    notes: str = ""

    def to_json(self, path: str | Path) -> None:
        payload = {
            "name": self.name,
            "reaction_ec": self.reaction_ec,
            "reaction_smarts": self.reaction_smarts,
            "notes": self.notes,
            "ligand": {
                **{k: v for k, v in asdict(self.ligand).items() if k != "coords"},
                "coords": self.ligand.coords.tolist(),
            },
            "residues": [
                {
                    **{k: v for k, v in asdict(r).items() if k != "coords"},
                    "coords": r.coords.tolist(),
                }
                for r in self.residues
            ],
        }
        Path(path).write_text(json.dumps(payload, indent=2))

    @classmethod
    def from_json(cls, path: str | Path) -> "Theozyme":
        d = json.loads(Path(path).read_text())
        lig = Ligand(**{**d["ligand"], "coords": np.array(d["ligand"]["coords"])})
        res = [
            CatalyticResidue(
                **{
                    **r,
                    "coords": np.array(r["coords"]),
                    "contacts": [tuple(c) for c in r.get("contacts", [])],
                }
            )
            for r in d["residues"]
        ]
        return cls(
            name=d["name"],
            residues=res,
            ligand=lig,
            reaction_ec=d.get("reaction_ec"),
            reaction_smarts=d.get("reaction_smarts"),
            notes=d.get("notes", ""),
        )

    def constraint_table(self) -> list[dict]:
        out = []
        for i, r in enumerate(self.residues):
            for res_atom, lig_atom, target, tol in r.contacts:
                out.append({
                    "res_index": i,
                    "resname": r.resname,
                    "role": r.role,
                    "res_atom": res_atom,
                    "lig_atom": lig_atom,
                    "target": float(target),
                    "tol": float(tol),
                })
        return out

    def validate(self) -> list[str]:
        problems: list[str] = []
        if not self.residues:
            problems.append("theozyme has no catalytic residues")
        if not self.constraint_table():
            problems.append("no catalytic contacts defined")
        for c in self.constraint_table():
            r = self.residues[c["res_index"]]
            try:
                d = float(np.linalg.norm(r.atom(c["res_atom"]) - self.ligand.atom(c["lig_atom"])))
                if abs(d - c["target"]) > c["tol"]:
                    problems.append(
                        f"constraint {r.resname}.{c['res_atom']}-{c['lig_atom']}: "
                        f"{d:.2f} A vs {c['target']:.2f}+/-{c['tol']:.2f}"
                    )
            except (KeyError, ValueError) as exc:
                problems.append(f"constraint references missing atom: {exc}")
        return problems


@dataclass
class DesignCandidate:
    design_id: str
    backbone_pdb: Path | None = None
    sequence: str | None = None
    predicted_pdb: Path | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    stage: str = "init"
    failed: str | None = None

    def record(self, **kw: float) -> None:
        self.metrics.update({k: float(v) for k, v in kw.items()})


def kabsch(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    P, Q = np.asarray(P, float), np.asarray(Q, float)
    pc, qc = P.mean(0), Q.mean(0)
    H = (P - pc).T @ (Q - qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    return R, qc - R @ pc


def rmsd(P: np.ndarray, Q: np.ndarray, superpose: bool = True) -> float:
    P, Q = np.asarray(P, float), np.asarray(Q, float)
    if P.shape != Q.shape:
        raise ValueError(f"shape mismatch {P.shape} vs {Q.shape}")
    if superpose:
        R, t = kabsch(P, Q)
        P = (R @ P.T).T + t
    return float(np.sqrt(((P - Q) ** 2).sum(-1).mean()))


@dataclass
class Residue:
    resname: str
    resseq: int
    chain: str
    atoms: dict[str, np.ndarray]
    bfactor: float = 0.0

    @property
    def ca(self) -> np.ndarray | None:
        return self.atoms.get("CA")


@dataclass
class ParsedStructure:
    residues: list[Residue]
    hetatms: list[tuple[str, str, np.ndarray]] = field(default_factory=list)
    path: Path | None = None

    @property
    def sequence(self) -> str:
        return "".join(AA3_TO_AA1.get(r.resname, "X") for r in self.residues)

    @property
    def ca_coords(self) -> np.ndarray:
        return np.array([r.ca for r in self.residues if r.ca is not None])

    def mean_plddt(self) -> float:
        vals = [r.bfactor for r in self.residues]
        return float(np.mean(vals)) if vals else 0.0


def parse_pdb(path: str | Path, model: int = 1) -> ParsedStructure:
    residues: dict[tuple[str, int], Residue] = {}
    for line in Path(path).read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")):
            name = line[12:16].strip()
            resname = line[17:20].strip()
            chain = line[21].strip() or "A"
            try:
                resseq = int(line[22:26])
                xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            except ValueError:
                continue
            if resname not in AA3_TO_IDX:
                continue
            try:
                b = float(line[60:66])
            except ValueError:
                b = 0.0
            key = (chain, resseq)
            if key not in residues:
                residues[key] = Residue(resname, resseq, chain, {}, b)
            residues[key].atoms[name] = xyz
    ordered = [residues[k] for k in sorted(residues, key=lambda k: (k[0], k[1]))]
    return ParsedStructure(ordered, [], Path(path))


if _TORCH:
    class EGNNLayer(nn.Module):
        def __init__(self, dim: int, edge_dim: int = 0, hidden: int | None = None):
            super().__init__()
            hidden = hidden or dim
            self.edge_mlp = nn.Sequential(
                nn.Linear(2 * dim + 1 + edge_dim, hidden),
                nn.SiLU(),
                nn.Linear(hidden, hidden),
                nn.SiLU(),
            )
            self.node_mlp = nn.Sequential(
                nn.Linear(dim + hidden, hidden), nn.SiLU(), nn.Linear(hidden, dim)
            )
            self.att = nn.Sequential(nn.Linear(hidden, 1), nn.Sigmoid())

        def forward(self, h: torch.Tensor, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            B, N, _ = h.shape
            d2 = ((x[:, :, None] - x[:, None, :]) ** 2).sum(-1, keepdim=True)
            hi = h[:, :, None].expand(B, N, N, h.size(-1))
            hj = h[:, None, :].expand(B, N, N, h.size(-1))
            m = self.edge_mlp(torch.cat([hi, hj, d2], dim=-1))
            m = m * self.att(m)
            pair_mask = (mask[:, :, None] & mask[:, None, :]).unsqueeze(-1).float()
            eye = torch.eye(N, device=h.device, dtype=torch.bool)[None, :, :, None]
            m = m * pair_mask * (~eye).float()
            agg = m.sum(dim=2)
            h = h + self.node_mlp(torch.cat([h, agg], dim=-1))
            return h * mask.unsqueeze(-1)

    class CatalyticGeometryNet(nn.Module):
        def __init__(
            self,
            n_atom_types: int = 40,
            n_res_types: int = 21,
            n_roles: int = 8,
            dim: int = 128,
            layers: int = 4,
            n_contacts: int = 8,
        ):
            super().__init__()
            self.atom_emb = nn.Embedding(n_atom_types, dim // 2)
            self.res_emb = nn.Embedding(n_res_types, dim // 4)
            self.role_emb = nn.Embedding(n_roles, dim // 4)
            self.layers = nn.ModuleList([EGNNLayer(dim) for _ in range(layers)])
            self.norm = nn.LayerNorm(dim)
            self.cls = nn.Sequential(
                nn.Linear(dim, dim), nn.SiLU(), nn.Dropout(0.1), nn.Linear(dim, 1)
            )
            self.dist = nn.Sequential(
                nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, n_contacts)
            )

        def forward(self, batch: dict) -> dict:
            h = torch.cat(
                [
                    self.atom_emb(batch["atom_type"]),
                    self.res_emb(batch["res_type"]),
                    self.role_emb(batch["role"]),
                ],
                dim=-1,
            )
            x, mask = batch["coords"], batch["mask"]
            for layer in self.layers:
                h = layer(h, x, mask)
            h = self.norm(h)
            denom = mask.sum(1, keepdim=True).clamp(min=1)
            pooled = (h * mask.unsqueeze(-1)).sum(1) / denom
            return {
                "logit": self.cls(pooled).squeeze(-1),
                "dist": self.dist(pooled),
            }

        def score(self, batch: dict) -> torch.Tensor:
            return torch.sigmoid(self.forward(batch)["logit"])


if _TORCH:
    __all__ = [
        "CatalyticResidue", "Ligand", "Theozyme", "DesignCandidate",
        "Residue", "ParsedStructure", "parse_pdb",
        "kabsch", "rmsd",
        "EGNNLayer", "CatalyticGeometryNet",
    ]
else:
    __all__ = [
        "CatalyticResidue", "Ligand", "Theozyme", "DesignCandidate",
        "Residue", "ParsedStructure", "parse_pdb",
        "kabsch", "rmsd",
    ]
