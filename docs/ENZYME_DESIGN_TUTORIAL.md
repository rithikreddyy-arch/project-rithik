# De Novo Enzyme Design: A Working Pipeline and an Open Problem

This repository implements the standard de novo enzyme design pipeline —
theozyme → all-atom diffusion → ligand-aware inverse folding → structure
prediction → filtering — and adds one thing the standard pipeline is missing:
a learned filter for **catalytic geometry**, trained on curated natural active
sites against decoys that mimic real design failure modes.

It is written to be read by people who will immediately ask "does this actually
work?" So let me answer that first.

---

## 1. What works, what doesn't, and where the gap is

De novo enzyme design is not a solved problem. It is barely a working one.

The generative stack is genuinely strong now. RFdiffusionAA can scaffold an
all-atom catalytic motif around a small molecule; LigandMPNN designs sequences
that see the substrate; AF2/ESMFold tells you whether the sequence folds to what
you drew. Binder design has been transformed by this stack, and enzyme design
has inherited the machinery.

What has *not* transformed is the hit rate for **catalysis**. Designed backbones
fold. Designed pockets bind. Designed enzymes mostly do not turn over, and when
they do, they turn over orders of magnitude below their natural counterparts and
usually require directed evolution to become useful. The field's own published
campaigns still report low single-digit percentages of designs with detectable
activity.

**Why?** The dominant filters are `pLDDT`, backbone RMSD, and motif RMSD. These
answer "did it fold like I asked?" They do not answer "is the transition state
actually stabilised?" A design can have 0.6 Å motif RMSD — passing every filter
in the standard stack — while the general base sits 1.4 Å from where it must sit
to abstract the proton. Sub-angstrom errors are catalytically fatal and
RMSD-invisible, because RMSD averages over atoms that don't matter alongside the
two or three that do.

**The proposition of this repo:** train a model to score catalytic competence
directly, using the one dataset where catalytic geometry is curated with
mechanistic roles (M-CSA), against decoys constructed to be exactly the
near-misses that current filters wave through. Then ask, on held-out natural
enzymes, whether it separates real active sites from geometric near-misses
better than RMSD does.

That question is falsifiable, cheap to answer, and answerable before you spend a
cent on gene synthesis. `train_enzyme.py --eval-baseline` is the null model.
**If the learned scorer does not beat it, the honest thing is to report that.**

---

## 2. Installation

```bash
git clone https://github.com/<you>/project-rithik.git
cd project-rithik
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

External tools are **not** vendored. Install and point the config at them:

| Tool | Why | Repo |
|---|---|---|
| RFdiffusionAA (or RFdiffusion2) | all-atom motif scaffolding *with the ligand present* | `baker-laboratory/rf_diffusion_all_atom` |
| LigandMPNN | inverse folding that can see the substrate | `dauparas/LigandMPNN` |
| ESMFold | fast triage folding | via `transformers` |
| AF2 / Boltz-2 / Chai-1 | final validation, ideally co-folding the ligand | (optional but strongly recommended) |

> **Do not substitute vanilla RFdiffusion or vanilla ProteinMPNN.** Both are
> blind to the small molecule. They will hand you a beautifully folded protein
> with a hydrophobe where your oxyanion hole was supposed to be.

---

## 3. The theozyme is the science. Everything else is engineering.

A theozyme is a quantum-mechanically idealised arrangement of catalytic groups
around a transition-state model. It encodes an actual mechanistic hypothesis:
*this geometry, held rigidly, stabilises this transition state.*

Every subsequent stage is machinery in service of holding that geometry. **If
the theozyme is wrong, the pipeline will faithfully build you a thousand
well-folded proteins that do nothing.**

Build it in `pyscf`/`Gaussian`/ORCA, or take it from the literature.

---

## 4. Running a design campaign

```bash
python src/inference_enzyme.py \
    --config configs/enzyme_design.yaml \
    --theozyme theozymes/ester_hydrolase_v1.json \
    --outdir runs/hydrolase_v1
```

Outputs:

```
runs/hydrolase_v1/
├── 01_backbones/          diffused scaffolds holding the motif
├── 02_sequences/          LigandMPNN designs, catalytic residues fixed
├── 03_predicted/          folded structures
├── ranked_designs.csv     the funnel output, ranked
├── order.fasta            top N, ready for gene synthesis
└── state.json             resumable checkpoint
```

Expect this attrition:

| Stage | Surviving |
|---|---|
| Backbones generated | 1000 |
| Sequences designed | 8000 |
| Fold + pLDDT + RMSD pass | 5–20% |
| **Catalytic constraints satisfied** | **1–10% of those** |
| Ordered | ~96 |
| **Measurably active** | **low single-digit %, optimistically** |

---

## 5. Training the catalytic scorer

```bash
python src/train_enzyme.py --config configs/enzyme_design.yaml --smoke-test
python src/train_enzyme.py --config configs/enzyme_design.yaml --eval-baseline
python src/train_enzyme.py --config configs/enzyme_design.yaml
```

**Data:** M-CSA gives ~1000 enzymes with literature-curated catalytic residues.

**Decoys:** Hard negatives that reproduce design failure modes:
- `identity_swap` — right geometry, wrong chemistry
- `geometry_jitter` — right chemistry, subtly wrong geometry (0.5–2.0 Å displacements)
- `rotamer_flip` — catalytic sidechain in non-productive rotamer

**Splits:** By EC third-level group to prevent homolog leakage.

**Key metric:** `val/auroc_geometry_jitter`. If the model scores 0.95 overall
but 0.60 on jitter, it's useless — it can't see the failure mode you built it for.

---

## 6. Honest limitations

1. **~1000 training examples** from M-CSA. Small dataset, biased toward well-studied enzymes.
2. **Distribution shift:** Trained on natural, evolved active sites; applied to designed ones.
3. **Geometry is necessary, not sufficient.** Catalysis needs dynamics, protonation, loop motion.
4. **ESMFold cannot see your ligand** and is overconfident on de novo sequences.
5. **In-silico success ≠ activity.** Design the assay before the protein. Order redundantly.

---

## 7. References

- Krishna et al. **Generalized biomolecular modeling and design with RoseTTAFold All-Atom.** *Science* (2024).
- Dauparas et al. **Atom level enzyme active site scaffolding using RFdiffusion2.** (2025).
- Watson et al. **De novo design of protein structure and function with RFdiffusion.** *Nature* (2023).
- Ribeiro et al. **Mechanism and Catalytic Site Atlas (M-CSA).** *Nucleic Acids Research*.
- Satorras, Hoogeboom & Welling. **E(n) Equivariant Graph Neural Networks.** *ICML* (2021).
- Röthlisberger et al. **Kemp elimination catalysts by computational enzyme design.** *Nature* (2008).

## Next Steps

1. ✅ Install RFdiffusionAA, LigandMPNN, ESMFold
2. 🔄 Write theozyme for your target reaction
3. 🧪 Design and validate with wet-lab experiments
4. 📊 Publish designs + experimental validation

Good luck! 🚀
