# Project Rithik: AI in Molecular Biology

A machine learning framework for analyzing biological sequences, predicting protein structures, and modeling molecular interactions.

## Quick Start

```bash
# Clone and install
git clone https://github.com/rithikreddyy-arch/project-rithik.git
cd project-rithik
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run tests
pytest tests/

# Train a model
python src/train.py --config configs/default.yaml
```

## Stack

- **Language:** Python 3.10+
- **ML Framework:** PyTorch + PyTorch Lightning
- **Biology Libraries:** BioPython, RDKit, ESMFold
- **Data:** NumPy, Pandas, Polars
- **Utilities:** Hydra (config), wandb (tracking)

## Directory Structure

```
project-rithik/
├── README.md
├── requirements.txt          # Python dependencies
├── pyproject.toml           # Package config
├── setup.py                 # Installation
│
├── src/
│   ├── __init__.py
│   ├── data/               # Data loading & preprocessing
│   │   ├── loaders.py      # FASTA, PDB, SMILES parsers
│   │   ├── preprocessing.py # Tokenization, alignment
│   │   └── datasets.py     # PyTorch Dataset classes
│   │
│   ├── models/             # ML architectures
│   │   ├── encoders.py     # Sequence encoders (Transformer, CNN)
│   │   ├── decoders.py     # Structure predictors
│   │   └── losses.py       # Custom loss functions
│   │
│   ├── train.py            # Training loop
│   ├── inference.py        # Prediction pipeline
│   └── utils.py            # Helpers (metrics, visualization)
│
├── configs/                # Configuration files
│   └── default.yaml        # Hyperparameters, paths
│
├── tests/                  # Unit & integration tests
│   ├── test_loaders.py
│   ├── test_models.py
│   └── test_integration.py
│
├── notebooks/              # Exploratory analysis
│   └── eda.ipynb
│
└── data/                   # Data directory (gitignored)
    ├── raw/               # Original sequences, structures
    ├── processed/         # Tokenized, aligned data
    └── models/            # Trained checkpoints
```

## Key Concepts

- **Sequences:** DNA, RNA, or protein strings tokenized into k-mers or BPE tokens
- **Structures:** PDB files converted to coordinate tensors or distance maps
- **Embeddings:** Pre-trained models (ESM, ProtBERT) for transfer learning
- **Tasks:** Classification, generation, structure prediction, binding affinity

## Documentation

- [Setup Guide](docs/setup.md)
- [Data Format Guide](docs/data.md)
- [Model Architecture](docs/models.md)
- [Examples](notebooks/)
