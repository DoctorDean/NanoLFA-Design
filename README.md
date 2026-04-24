# NanoLFA-Design

**Iterative AlphaFold-Guided Nanobody Design for Lateral Flow Immunoassays**

Computational pipeline for the rational design of camelid single-domain antibodies (VHHs / nanobodies) targeting urinary hormone metabolites — such as **pregnanediol-3-glucuronide (PdG)** and **estrone-3-glucuronide (E3G)** — optimized for deployment in lateral flow assays.

---

## Motivation

Home-use LFAs for fertility monitoring require antibodies with a demanding performance profile:

- **High affinity** for small-molecule steroid glucuronide conjugates (MW ~450–500 Da)
- **Sharp specificity** against structurally similar urinary metabolites
- **Fast association kinetics** (analyte transit across test line is <60 seconds)
- **Ambient stability** (no cold chain; shelf life >12 months at 25°C)
- **Orientation-compatible binding** when conjugated to gold nanoparticles or immobilized on nitrocellulose

Nanobodies offer intrinsic advantages for this application: single-domain format, convex paratope geometry suited to hapten binding pockets, superior thermal stability, ease of recombinant production, and straightforward oriented conjugation via a single C-terminal tag.

This pipeline replaces stochastic immunization/screening campaigns with a structure-guided iterative design loop powered by AlphaFold Multimer, ProteinMPNN, and physics-based rescoring.

---

## Pipeline Overview

```
Phase 1 ─ Target Preparation
  └─ Hormone 3D structures (PdG, E3G, cross-reactants)
  └─ Hapten–carrier conjugate modeling
  └─ Epitope surface mapping

Phase 2 ─ Seed Nanobody Generation
  └─ Germline VHH scaffold curation (IMGT)
  └─ CDR3 loop sampling via RFdiffusion
  └─ ProteinMPNN inverse folding for initial sequences

Phase 3 ─ Iterative Design Loop (3–5 rounds)
  ┌─▶ 3a. AlphaFold-Multimer complex prediction
  │   3b. Interface analysis & scoring
  │   3c. CDR diversification (ProteinMPNN / directed mutagenesis)
  │   3d. Re-prediction & composite ranking
  │   3e. Developability filtering
  └───────────────────────────────────────┘

Phase 4 ─ Specificity Engineering
  └─ Cross-reactivity panel screening (in silico)
  └─ Negative design against structural analogs

Phase 5 ─ LFA-Specific Optimization
  └─ Kinetic accessibility scoring
  └─ Conjugation orientation modeling
  └─ Aggregation & stability prediction

Phase 6 ─ Experimental Feedback Integration
  └─ SPR/BLI validation data ingestion
  └─ Scoring function recalibration
  └─ Bayesian optimization of design parameters
```

---

## Target Analytes

| Analyte | Full Name | MW (Da) | CAS | Role |
|---------|-----------|---------|-----|------|
| **PdG** | Pregnanediol-3-glucuronide | 496.6 | 1852-43-3 | Progesterone metabolite; confirms ovulation |
| **E3G** | Estrone-3-glucuronide | 446.5 | 2479-49-4 | Estrogen metabolite; predicts fertile window |

### Cross-Reactivity Panel

| Compound | Relationship | Must discriminate? |
|----------|-------------|-------------------|
| Pregnanediol | PdG aglycone | Yes |
| Pregnanetriol | Structural analog | Yes |
| Pregnanolone glucuronide | Positional isomer | Yes |
| Estrone | E3G aglycone | Yes |
| Estradiol-3-glucuronide | Hydroxylation variant | Yes |
| Estriol-3-glucuronide | Trihydroxy variant | Yes |
| Androsterone glucuronide | Androgen metabolite | Yes |

---

## Repository Structure

```
nanobody-lfa-design/
├── README.md                       # This file
├── CHANGELOG.md                    # Version history
├── LICENSE                         # Apache 2.0
├── pyproject.toml                  # Project metadata & dependencies
├── requirements.txt                # Pinned dependencies
├── environment.yml                 # Conda environment specification
├── Makefile                        # Common commands (includes Docker targets)
├── Dockerfile                      # Multi-stage: core / gpu / full
├── docker-compose.yml              # Service definitions
├── .dockerignore                   # Docker build context exclusions
│
├── .github/
│   ├── workflows/
│   │   └── ci.yml                  # CI: lint + typecheck + tests + config validation
│   ├── CONTRIBUTING.md             # Contribution guidelines
│   ├── PULL_REQUEST_TEMPLATE.md    # PR template with scientific justification
│   ├── CODEOWNERS                  # Auto-reviewer assignment
│   ├── SECURITY.md                 # Vulnerability reporting policy
│   ├── dependabot.yml              # Automated dependency updates
│   └── ISSUE_TEMPLATE/
│       ├── bug_report.md
│       ├── feature_request.md
│       ├── scientific_question.md
│       └── config.yml
│
├── configs/
│   ├── default.yaml                # Master configuration
│   ├── scoring.yaml                # Scoring weights & thresholds
│   ├── md.yaml                     # MD validation parameters
│   ├── targets/
│   │   ├── pdg.yaml                # PdG-specific settings
│   │   └── e3g.yaml                # E3G-specific settings
│   └── hpc/
│       ├── slurm_af_round.sh       # Slurm template: single round
│       └── slurm_full_pipeline.sh  # Slurm template: full pipeline
│
├── docker/
│   └── entrypoint.sh               # Container entry point (GPU detection)
│
├── data/                           # Pipeline I/O (gitignored, mounted as volume)
│   ├── targets/                    # Phase 1 outputs
│   ├── templates/                  # Phase 2 scaffold library
│   ├── results/                    # Phase 3–5 per-round outputs
│   └── experimental/               # Phase 6 wet-lab data (user-provided)
│
├── docs/
│   ├── PROTOCOL.md                 # Detailed computational protocol
│   ├── SCORING.md                  # Scoring function documentation
│   ├── DOCKER.md                   # Docker/Singularity setup guide
│   └── architecture.html           # Interactive pipeline network diagram
│
├── notebooks/
│   ├── 01_target_preparation.ipynb
│   ├── 02_seed_generation.ipynb
│   ├── 03_design_loop.ipynb
│   ├── 04_specificity_analysis.ipynb
│   ├── 05_lfa_optimization.ipynb
│   └── 06_experimental_feedback.ipynb
│
├── scripts/
│   ├── run_pipeline.py             # Master pipeline orchestrator
│   ├── prepare_targets.py          # Phase 1: target preparation
│   ├── run_design_round.py         # Phase 3: single design round (HPC)
│   ├── run_md_validation.py        # Phase 3.5: MD validation
│   ├── screen_crossreactivity.py   # Phase 4: specificity screening
│   ├── optimize_lfa.py             # Phase 5: LFA compatibility
│   ├── ingest_experimental.py      # Phase 6: experimental feedback
│   ├── docker_smoke_test.py        # Synthetic end-to-end test
│   └── setup/
│       ├── fetch_imgt_germlines.py # Phase 2: scaffold curation
│       ├── download_af_params.sh   # AlphaFold database download
│       └── download_proteinmpnn.sh # ProteinMPNN weight download
│
├── src/
│   └── nanolfa/
│       ├── __init__.py
│       ├── core/
│       │   ├── pipeline.py         # Pipeline orchestration
│       │   ├── config.py           # Configuration management
│       │   ├── logging.py          # Structured per-round logging
│       │   ├── hpc.py              # Slurm/PBS/local job submission
│       │   ├── tracking.py         # Weights & Biases integration
│       │   └── calibration.py      # Experimental data ingestion & recalibration
│       ├── models/
│       │   ├── alphafold.py        # AF3 / AF-Multimer wrapper
│       │   ├── proteinmpnn.py      # ProteinMPNN CDR design
│       │   ├── rfdiffusion.py      # RFdiffusion CDR3 backbone generation
│       │   ├── esmfold.py          # ESMFold prescreening (per-region pLDDT)
│       │   └── md_validation.py    # OpenMM molecular dynamics validation
│       ├── scoring/
│       │   ├── composite.py        # Weighted composite scoring function
│       │   ├── confidence.py       # AF confidence extraction (ipTM, pDockQ, PAE)
│       │   ├── structural.py       # Interface geometry (Sc, BSA, contacts)
│       │   ├── energy.py           # Rosetta/FoldX/statistical binding energy
│       │   └── md_scores.py        # MD-derived score adjustments
│       ├── filters/
│       │   ├── developability.py   # Aggregation, charge, hydrophobicity, liabilities
│       │   ├── specificity.py      # Cross-reactivity screening + negative design
│       │   └── lfa_compat.py       # LFA kinetics + orientation + stability gate
│       ├── lfa/
│       │   ├── kinetics.py         # Kinetic accessibility estimation
│       │   ├── orientation.py      # Gold NP conjugation geometry
│       │   └── stability.py        # Thermal stability prediction
│       └── utils/
│           ├── pdb.py              # PDB I/O, chain extraction, RMSD
│           ├── sequence.py         # VHH annotation, validation, clustering
│           ├── chemistry.py        # RDKit conformer generation, epitope mapping
│           └── plotting.py         # Visualization helpers for all phases
│
└── tests/
    ├── conftest.py                 # Shared fixtures
    ├── test_scoring.py             # Composite scoring tests
    ├── test_filters.py             # Developability filter tests
    └── test_sequence.py            # VHH sequence utility tests
```

---

## Quick Start

### Option A: Docker (recommended)

```bash
git clone https://github.com/DoctorDean/nanobody-lfa-design.git
cd nanobody-lfa-design

# Build the CPU-only image (~3GB, no GPU needed)
make docker-core

# Verify everything works
make docker-smoke

# Run Phase 1: target preparation
docker compose run --rm core \
  python scripts/prepare_targets.py --config configs/targets/pdg.yaml

# Start Jupyter notebooks
make docker-notebook
# Open http://localhost:8888
```

See [docs/DOCKER.md](docs/DOCKER.md) for GPU images and HPC deployment.

### Option B: Conda (native)

#### Prerequisites

- Linux (Ubuntu 20.04+ / CentOS 7+)
- CUDA 11.8+ with NVIDIA GPU (A100 80GB recommended; V100 32GB minimum)
- Conda / Mamba
- AlphaFold 3 (for small-molecule docking)

#### Installation

```bash
git clone https://github.com/DoctorDean/nanobody-lfa-design.git
cd nanobody-lfa-design

# Create the conda environment
mamba env create -f environment.yml
conda activate nanolfa

# Install the package in development mode
pip install -e ".[dev]"

# Verify installation
make check

# Download required databases and models
make setup-data
```

### Run the Pipeline

```bash
# Full pipeline for PdG target
python scripts/run_pipeline.py --config configs/targets/pdg.yaml --rounds 5

# Single design round
python scripts/run_design_round.py \
  --target pdg \
  --round 2 \
  --input data/results/round_01/top_candidates.fasta \
  --n-variants 300

# Cross-reactivity screening
python scripts/screen_crossreactivity.py \
  --candidates data/results/round_05/top_candidates.fasta \
  --config configs/targets/pdg.yaml

# MD validation of top candidates
python scripts/run_md_validation.py \
  --candidates data/results/round_05/top_candidates.fasta \
  --complex-dir data/results/round_05/predictions/ \
  --duration-ns 10

# LFA compatibility screening
python scripts/optimize_lfa.py \
  --candidates data/results/specificity/specific_candidates.fasta \
  --config configs/targets/pdg.yaml

# Experimental feedback (after wet-lab data is available)
python scripts/ingest_experimental.py \
  --spr data/experimental/spr_kinetics.csv \
  --scores data/results/round_05/scores.tsv \
  --config configs/targets/pdg.yaml
```

---

## Scoring Function

Candidates are ranked by a composite score (details in `docs/SCORING.md`):

```
S_composite = w1·ipTM + w2·pLDDT_interface + w3·SC + w4·ΔG_bind + w5·BSA_norm + w6·S_dev

where:
  ipTM              = AlphaFold interface predicted TM-score       (w1 = 0.25)
  pLDDT_interface   = mean pLDDT of interface residues (≤5Å)      (w2 = 0.20)
  SC                = Lawrence–Colman shape complementarity         (w3 = 0.15)
  ΔG_bind           = Rosetta/FoldX binding free energy            (w4 = 0.20)
  BSA_norm           = normalized buried surface area               (w5 = 0.10)
  S_dev             = developability meta-score                     (w6 = 0.10)
```

### Decision Thresholds

| Metric | Advance | Borderline | Reject |
|--------|---------|------------|--------|
| ipTM | ≥ 0.75 | 0.60–0.75 | < 0.60 |
| pLDDT (interface) | ≥ 80 | 65–80 | < 65 |
| Shape complementarity | ≥ 0.65 | 0.55–0.65 | < 0.55 |
| ΔG_bind (REU) | ≤ −30 | −20 to −30 | > −20 |
| CDR3 net charge | −2 to +2 | ±3 | > |±3| |
| Aggregation score | < 0.3 | 0.3–0.5 | > 0.5 |

---

## Configuration

All parameters are managed through YAML configs with hierarchical overrides:

```yaml
# configs/default.yaml (abbreviated)
pipeline:
  max_rounds: 5
  variants_per_round: 300
  top_k_advance: 50
  convergence_threshold: 0.02    # stop if score delta < this

alphafold:
  version: "multimer_v2.3"       # or "af3" for AlphaFold 3
  num_models: 5
  num_recycles: 12
  max_template_date: "2025-01-01"
  use_templates: true
  gpu_memory_gb: 80

proteinmpnn:
  sampling_temperature: 0.1
  num_sequences: 100
  backbone_noise: 0.02
  cdr_only: true                 # only redesign CDR loops
  fixed_positions: "framework"   # freeze framework residues

scoring:
  weights:
    iptm: 0.25
    plddt_interface: 0.20
    shape_complementarity: 0.15
    binding_energy: 0.20
    buried_surface_area: 0.10
    developability: 0.10
```

---

## Docker

Three image tiers for different use cases:

| Image | Size | GPU? | Use case |
|---|---|---|---|
| `nanolfa:core` | ~3 GB | No | Phase 1–2, scoring, filters, tests |
| `nanolfa:gpu` | ~12 GB | Yes | Above + ESMFold prescreening |
| `nanolfa:full` | ~25 GB | Yes | Full pipeline including AF3, ProteinMPNN, RFdiffusion |

```bash
make docker-core       # build CPU image
make docker-smoke      # run synthetic end-to-end test
make docker-notebook   # start Jupyter on port 8888
```

See [docs/DOCKER.md](docs/DOCKER.md) for GPU setup, cloud deployment, and Singularity conversion.

---

## Architecture

An interactive network diagram of the full pipeline is available at
[docs/architecture.html](docs/architecture.html). Open it in a browser to
explore the data flow between all 39 components with hover tooltips
showing inputs, outputs, and file paths.

---

## Citation

If you use this pipeline, please cite:

```bibtex
@software{nanolfa_design,
  title   = {NanoLFA-Design: Iterative AlphaFold-Guided Nanobody Design for Lateral Flow Immunoassays},
  year    = {2026},
  url     = {https://github.com/DoctorDean/nanobody-lfa-design}
}
```

And the foundational tools:

- Jumper et al. (2021) *Nature* — AlphaFold
- Dauparas et al. (2022) *Science* — ProteinMPNN
- Watson et al. (2023) *Nature* — RFdiffusion
- Muyldermans (2013) *Annu. Rev. Biochem.* — Nanobodies

---

## License

Apache 2.0. See [LICENSE](LICENSE).

---

## Contributing

See [.github/CONTRIBUTING.md](.github/CONTRIBUTING.md) for guidelines. All computational designs must be experimentally validated before publication claims.
