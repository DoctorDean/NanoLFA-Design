# Changelog

All notable changes to the NanoLFA-Design pipeline will be documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.5.0] — 2026-04-21

### Added

- **Project scaffold**: full repository structure, documentation framework, and CI skeleton
- **Target definitions**: PdG and E3G analyte configurations with cross-reactivity panels
- **Configuration system**: hierarchical YAML configs with per-target overrides
  - `configs/default.yaml` — master pipeline parameters
  - `configs/scoring.yaml` — composite scoring weights and thresholds
  - `configs/alphafold.yaml` — AlphaFold-Multimer run parameters
  - `configs/proteinmpnn.yaml` — ProteinMPNN sequence design parameters
  - `configs/targets/pdg.yaml` — PdG-specific settings
  - `configs/targets/e3g.yaml` — E3G-specific settings
- **Scoring framework**: composite ranking function with six weighted components
  - ipTM, pLDDT (interface), shape complementarity, ΔG_bind, BSA, developability
  - Decision thresholds documented in `docs/THRESHOLDS.md`
- **Developability filters**: aggregation propensity, charge, hydrophobicity, humanness
- **LFA compatibility module**: kinetic accessibility, conjugation orientation, thermal stability
- **Documentation**:
  - `docs/PROTOCOL.md` — full computational protocol with tool configurations
  - `docs/SCORING.md` — scoring function derivation and calibration notes
  - `docs/THRESHOLDS.md` — threshold rationale with literature references
- **Pipeline scripts**: orchestrator and per-phase execution scripts
- **Conda environment**: `environment.yml` with pinned dependencies
- **CI**: GitHub Actions workflow for linting, type checking, and unit tests

### Technical Decisions

- **Nanobody-only design**: VHH scaffolds selected over conventional IgG Fab/scFv formats
  due to superior LFA performance (single-domain stability, oriented conjugation,
  convex paratope geometry for hapten binding)
- **AlphaFold 3 preferred for small-molecule targets**: AF3's diffusion-based architecture
  handles ligand-protein complexes natively; AF-Multimer v2.3 used as fallback for
  protein–protein complexes only
- **ProteinMPNN for CDR redesign**: fixed-backbone sequence design with framework residues
  frozen; CDR3 loop is primary diversification target
- **Composite scoring**: equal-ish weighting across confidence, geometry, and energy terms
  to avoid over-reliance on any single predictor; weights will be recalibrated after
  Phase 6 experimental feedback

---

## [Unreleased]

### Added

- **Phase 1 target preparation** (`src/nanolfa/utils/chemistry.py`):
  ETKDG v3 conformer generation with MMFF94/UFF minimization, SDF/PDB/MOL2
  export, molecular property computation, per-atom SASA epitope mapping with
  linker occlusion modeling, and pharmacophore feature extraction
- **Target preparation CLI** (`scripts/prepare_targets.py`): multi-target
  support, configurable conformer count and force field, cross-reactant
  panel processing, JSON summary output
- **Phase 1 notebook** (`notebooks/01_target_preparation.ipynb`): interactive
  walkthrough with 2D visualization, property radar charts, SASA distribution
  plots, pharmacophore comparison, and design implications for Phase 2
- **VHH sequence utilities** (`src/nanolfa/utils/sequence.py`): IMGT region
  annotation, VHH hallmark residue validation (positions 37/44/45/47),
  canonical disulfide verification, greedy leader-follower sequence
  clustering, and a bundled library of 10 representative VHH germline
  scaffolds from alpaca and dromedary (curated from IMGT/PDB)
- **Germline scaffold curation CLI** (`scripts/setup/fetch_imgt_germlines.py`):
  loads bundled or user-provided VHH sequences, validates VHH hallmarks,
  clusters by framework identity, exports FASTA + JSON with per-scaffold
  region annotation and validation reports
- **Enhanced ESMFold prescreening** (`src/nanolfa/models/esmfold.py`):
  per-IMGT-region pLDDT breakdown (FR1/CDR1/.../FR4), CDR confidence
  ratio metric (CDR pLDDT / framework pLDDT), PDB output for predicted
  structures, dedicated `screen_scaffolds()` method for Phase 2 evaluation
- **Phase 2 notebook** (`notebooks/02_seed_generation.ipynb`): germline
  library loading, VHH validation scoring with bar charts, CDR3 length
  distribution, framework clustering, ESMFold screening with per-region
  heatmap and CDR confidence ratio plot
- **Sequence utility tests** (`tests/test_sequence.py`): 15 tests covering
  region annotation, VHH validation, clustering, bundled germline loading,
  and library curation
- **RFdiffusion wrapper** (`src/nanolfa/models/rfdiffusion.py`): CDR3
  backbone generation with IMGT-aware contig specification, target-guided
  diffusion with hotspot residues, and batch scaffold processing
- **HPC job manager** (`src/nanolfa/core/hpc.py`): Slurm and PBS job
  submission with sbatch/qsub script generation, dependency chaining,
  job arrays, Singularity container wrapping, status polling, and
  multi-job wait
- **Experiment tracking** (`src/nanolfa/core/tracking.py`): Weights &
  Biases integration for per-round metrics, candidate score tables,
  convergence curves, and artifact uploads; graceful no-op fallback
- **Structured logging** (`src/nanolfa/core/logging.py`): per-round log
  file rotation and HPC-friendly formatting
- **Slurm job templates** (`configs/hpc/`): single-round and full-pipeline
  templates with environment variable injection
- **Phase 3 notebook** (`notebooks/03_design_loop.ipynb`): scoring function
  visualization, simulated 5-round design loop with convergence analysis,
  per-metric distribution box plots, W&B tracking demo, HPC submission
- **Cross-reactivity screening** (`src/nanolfa/filters/specificity.py`):
  rewritten with full AF3 off-target prediction integration, per-compound
  contact analysis, three-tier classification (specific/borderline/cross-
  reactive), negative-design rescue pathway for borderline candidates,
  and comprehensive reporting (TSV, JSON, cross-reactivity matrix)
- **Screening CLI** (`scripts/screen_crossreactivity.py`): rewritten with
  `--rescue` flag for negative-design rescue, detailed terminal summary
  with tier counts and top candidate listing
- **Phase 4 notebook** (`notebooks/04_specificity_analysis.ipynb`):
  simulated cross-reactivity data, heatmap of candidate-by-compound
  scores, selectivity ratio analysis, on-target vs off-target scatter
  plot with selectivity contours, per-compound box plots, negative
  design concept walkthrough
- **Kinetic accessibility module** (`src/nanolfa/lfa/kinetics.py`):
  ray-casting pocket openness, CDR rigidity from pLDDT, charge
  complementarity at interface, composite kon estimation with
  expected binding fraction at LFA flow time
- **Thermal stability module** (`src/nanolfa/lfa/stability.py`):
  sequence-based Tm/Tagg prediction from amino acid composition,
  CDR3 flexibility penalty, free cysteine and methionine risk scoring
- **LFA compatibility filter** (`src/nanolfa/filters/lfa_compat.py`):
  unified gate combining kinetics, orientation, and stability with
  composite LFA score and batch filtering
- **LFA optimization CLI** (`scripts/optimize_lfa.py`): evaluates
  candidates for LFA deployment suitability with detailed TSV output
- **Phase 5 notebook** (`notebooks/05_lfa_optimization.ipynb`):
  thermal stability bar charts, conjugation orientation scatter plot,
  kinetic accessibility stacked components, combined LFA pass/fail table
- **Experimental calibration module** (`src/nanolfa/core/calibration.py`):
  CSV ingestion for SPR kinetics, thermal shift, and LFA prototype data;
  per-metric Pearson/Spearman correlation analysis vs experimental log(KD);
  ridge regression weight recalibration with cross-validation; correlation-
  based recalibration fallback; hidden gem detection via re-ranking;
  calibration JSON export for config update
- **Experimental ingestion CLI** (`scripts/ingest_experimental.py`):
  step-by-step workflow with correlation table, weight change summary,
  hidden gem listing, and calibration JSON output
- **Phase 6 notebook** (`notebooks/06_experimental_feedback.ipynb`):
  simulated experimental data, metric-vs-log(KD) scatter plots with
  regression lines, weight recalibration comparison bar chart, correlation
  strength ranking, hidden gem rank-change arrow visualization
- **Docker multi-stage build** (`Dockerfile`): three image tiers — `core`
  (CPU-only, ~3GB: RDKit + scoring + filters), `gpu` (~12GB: + PyTorch +
  ESMFold), `full` (~25GB: + AlphaFold 3 + ProteinMPNN + RFdiffusion)
- **Docker Compose** (`docker-compose.yml`): services for core, gpu, full,
  notebook (Jupyter on port 8888), and test runner
- **Container entrypoint** (`docker/entrypoint.sh`): GPU auto-detection,
  conda activation, data directory setup
- **Smoke test** (`scripts/docker_smoke_test.py`): synthetic end-to-end
  validation of all six phases without GPU or external tools — proves the
  pipeline executes correctly inside the container
- **Docker documentation** (`docs/DOCKER.md`): build instructions, cloud
  GPU setup (Lambda/Vast.ai/GCP), Singularity/Apptainer conversion for
  HPC, AlphaFold 3 database setup, troubleshooting
- **Makefile Docker targets**: docker-core, docker-gpu, docker-full,
  docker-smoke, docker-test, docker-notebook

### Fixed (post-0.1.0)

- Resolve all ruff violations from initial CI run (unused imports, legacy
  `Optional[X]` syntax modernized to `X | None`, long lines wrapped,
  `zip()` calls updated with `strict=` parameter, ambiguous variable names
  in tests renamed)
- Fix loop variable binding bug in `ChainSelect` closure in
  `scoring/composite.py` — previously the chain selector would reference
  the wrong chain due to Python's late-binding behavior, which would have
  caused incorrect BSA calculations
- Relax mypy config to `ignore_missing_imports = true` and
  `warn_return_any = false` — scientific libraries (Biopython, FreeSASA,
  PyRosetta, RDKit) have incomplete type stubs. Will tighten as typed
  wrappers are added.
- Add `DictConfig` type narrowing in `core/config.py` to satisfy mypy
  with OmegaConf's union return type
- Make ESMFold lazy loader type-safe by assigning to local variable
  before setting instance field
- Remove `pytest-cov` from default addopts to avoid CI failure when
  coverage plugin isn't installed
- Migrate ruff `select` from top-level to `[tool.ruff.lint]` section
  (deprecated location)

### Planned for v0.2.0

- ~~Phase 1 implementation: target preparation scripts with RDKit/OpenBabel integration~~ ✅
- ~~Hapten–carrier conjugate builder for BSA/KLH linker modeling~~ ✅
- ~~Germline VHH scaffold library (curated from IMGT, >200 unique frameworks)~~ ✅
- ~~ESMFold fast pre-screening module for rapid variant triage~~ ✅

### Planned for v0.3.0

- ~~Phase 3 iterative loop: full automation of predict → score → diversify → re-rank cycle~~ ✅
- ~~Slurm/PBS job submission templates for HPC clusters~~ ✅
- ~~Weights & Biases integration for experiment tracking~~ ✅

### Planned for v0.4.0

- ~~Phase 4 cross-reactivity screening with automated negative-design constraints~~ ✅
- ~~Phase 5 LFA optimization with gold nanoparticle conjugation modeling~~ ✅
- ~~Phase 6 experimental data ingestion and Bayesian score recalibration~~ ✅

### Planned for v1.0.0

- End-to-end validated pipeline with published experimental confirmation
- Pre-trained scoring models calibrated on internal SPR/BLI data
- Docker container for reproducible deployment
