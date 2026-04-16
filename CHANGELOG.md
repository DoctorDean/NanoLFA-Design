# Changelog

All notable changes to the NanoLFA-Design pipeline will be documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] — 2026-04-14

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

### Planned for v0.2.0

- Phase 1 implementation: target preparation scripts with RDKit/OpenBabel integration
- Hapten–carrier conjugate builder for BSA/KLH linker modeling
- Germline VHH scaffold library (curated from IMGT, >200 unique frameworks)
- ESMFold fast pre-screening module for rapid variant triage

### Planned for v0.3.0

- Phase 3 iterative loop: full automation of predict → score → diversify → re-rank cycle
- Slurm/PBS job submission templates for HPC clusters
- Weights & Biases integration for experiment tracking

### Planned for v0.4.0

- Phase 4 cross-reactivity screening with automated negative-design constraints
- Phase 5 LFA optimization with gold nanoparticle conjugation modeling
- Phase 6 experimental data ingestion and Bayesian score recalibration

### Planned for v1.0.0

- End-to-end validated pipeline with published experimental confirmation
- Pre-trained scoring models calibrated on internal SPR/BLI data
- Docker container for reproducible deployment
