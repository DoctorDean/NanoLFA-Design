# ============================================================================
# NanoLFA-Design — Multi-Stage Dockerfile
# ============================================================================
#
# Three image tiers (build whichever you need):
#
#   nanolfa:core    — Python + RDKit + scoring + filters (no GPU, ~3GB)
#                     Runs: Phase 1, Phase 2 (no ESMFold), Phase 6
#
#   nanolfa:gpu     — core + PyTorch + ESMFold + CUDA runtime (~12GB)
#                     Runs: Phase 2 (with ESMFold prescreening)
#
#   nanolfa:full    — gpu + AlphaFold3 + ProteinMPNN + RFdiffusion (~25GB)
#                     Runs: Full pipeline end-to-end
#
# Usage:
#   docker build --target core -t nanolfa:core .
#   docker build --target gpu  -t nanolfa:gpu .
#   docker build --target full -t nanolfa:full .
#
# Or via docker-compose:
#   docker compose build core
#   docker compose run --rm core python scripts/prepare_targets.py ...
#
# ============================================================================

# ---- Stage 1: Base system with conda ----------------------------------------
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
        wget curl git make ca-certificates \
        libgl1-mesa-glx libglib2.0-0 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Install Mambaforge (fast conda)
ENV CONDA_DIR=/opt/conda
RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh \
        -O /tmp/mambaforge.sh \
    && bash /tmp/mambaforge.sh -b -p $CONDA_DIR \
    && rm /tmp/mambaforge.sh
ENV PATH=$CONDA_DIR/bin:$PATH

# ---- Stage 2: Core scientific environment (CPU, no GPU models) ---------------
FROM base AS core

WORKDIR /app

# Copy environment spec and install core dependencies
COPY environment.yml /tmp/environment.yml

# Create a trimmed environment without GPU-heavy packages
# This keeps the core image small (~3GB)
RUN mamba create -n nanolfa python=3.10 -y && \
    mamba install -n nanolfa -c conda-forge -c bioconda -y \
        numpy=1.26.4 scipy=1.13.0 pandas=2.2.2 \
        biopython=1.83 rdkit=2024.03.3 openbabel=3.1.1 \
        matplotlib=3.9.0 seaborn=0.13.2 \
        click rich pyyaml && \
    /opt/conda/envs/nanolfa/bin/pip install --no-cache-dir \
        omegaconf==2.3.0 hydra-core==1.3.2 \
        freesasa==2.2.1 wandb==0.17.0 \
        pytest==8.2.0 ruff==0.4.4 mypy==1.10.0 && \
    mamba clean -afy

# Activate the nanolfa environment by default
ENV PATH=/opt/conda/envs/nanolfa/bin:$PATH
ENV CONDA_DEFAULT_ENV=nanolfa

# Copy project files
COPY pyproject.toml setup.cfg* README.md LICENSE ./
COPY src/ src/
COPY configs/ configs/
COPY scripts/ scripts/
COPY tests/ tests/
COPY notebooks/ notebooks/
COPY docs/ docs/
COPY Makefile ./
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Install the nanolfa package
RUN pip install --no-cache-dir -e .

# Verify core installation
RUN python -c "from nanolfa.utils.chemistry import generate_conformers; print('Core OK')" && \
    python -c "from nanolfa.utils.sequence import load_bundled_germlines; print('Sequence OK')" && \
    python -c "from nanolfa.scoring.composite import CompositeScorer; print('Scoring OK')"

# Default data directories
RUN mkdir -p /data/targets /data/templates /data/results /data/experimental

VOLUME ["/data"]
ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "--version"]

# ---- Stage 3: GPU environment with PyTorch + ESMFold -------------------------
FROM core AS gpu

# Install PyTorch with CUDA support
RUN mamba install -n nanolfa -c pytorch -c nvidia -y \
        pytorch=2.3.0 pytorch-cuda=11.8 && \
    mamba clean -afy

# Install ESMFold
RUN pip install --no-cache-dir \
        fair-esm==2.0.1

# Verify GPU stack
RUN python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')" && \
    python -c "from nanolfa.models.esmfold import ESMFoldPrescreen; print('ESMFold OK')"

# ---- Stage 4: Full pipeline with AF3 + ProteinMPNN + RFdiffusion ------------
FROM gpu AS full

# Environment variables for tool locations
ENV ALPHAFOLD_PATH=/opt/alphafold
ENV PROTEINMPNN_PATH=/opt/ProteinMPNN
ENV RFDIFFUSION_PATH=/opt/RFdiffusion
ENV AF_DB_PATH=/data/alphafold_db

# Install additional ML frameworks
RUN pip install --no-cache-dir \
        jax[cuda12]==0.4.28 \
        dm-haiku==0.0.12 \
        tensorflow==2.16.1

# ProteinMPNN
RUN git clone --depth 1 https://github.com/dauparas/ProteinMPNN.git $PROTEINMPNN_PATH && \
    echo "ProteinMPNN installed at $PROTEINMPNN_PATH"

# RFdiffusion (clone repo; weights downloaded separately)
RUN git clone --depth 1 https://github.com/RosettaCommons/RFdiffusion.git $RFDIFFUSION_PATH && \
    cd $RFDIFFUSION_PATH && \
    pip install --no-cache-dir -e . 2>/dev/null || echo "RFdiffusion deps may need manual install" && \
    echo "RFdiffusion installed at $RFDIFFUSION_PATH"

# AlphaFold 3 — requires manual installation due to licensing.
# Users must:
#   1. Accept the AF3 license at https://github.com/google-deepmind/alphafold3
#   2. Clone the repo into /opt/alphafold inside the container
#   3. Download model parameters
#
# Placeholder directory structure:
RUN mkdir -p $ALPHAFOLD_PATH && \
    echo "AlphaFold 3 not pre-installed. See docs/DOCKER.md for setup instructions." \
    > $ALPHAFOLD_PATH/INSTALL_INSTRUCTIONS.txt

# Verify full stack
RUN python -c "from nanolfa.models.proteinmpnn import ProteinMPNNDesigner; print('MPNN OK')" && \
    python -c "from nanolfa.models.rfdiffusion import RFdiffusionRunner; print('RFdiff OK')" && \
    python -c "from nanolfa.models.alphafold import AlphaFoldRunner; print('AF runner OK')"

# Label
LABEL org.opencontainers.image.title="NanoLFA-Design" \
      org.opencontainers.image.description="Iterative AlphaFold-guided nanobody design for lateral flow immunoassays" \
      org.opencontainers.image.source="https://github.com/DoctorDean/nanobody-lfa-design" \
      org.opencontainers.image.licenses="Apache-2.0"
