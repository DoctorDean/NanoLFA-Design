# Docker Setup Guide

## Overview

NanoLFA-Design provides three Docker image tiers, each building on the last:

| Image | Size | GPU? | What it runs |
|---|---|---|---|
| `nanolfa:core` | ~3 GB | No | Phase 1, Phase 2 (no ESMFold), Phase 6, all tests |
| `nanolfa:gpu` | ~12 GB | Yes | Above + ESMFold prescreening |
| `nanolfa:full` | ~25 GB | Yes | Full pipeline including AlphaFold 3, ProteinMPNN, RFdiffusion |

Start with `core` to verify everything works, then build `gpu` or `full` when
you need those tools.

## Prerequisites

- Docker 24+ (or Podman)
- Docker Compose v2
- For GPU images: NVIDIA Container Toolkit (`nvidia-docker`)
- For `full` image: ~50 GB disk space (image + AF databases)

### Installing NVIDIA Container Toolkit

```bash
# Ubuntu/Debian
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify: `docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi`

## Quick Start

### Build

```bash
# CPU-only (fastest build, smallest image)
docker compose build core

# With GPU support
docker compose build gpu

# Full pipeline (takes 20–30 minutes)
docker compose build full
```

Or without Compose:

```bash
docker build --target core -t nanolfa:core .
docker build --target gpu  -t nanolfa:gpu .
docker build --target full -t nanolfa:full .
```

### Verify

```bash
# Run the smoke test (validates all phases work)
docker compose run --rm core python scripts/docker_smoke_test.py

# Run the unit test suite
docker compose run --rm test
```

### Run Pipeline Phases

```bash
# Phase 1: Target preparation
docker compose run --rm core \
    python scripts/prepare_targets.py \
    --config configs/targets/pdg.yaml \
    --output /data/targets

# Phase 2: Scaffold curation
docker compose run --rm core \
    python scripts/setup/fetch_imgt_germlines.py \
    --output /data/templates/germline_vhh/

# Full pipeline (requires 'full' image + GPU)
docker compose run --rm full \
    python scripts/run_pipeline.py \
    --config configs/targets/pdg.yaml \
    --rounds 5

# Interactive Jupyter notebooks
docker compose up notebook
# Open http://localhost:8888 in your browser
```

## Data Persistence

The container mounts `./data` from your host machine to `/data` inside the
container. All pipeline outputs (structures, scores, FASTA files) are written
there and persist after the container exits.

```
./data/
├── targets/          # Phase 1 outputs (SDF, PDB, summaries)
├── templates/        # Phase 2 scaffolds
├── results/          # Phase 3–5 outputs (per-round scores, candidates)
├── experimental/     # Phase 6 input (your SPR/LFA CSV files go here)
└── alphafold_db/     # AF databases (mount read-only for 'full' image)
```

## AlphaFold 3 Setup (for `full` image)

AlphaFold 3 requires accepting a license and downloading ~2TB of databases.
These are NOT included in the Docker image.

1. **Accept the license** at https://github.com/google-deepmind/alphafold3

2. **Clone AF3 into the container** (or mount from host):
   ```bash
   docker compose run --rm full bash
   # Inside container:
   cd /opt
   git clone https://github.com/google-deepmind/alphafold3.git alphafold
   cd alphafold
   pip install -e .
   ```

3. **Download databases** (this takes hours and ~2TB disk):
   ```bash
   # On host, create the database directory
   mkdir -p ./alphafold_db
   # Download using AF3's script
   docker compose run --rm full \
       python /opt/alphafold/scripts/download_all_data.py \
       --download_dir /data/alphafold_db
   ```

4. **Set the database path** in docker-compose.yml or via environment:
   ```bash
   AF_DB_PATH=./alphafold_db docker compose run --rm full \
       python scripts/run_pipeline.py --config configs/targets/pdg.yaml
   ```

## Cloud GPU Usage

### Lambda Labs / Vast.ai / RunPod

These providers offer on-demand GPU instances with Docker support:

```bash
# On the cloud instance:
git clone https://github.com/DoctorDean/nanobody-lfa-design.git
cd nanobody-lfa-design
docker compose build gpu   # or 'full' if AF3 databases are available
docker compose run --rm gpu python scripts/docker_smoke_test.py
```

### Google Cloud (with GPU)

```bash
# Create a GPU instance
gcloud compute instances create nanolfa-gpu \
    --zone=us-central1-a \
    --machine-type=n1-standard-8 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --image-family=common-cu118 \
    --image-project=deeplearning-platform-release

# SSH in and run
gcloud compute ssh nanolfa-gpu
# Then clone + docker compose as above
```

## Singularity/Apptainer (HPC Clusters)

Most HPC clusters don't allow Docker but support Singularity/Apptainer.
Convert the Docker image:

```bash
# On a machine with Docker
docker build --target full -t nanolfa:full .
docker save nanolfa:full -o nanolfa_full.tar

# On the HPC cluster
singularity build nanolfa_full.sif docker-archive://nanolfa_full.tar

# Run
singularity exec --nv nanolfa_full.sif \
    python scripts/run_pipeline.py --config configs/targets/pdg.yaml
```

Update `configs/default.yaml` to use the Singularity image:
```yaml
compute:
  container: /path/to/nanolfa_full.sif
```

## Troubleshooting

**`nvidia-smi` works on host but not in container:**
Ensure nvidia-container-toolkit is installed and Docker is restarted.
Run `docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi`.

**Out of memory during AlphaFold:**
Increase Docker's shared memory: `docker compose run --rm --shm-size=16g full ...`
Or edit docker-compose.yml's `shm_size` setting.

**Build fails at conda install:**
Conda solver can be slow. Make sure you're using `mamba` (the Dockerfile
uses Mambaforge by default). If builds stall, try `docker build --no-cache`.

**RDKit import errors in core image:**
Ensure the conda environment is active. The entrypoint script handles this,
but if you override the entrypoint, add:
`export PATH=/opt/conda/envs/nanolfa/bin:$PATH`
