# ============================================================================
# NanoLFA-Design — Slurm Job Template: AlphaFold Prediction Round
# ============================================================================
# Template for submitting a single design round to a Slurm-managed
# GPU cluster. The HPCManager generates actual .sbatch scripts from
# this template and the pipeline config.
#
# Usage (manual):
#   sbatch configs/hpc/slurm_af_round.sh
#
# Usage (via pipeline):
#   python scripts/run_pipeline.py --config configs/targets/pdg.yaml
#   (the pipeline submits jobs automatically via HPCManager)
# ============================================================================

# --- Slurm directives (override via HPCManager config) ---
# These are defaults; the HPCManager will generate per-round scripts
# with actual values from configs/default.yaml compute section.

#SBATCH --job-name=nanolfa_af_round
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# --- Environment ---
# Uncomment and modify for your cluster:
# module load cuda/11.8
# module load anaconda3
# source activate nanolfa

# For Singularity/Apptainer containers:
# CONTAINER=/path/to/nanolfa.sif
# singularity exec --nv $CONTAINER \

# --- Variables (set by HPCManager) ---
TARGET=${TARGET:-pdg}
ROUND=${ROUND:-1}
CONFIG=${CONFIG:-configs/targets/${TARGET}.yaml}
INPUT_FASTA=${INPUT_FASTA:-data/results/round_$(printf "%02d" $((ROUND-1)))/top_candidates.fasta}
N_VARIANTS=${N_VARIANTS:-300}

echo "================================================"
echo "NanoLFA-Design: AlphaFold Prediction Round"
echo "Target:    ${TARGET}"
echo "Round:     ${ROUND}"
echo "Config:    ${CONFIG}"
echo "Input:     ${INPUT_FASTA}"
echo "Variants:  ${N_VARIANTS}"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Hostname:  $(hostname)"
echo "Date:      $(date)"
echo "================================================"

python scripts/run_design_round.py \
    --target ${TARGET} \
    --round ${ROUND} \
    --input ${INPUT_FASTA} \
    --n-variants ${N_VARIANTS} \
    --config ${CONFIG}

EXIT_CODE=$?
echo "Exit code: ${EXIT_CODE}"
echo "Finished:  $(date)"
exit ${EXIT_CODE}
