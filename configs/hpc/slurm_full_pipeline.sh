# ============================================================================
# NanoLFA-Design — Slurm Job Template: Full Pipeline
# ============================================================================
# Submits the entire multi-round design loop as a single long-running job.
# For larger runs, consider submitting individual rounds as separate jobs
# with dependencies (the HPCManager does this automatically).
# ============================================================================

#SBATCH --job-name=nanolfa_pipeline
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=120:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# --- Environment ---
# module load cuda/11.8
# source activate nanolfa

TARGET=${TARGET:-pdg}
CONFIG=${CONFIG:-configs/targets/${TARGET}.yaml}
ROUNDS=${ROUNDS:-5}
SEED_FASTA=${SEED_FASTA:-data/templates/germline_vhh/scaffolds.fasta}

echo "================================================"
echo "NanoLFA-Design: Full Pipeline"
echo "Target:    ${TARGET}"
echo "Config:    ${CONFIG}"
echo "Rounds:    ${ROUNDS}"
echo "Seeds:     ${SEED_FASTA}"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Hostname:  $(hostname)"
echo "Date:      $(date)"
echo "================================================"

python scripts/run_pipeline.py \
    --config ${CONFIG} \
    --rounds ${ROUNDS} \
    --seed-fasta ${SEED_FASTA}

EXIT_CODE=$?
echo "Exit code: ${EXIT_CODE}"
echo "Finished:  $(date)"
exit ${EXIT_CODE}
