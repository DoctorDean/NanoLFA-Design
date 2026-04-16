#!/usr/bin/env python3
"""Run the NanoLFA-Design pipeline end-to-end.

Usage:
    python scripts/run_pipeline.py --config configs/targets/pdg.yaml --rounds 5
    python scripts/run_pipeline.py --config configs/targets/e3g.yaml --seed-fasta seeds.fasta
"""

from nanolfa.core.pipeline import main

if __name__ == "__main__":
    main()
