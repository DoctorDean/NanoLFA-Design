"""Wrapper for AlphaFold-Multimer and AlphaFold 3 predictions.

Handles job submission, output parsing, and batch prediction management
for both nanobody-protein and nanobody-small-molecule complexes.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from omegaconf import DictConfig

if TYPE_CHECKING:
    from nanolfa.core.pipeline import Candidate

logger = logging.getLogger(__name__)


@dataclass
class AFPrediction:
    """Parsed result from a single AlphaFold prediction."""

    model_id: str
    pdb_path: Path
    iptm: float
    ptm: float
    mean_plddt: float
    ranking_confidence: float
    has_clash: bool = False


class AlphaFoldRunner:
    """Manage AlphaFold-Multimer / AF3 predictions for nanobody–target complexes.

    Supports two modes:
    - AF-Multimer v2.3: for nanobody–protein complexes (hapten conjugated to carrier)
    - AlphaFold 3: for nanobody–small-molecule complexes (direct hapten binding)

    AF3 is preferred for small-molecule targets (PdG, E3G) as it natively handles
    ligand–protein interactions via its diffusion-based architecture.
    """

    def __init__(self, config: DictConfig) -> None:
        self.config = config
        self.version = config.version  # "af3" or "multimer_v2.3"
        self.install_path = Path(config.install_path)
        self.db_path = Path(config.database_path)
        self.num_models = config.num_models
        self.num_recycles = config.num_recycles
        self.num_seeds = config.num_seeds
        self.use_amber_relax = config.use_amber_relax

        logger.info(
            "AlphaFold runner: version=%s, models=%d, recycles=%d, seeds=%d",
            self.version, self.num_models, self.num_recycles, self.num_seeds,
        )

    def predict_batch(
        self,
        candidates: list[Candidate],
        target_config: DictConfig,
        output_dir: Path,
    ) -> list[Candidate]:
        """Run AlphaFold predictions for a batch of candidates.

        For each candidate, predicts the complex with the target analyte and
        populates the candidate's structure_path and complex_path fields.

        Args:
            candidates: List of Candidate objects with sequences.
            target_config: Target-specific config (contains SMILES, ligand file, etc.)
            output_dir: Directory for prediction outputs.

        Returns:
            Candidates with populated structure/complex paths.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        predicted: list[Candidate] = []

        for i, cand in enumerate(candidates):
            cand_dir = output_dir / cand.candidate_id
            cand_dir.mkdir(parents=True, exist_ok=True)

            logger.debug(
                "Predicting %d/%d: %s (len=%d)",
                i + 1, len(candidates), cand.candidate_id, len(cand.sequence),
            )

            try:
                if self.version == "af3":
                    result = self._predict_af3(cand, target_config, cand_dir)
                else:
                    result = self._predict_multimer(cand, target_config, cand_dir)

                # Select best model by ranking_confidence
                best = max(result, key=lambda r: r.ranking_confidence)
                cand.complex_path = best.pdb_path
                cand.iptm = best.iptm
                predicted.append(cand)

            except Exception as e:
                logger.error("Prediction failed for %s: %s", cand.candidate_id, e)
                continue

        logger.info(
            "Predicted %d/%d candidates successfully", len(predicted), len(candidates)
        )
        return predicted

    def _predict_af3(
        self,
        candidate: Candidate,
        target_config: DictConfig,
        output_dir: Path,
    ) -> list[AFPrediction]:
        """Run AlphaFold 3 prediction for nanobody + small-molecule ligand.

        AF3 takes a JSON input specifying protein sequences and ligand structures.
        """
        # Build AF3 input JSON
        ligand_file = Path(target_config.alphafold.af3.ligand_file) if hasattr(
            target_config, 'alphafold'
        ) else None

        seeds = list(range(42, 42 + self.num_seeds))

        input_spec = {
            "name": candidate.candidate_id,
            "modelSeeds": seeds,
            "sequences": [
                {
                    "protein": {
                        "id": "A",
                        "sequence": candidate.sequence,
                    }
                }
            ],
        }

        # Add ligand
        if ligand_file and ligand_file.exists():
            sdf_contents = ligand_file.read_text()
            input_spec["ligands"] = [{"id": "B", "sdf": sdf_contents}]
        elif target_config.smiles:
            input_spec["ligands"] = [{"id": "B", "smiles": target_config.smiles}]
        else:
            raise ValueError(f"No ligand info for target {target_config.name}")

        # Write input JSON
        input_json = output_dir / "af3_input.json"
        with open(input_json, "w") as f:
            json.dump(input_spec, f, indent=2)

        # Run AF3
        cmd = [
            "python", str(self.install_path / "run_alphafold.py"),
            "--json_path", str(input_json),
            "--model_dir", str(self.db_path / "af3_models"),
            "--output_dir", str(output_dir),
            "--num_diffusion_samples", str(self.config.af3.diffusion_samples),
        ]

        logger.debug("AF3 command: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

        if result.returncode != 0:
            raise RuntimeError(f"AF3 failed: {result.stderr[:500]}")

        return self._parse_af3_output(output_dir)

    def _predict_multimer(
        self,
        candidate: Candidate,
        target_config: DictConfig,
        output_dir: Path,
    ) -> list[AFPrediction]:
        """Run AF-Multimer v2.3 prediction for nanobody + protein target."""
        # Write FASTA with nanobody + target chains
        fasta_path = output_dir / "input.fasta"
        with open(fasta_path, "w") as f:
            f.write(f">nanobody\n{candidate.sequence}\n")
            # For hapten-carrier conjugate, use carrier protein epitope region
            # This would be loaded from the target config
            if hasattr(target_config, "carrier_epitope_sequence"):
                f.write(f">carrier\n{target_config.carrier_epitope_sequence}\n")

        cmd = [
            "python", str(self.install_path / "run_alphafold.py"),
            "--fasta_paths", str(fasta_path),
            "--model_preset", "multimer",
            "--num_multimer_predictions_per_model", "1",
            "--max_template_date", self.config.max_template_date,
            "--output_dir", str(output_dir),
            "--data_dir", str(self.db_path),
        ]

        if self.use_amber_relax:
            cmd.append("--use_gpu_relax")

        logger.debug("AF-Multimer command: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)

        if result.returncode != 0:
            raise RuntimeError(f"AF-Multimer failed: {result.stderr[:500]}")

        return self._parse_multimer_output(output_dir)

    def _parse_af3_output(self, output_dir: Path) -> list[AFPrediction]:
        """Parse AF3 output files into AFPrediction objects."""
        predictions: list[AFPrediction] = []

        for pdb_file in sorted(output_dir.glob("*.pdb")):
            json_file = pdb_file.with_suffix(".json")
            if json_file.exists():
                with open(json_file) as f:
                    data = json.load(f)
                predictions.append(
                    AFPrediction(
                        model_id=pdb_file.stem,
                        pdb_path=pdb_file,
                        iptm=data.get("iptm", 0.0),
                        ptm=data.get("ptm", 0.0),
                        mean_plddt=data.get("mean_plddt", 0.0),
                        ranking_confidence=data.get("ranking_confidence", data.get("iptm", 0.0)),
                    )
                )

        if not predictions:
            raise RuntimeError(f"No AF3 predictions found in {output_dir}")

        return predictions

    def _parse_multimer_output(self, output_dir: Path) -> list[AFPrediction]:
        """Parse AF-Multimer output files into AFPrediction objects."""
        predictions: list[AFPrediction] = []

        # AF2 writes ranking_debug.json with model rankings
        ranking_file = output_dir / "ranking_debug.json"
        if ranking_file.exists():
            with open(ranking_file) as f:
                ranking = json.load(f)

            for model_name, iptm_ptm in ranking.get("iptm+ptm", {}).items():
                pdb_path = output_dir / f"relaxed_{model_name}.pdb"
                if not pdb_path.exists():
                    pdb_path = output_dir / f"unrelaxed_{model_name}.pdb"
                if pdb_path.exists():
                    predictions.append(
                        AFPrediction(
                            model_id=model_name,
                            pdb_path=pdb_path,
                            iptm=ranking.get("iptm", {}).get(model_name, 0.0),
                            ptm=ranking.get("ptm", {}).get(model_name, 0.0),
                            mean_plddt=ranking.get("plddts", {}).get(model_name, 0.0),
                            ranking_confidence=iptm_ptm,
                        )
                    )

        if not predictions:
            # Fallback: scan for PDB files directly
            for pdb_file in sorted(output_dir.glob("*model*.pdb")):
                predictions.append(
                    AFPrediction(
                        model_id=pdb_file.stem,
                        pdb_path=pdb_file,
                        iptm=0.0,
                        ptm=0.0,
                        mean_plddt=0.0,
                        ranking_confidence=0.0,
                    )
                )

        return predictions
