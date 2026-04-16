# Computational Protocol

## Detailed Tool Configurations and Execution Steps

This document specifies the exact computational procedures, tool invocations, and
decision logic for each phase of the NanoLFA-Design pipeline.

---

## Phase 1 — Target Preparation

### 1.1 Hormone 3D Structure Generation

**Tool**: RDKit + OpenBabel + Gaussian/xTB

For each target analyte (PdG, E3G) and every member of the cross-reactivity panel:

1. Parse SMILES from target config.
2. Generate 3D coordinates with RDKit's `AllChem.EmbedMolecule()` using the ETKDG
   conformer generator (MMFF94 force field, 200 conformers, random seed 42).
3. Select the lowest-energy conformer.
4. Optional: refine with GFN2-xTB semi-empirical QM for accurate partial charges
   and geometry. This matters because steroid ring puckering affects binding pocket
   shape recognition.
5. Export as SDF (for AlphaFold 3) and MOL2 (for docking tools).

```bash
python scripts/prepare_targets.py \
  --config configs/targets/pdg.yaml \
  --method etkdg+xtb \
  --n-conformers 200
```

### 1.2 Hapten–Carrier Conjugate Modeling

**Tool**: RDKit + custom linker builder

The immune system and our computational pipeline see the hapten in the context of
its conjugation to a carrier protein. The linker chemistry and attachment point
determine which parts of the hapten are exposed (and therefore immunogenic) vs.
buried against the carrier.

For PdG (carbodiimide coupling via glucuronide COOH):
- The glucuronide moiety is partially occluded by the linker.
- The steroid A/B ring system is maximally exposed.
- Therefore, nanobodies should primarily contact the steroid core,
  with the glucuronide acting as a secondary recognition element.

For E3G (carbodiimide coupling via glucuronide COOH):
- Similar geometry: aromatic A-ring and steroid D-ring are exposed.
- The glucuronide carboxylate is consumed by the linker.

Procedure:
1. Build the linker–hapten conjugate in RDKit.
2. Attach to a representative BSA surface patch (PDB: 4F5S, domain II).
3. Energy-minimize the conjugate on the protein surface.
4. Compute the solvent-accessible surface of the hapten atoms
   to identify the "epitope surface" available to nanobody binding.

### 1.3 Epitope Surface Mapping

**Tool**: FreeSASA + custom analysis

Calculate per-atom SASA of the hapten in the conjugate context.
Atoms with SASA > 5 Å² are classified as epitope-accessible.
Generate a pharmacophore map: H-bond donors, acceptors, hydrophobic patches,
aromatic rings — this guides CDR design in Phase 2.

---

## Phase 2 — Seed Nanobody Generation

### 2.1 Germline VHH Scaffold Curation

**Source**: IMGT (http://www.imgt.org)

1. Download all camelid IGHV, IGHD, IGHJ germline sequences.
2. Focus on *Vicugna pacos* (alpaca) and *Camelus dromedarius* (dromedary) —
   best-characterized VHH germlines.
3. Select representative frameworks by clustering at 90% sequence identity
   (CD-HIT). Retain ≥10 unique framework families.
4. For each framework, model the 3D structure using AlphaFold (monomer mode,
   5 models, 12 recycles). Select the model with highest pLDDT.
5. Verify the canonical framework disulfide (Cys23–Cys104, IMGT numbering).
6. Verify hallmark VHH substitutions at positions 37, 44, 45, 47 (IMGT) that
   distinguish VHH from conventional VH.

```bash
python scripts/setup/fetch_imgt_germlines.py \
  --species "Vicugna pacos" "Camelus dromedarius" \
  --cluster-identity 0.90 \
  --output data/templates/germline_vhh/
```

### 2.2 CDR3 Loop Generation via RFdiffusion

**Tool**: RFdiffusion

For hapten-binding nanobodies, CDR3 is the primary determinant of specificity.
We use RFdiffusion to generate diverse CDR3 backbone geometries that are
sterically compatible with the hapten epitope surface.

Configuration:
```yaml
rfdiffusion:
  model: "Complex_beta_ckpt.pt"
  num_designs: 50
  contigs: "A1-26/0 A40-55/0 A66-104/12-20 A105-120/0"
  #         FR1     CDR1(fix) FR2    CDR3(design) FR3+FR4
  #         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  #         Nanobody chain with CDR3 loop designed de novo
  potentials:
    guide_scale: 2.0
    guide_potentials: "monomer_ROG,binder_distance_ReLU"
  noise_scale: 1.0
  diffusion_steps: 50
```

The `A66-104/12-20` contig tells RFdiffusion to design a 12–20 residue CDR3 loop
between framework positions 66 and 105 (IMGT numbering).

For each of the 50 backbone designs:
1. Score the backbone with AF (fast monomer mode) — reject if pLDDT < 70.
2. Check for steric clashes with the hapten (placed by superposition onto the
   epitope surface from Phase 1).

### 2.3 Sequence Design via ProteinMPNN

**Tool**: ProteinMPNN

For each surviving CDR3 backbone from 2.2:

```bash
python /opt/ProteinMPNN/protein_mpnn_run.py \
  --pdb_path <backbone.pdb> \
  --out_folder data/results/seed_sequences/ \
  --num_seq_per_target 100 \
  --sampling_temp 0.1 \
  --backbone_noise 0.02 \
  --seed 42 \
  --batch_size 1 \
  --chain_id_jsonl <fixed_chains.jsonl> \
  --fixed_positions_jsonl <framework_positions.jsonl>
```

Key parameters:
- `sampling_temp=0.1`: conservative; produces sequences close to the energy minimum
  for each backbone. We increase this in later rounds for diversity.
- `fixed_positions`: all framework residues (FR1, FR2, FR3, FR4) are frozen.
  Only CDR1, CDR2, CDR3 residues are designed.
- `backbone_noise=0.02`: 0.02Å Gaussian noise adds slight diversity.

Amino acid biases (applied via `--bias_AA_jsonl`):
- Penalize Cys (free cysteines cause aggregation)
- Penalize Met (oxidation under LFA storage conditions)
- Penalize Asn-Gly, Asn-Ser motifs (deamidation hotspots)
- Boost Trp, Tyr, Phe in CDR3 (aromatic contacts with steroid ring system)

Output: ~5,000 seed sequences (50 backbones × 100 sequences each).

### 2.4 Fast Pre-Screening with ESMFold

**Tool**: ESMFold (via fair-esm)

Before committing expensive AF-Multimer runs, rapidly screen all 5,000 seeds:

```python
import esm

model = esm.pretrained.esmfold_v1()
# For each sequence:
prediction = model.infer(sequence)
plddt = prediction["plddt"].mean()
# Reject if pLDDT < 60
```

Retain the top 300 sequences (by mean pLDDT) for full AlphaFold evaluation.

---

## Phase 3 — Iterative Design Loop

### Overview

Each round follows: Predict → Score → Diversify → Re-predict → Filter.
Rounds 1–2 are exploratory (higher diversity). Rounds 3–5 are exploitative
(lower temperature, focused mutations).

### 3a. AlphaFold-Multimer Complex Prediction

**Tool**: AlphaFold 3 (preferred) or AF-Multimer v2.3

For each of the 300 candidates entering a round:

**AF3 invocation** (for nanobody + small-molecule hapten):
```bash
python /opt/alphafold3/run_alphafold.py \
  --json_path <input_spec.json> \
  --model_dir /data/af3_models/ \
  --output_dir data/results/round_XX/af3_predictions/ \
  --num_diffusion_samples 5 \
  --num_seeds 3
```

The `input_spec.json` for AF3 includes:
```json
{
  "name": "nanobody_pdg_candidate_001",
  "modelSeeds": [42, 123, 456],
  "sequences": [
    {
      "protein": {
        "id": "A",
        "sequence": "<NANOBODY_SEQUENCE>"
      }
    }
  ],
  "ligands": [
    {
      "id": "B",
      "sdf": "<PdG_SDF_CONTENTS>"
    }
  ]
}
```

**AF-Multimer v2.3 invocation** (fallback, for nanobody + protein-conjugate):
```bash
python /opt/alphafold/run_alphafold.py \
  --fasta_paths <nanobody_target.fasta> \
  --model_preset multimer \
  --num_multimer_predictions_per_model 1 \
  --max_template_date 2025-01-01 \
  --num_recycle 12 \
  --output_dir data/results/round_XX/af_predictions/ \
  --use_gpu_relax
```

**Compute budget per round**: 300 candidates × 5 models × 3 seeds = 4,500 predictions.
At ~15 GPU-min each on A100: ~1,125 GPU-hours per round. Parallelize across cluster.

### 3b. Interface Analysis & Scoring

**Tools**: BioPython + FreeSASA + Rosetta + custom scripts

For each predicted complex:

1. **Extract confidence metrics** from AF output:
   - `ipTM`: interface predicted TM-score (from AF ranking output)
   - `pLDDT` at interface: mean pLDDT of residues within 5Å of the ligand

2. **Shape complementarity** (Lawrence-Colman Sc):
   ```python
   # Using Rosetta's InterfaceAnalyzer
   from pyrosetta import *
   from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
   
   analyzer = InterfaceAnalyzerMover("A_B")
   analyzer.set_compute_interface_sc(True)
   analyzer.apply(pose)
   sc = analyzer.get_interface_sc()
   ```

3. **Binding energy** (Rosetta ΔΔG or FoldX):
   ```bash
   # Rosetta InterfaceAnalyzer
   /opt/rosetta/bin/InterfaceAnalyzer.linuxgccrelease \
     -s complex.pdb \
     -interface A_B \
     -compute_packstat \
     -out:file:score_only score.sc
   ```
   Key metric: `dG_separated` (binding energy in REU).

4. **Buried surface area**:
   ```python
   import freesasa
   
   structure = freesasa.Structure("complex.pdb")
   result_complex = freesasa.calc(structure)
   # Calculate BSA = SASA_A + SASA_B - SASA_AB
   ```

5. **Composite scoring**:
   ```python
   from nanolfa.scoring.composite import CompositeScorer
   
   scorer = CompositeScorer(config="configs/scoring.yaml")
   score = scorer.score(
       iptm=0.82,
       plddt_interface=85.3,
       shape_complementarity=0.71,
       binding_energy=-35.2,
       buried_surface_area=450.0,
       developability_score=0.85
   )
   # Returns: normalized composite score ∈ [0, 1]
   ```

### 3c. CDR Diversification

**Tool**: ProteinMPNN + directed mutagenesis

Take the top 50 candidates from 3b. For each, generate variants:

**Strategy varies by round:**

| Round | Temperature | Variants/candidate | CDRs redesigned | Strategy |
|-------|------------|-------------------|----------------|----------|
| 1 | 0.2 | 10 | CDR3 only | Explore CDR3 sequence space |
| 2 | 0.15 | 8 | CDR3 + CDR2 | Refine CDR3, explore CDR2 |
| 3 | 0.10 | 6 | CDR3 + CDR2 + CDR1 | Fine-tune all CDRs |
| 4 | 0.05 | 4 | All CDRs | Polish; point mutations only |
| 5 | 0.05 | 4 | All CDRs | Final optimization |

Additional diversification approaches per round:
- **Rounds 1–2**: ProteinMPNN redesign of entire CDR loops.
- **Rounds 3–4**: single-point saturation mutagenesis at top interface contacts.
- **Round 5**: combinatorial mutations at the 3–5 most impactful positions
  (identified from correlation of mutations with score improvements across rounds).

### 3d. Re-Prediction & Ranking

Repeat step 3a on all new variants. Score with 3b. Rank by composite score.
Select top-K for next round (K=50 for rounds 1–4; K=20 for round 5 → wet lab).

### 3e. Developability Filtering

**Tools**: Aggrescan3D + CamSol + custom sequence analysis

Applied as hard gates before advancing candidates:

```python
from nanolfa.filters.developability import DevelopabilityFilter

dev_filter = DevelopabilityFilter(config)

for candidate in ranked_candidates:
    report = dev_filter.evaluate(candidate)
    
    # Hard rejections
    if report.aggregation_score > 0.5:
        candidate.status = "REJECT"
    if report.cdr3_net_charge > 2 or report.cdr3_net_charge < -2:
        candidate.status = "REJECT"
    if report.liability_count > 3:
        candidate.status = "REJECT"
    if report.predicted_tm < 65.0:
        candidate.status = "REJECT"
    
    # Soft penalties (reduce composite score)
    if report.cdr_hydrophobicity > 0.55:
        candidate.composite_score *= 0.95
    if report.liability_count > 0:
        candidate.composite_score *= (1 - 0.02 * report.liability_count)
```

Sequence liability scan:
```python
liabilities = {
    "deamidation": ["NG", "NS", "NT", "NH"],
    "isomerization": ["DG", "DS", "DT"],
    "oxidation": ["M"],           # only if solvent-exposed
    "glycosylation": ["N[^P][ST]"],  # regex: N-X-S/T where X≠P
    "clipping": ["DP"],
}
```

---

## Phase 4 — Cross-Reactivity Screening

### 4.1 In Silico Specificity Panel

For each of the top 50 candidates after the final design round:

1. Run AF3 (or AF-Multimer) against every compound in the cross-reactivity panel.
2. Score each off-target complex with the same composite scoring function.
3. Compute selectivity ratio: `S_on_target / max(S_off_target)`.

**Decision rules:**
- If off-target ipTM > 0.50: **reject** (predicted to bind off-target).
- If selectivity ratio < 2.0: **reject** (insufficient discrimination).
- If off-target ΔG_bind < −10 REU: **flag** for experimental confirmation.

### 4.2 Negative Design (Optional Rescue)

For borderline candidates that bind one specific off-target:

1. Identify the off-target contact residues in the predicted complex.
2. Run ProteinMPNN with **negative design**: penalize amino acids that form
   favorable contacts with the off-target while maintaining on-target contacts.
3. Re-predict and re-score both on-target and off-target complexes.
4. Accept only if on-target score drops < 5% AND off-target score improves
   (increases, i.e., weaker binding) by > 20%.

---

## Phase 5 — LFA-Specific Optimization

### 5.1 Kinetic Accessibility Scoring

**Tool**: custom analysis from AF structures

LFA test lines require fast binding. Estimate relative kon from:

1. **Binding pocket openness**: calculate the solid angle subtended by the
   binding pocket entrance. Wider entrance → faster association.
   ```python
   from nanolfa.lfa.kinetics import estimate_pocket_accessibility
   
   access_score = estimate_pocket_accessibility(
       complex_pdb="complex.pdb",
       ligand_chain="B",
       receptor_chain="A",
       probe_radius=1.4  # Å (water radius)
   )
   ```

2. **Electrostatic steering**: compute the electrostatic potential at the
   pocket entrance using APBS. Complementary charge → faster association.

3. **Conformational flexibility of CDR loops**: high B-factors (or low pLDDT)
   in the binding loops may indicate conformational selection mechanism
   (slower) vs. lock-and-key (faster).

### 5.2 Conjugation Orientation Modeling

**Tool**: custom geometric analysis

Nanobodies on LFA must bind antigen while conjugated to gold nanoparticles
(detection) or immobilized on nitrocellulose (capture).

For gold conjugation via C-terminal His6 tag:
1. Model the His6 tag as a flexible linker extending from the C-terminus.
2. Place the nanoparticle surface at the tag endpoint.
3. Verify that the paratope (CDR loops) faces AWAY from the nanoparticle
   surface — i.e., the angle between the paratope centroid–nanobody centroid
   vector and the C-terminus–nanoparticle vector should be > 90°.

```python
from nanolfa.lfa.orientation import check_conjugation_clearance

clearance = check_conjugation_clearance(
    nanobody_pdb="nanobody.pdb",
    tag_location="C_terminal",
    nanoparticle_radius_nm=20,  # 40nm diameter / 2
    paratope_residues=cdr_residues
)
# Returns: angle (degrees), minimum distance (Å), pass/fail
```

### 5.3 Thermal Stability Prediction

**Tool**: ESM-IF + Rosetta ddG

Predict the apparent melting temperature (Tm) to ensure shelf life:

1. Compute Rosetta total energy of the monomer (indicator of fold stability).
2. Use ESM inverse folding perplexity as a proxy for designability/stability.
3. Flag candidates with computed Tm < 65°C.

---

## Phase 6 — Experimental Feedback Integration

### 6.1 Data Ingestion

After wet-lab characterization of the top 20 candidates:

```python
from nanolfa.core.pipeline import ingest_experimental_data

results = ingest_experimental_data(
    spr_data="data/experimental/spr_kinetics.csv",
    # Columns: candidate_id, kon, koff, KD, chi2
    thermal_data="data/experimental/thermal_shift.csv",
    # Columns: candidate_id, Tm_celsius, Tagg_celsius
    lfa_data="data/experimental/lfa_signal.csv",
    # Columns: candidate_id, test_line_intensity, control_line_intensity,
    #          signal_noise_ratio, detection_limit_ng_ml
)
```

### 6.2 Scoring Function Recalibration

Correlate computational predictions with experimental measurements:

1. Fit linear models: ipTM vs. log(KD), ΔG_bind vs. log(KD), etc.
2. Identify which computational metrics are most predictive of LFA performance.
3. Update scoring weights in `configs/scoring.yaml` based on multivariate
   regression coefficients.
4. If sufficient data (≥20 data points), train a Gaussian Process model for
   Bayesian optimization of the scoring function.

### 6.3 Re-Ranking and Iteration

With recalibrated scoring:
1. Re-score all candidates from previous rounds.
2. Identify any candidates that were computationally ranked low but would score
   high under the new calibration ("hidden gems").
3. Optionally, launch additional design rounds with the improved scoring function.

---

## Appendix A — Compute Requirements Summary

| Phase | Tool | GPU-hours/target | Wall time (4× A100) |
|-------|------|-----------------|---------------------|
| 1 | RDKit/xTB | ~1 (CPU) | 1 hour |
| 2 | RFdiffusion + ProteinMPNN + ESMFold | ~50 | 6 hours |
| 3 (per round) | AF-Multimer/AF3 + Rosetta | ~1,200 | 3 days |
| 3 (5 rounds) | — | ~6,000 | 15 days |
| 4 | AF3 (cross-reactivity panel) | ~500 | 2 days |
| 5 | Custom analysis | ~10 | 2 hours |
| **Total** | | **~6,560** | **~18 days** |

## Appendix B — Key Literature

- Jumper et al. (2021) Nature 596:583 — AlphaFold2
- Abramson et al. (2024) Nature 630:493 — AlphaFold3
- Dauparas et al. (2022) Science 378:49 — ProteinMPNN
- Watson et al. (2023) Nature 620:1089 — RFdiffusion
- Lin et al. (2023) Science 379:1123 — ESMFold
- Muyldermans (2013) Annu Rev Biochem 82:775 — Nanobodies
- Mitchell & Colwell (2018) Front Immunol 9:3 — VHH for diagnostics
