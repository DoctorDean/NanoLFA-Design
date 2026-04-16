# Scoring Function Documentation

## Overview

The NanoLFA-Design pipeline ranks nanobody candidates using a weighted composite score
that integrates six orthogonal metrics spanning structural confidence, physical
chemistry, and manufacturability. The composite score maps to [0, 1], where higher
is better.

## Composite Score Formula

```
S = Σ(wi · normalize(mi))    for i in {1..6}

where:
  w = weight vector (sums to 1.0)
  m = raw metric value
  normalize(m) = clip((m - m_min) / (m_max - m_min), 0, 1)
```

For metrics where lower is better (e.g., binding energy), the normalization is
inverted: `normalize(m) = clip((m_max - m) / (m_max - m_min), 0, 1)`.

## Component Metrics

### 1. ipTM — Interface Predicted TM-Score (w = 0.25)

**Source**: AlphaFold-Multimer / AlphaFold 3 output
**Range**: 0.0–1.0 (0.4–0.95 after normalization clipping)
**Rationale**: ipTM is the strongest single predictor of whether two chains
actually interact. For antibody–antigen complexes, ipTM > 0.75 shows strong
correlation with true binding. For nanobody–hapten complexes, the expected range
is somewhat compressed (0.5–0.85) because small-molecule interfaces are smaller.

**Normalization**: min=0.4, max=0.95

### 2. pLDDT at Interface (w = 0.20)

**Source**: AlphaFold per-residue confidence
**Range**: 0–100 (50–95 after normalization clipping)
**Rationale**: Low pLDDT at the binding interface suggests the model is uncertain
about the conformation of the CDR loops in the bound state — a strong negative
signal. Interface pLDDT > 80 is associated with well-defined, rigid binding modes
which are desirable for fast on-rate kinetics in LFA.

**Calculation**: Mean pLDDT of all residues (on either chain) with any heavy atom
within 5.0 Å of any heavy atom on the partner chain.

**Normalization**: min=50.0, max=95.0

### 3. Shape Complementarity — Sc (w = 0.15)

**Source**: Lawrence-Colman shape correlation statistic (via Rosetta InterfaceAnalyzer)
**Range**: 0.0–1.0 (0.4–0.8 after clipping)
**Rationale**: Sc measures the geometric match between the two interacting surfaces.
Antibody–antigen interfaces typically show Sc = 0.64–0.68 (Lawrence & Colman, 1993).
For hapten-binding antibodies, where the antigen is small and fits into a pocket,
Sc can be higher (0.7–0.8) when the pocket is well-formed.

**Normalization**: min=0.40, max=0.80

### 4. Binding Energy — ΔG_bind (w = 0.20)

**Source**: Rosetta InterfaceAnalyzer `dG_separated` or FoldX `Interaction_Energy`
**Range**: Typically −60 to +10 REU (or kcal/mol for FoldX)
**Rationale**: Physics-based binding energy is orthogonal to AF confidence metrics.
It captures van der Waals packing, hydrogen bonding, desolvation, and electrostatics
at the interface. Rosetta's `ref2015` score function is well-calibrated for
protein–protein interfaces; less validated for protein–small molecule, so we use
this with appropriate skepticism for hapten targets.

**Protocol**: 3× constrained relax of the complex, then InterfaceAnalyzer with
`dG_separated` averaged across replicates.

**Normalization**: min=−50.0 (strong binder), max=0.0 (non-binder), INVERTED.

### 5. Buried Surface Area — BSA (w = 0.10)

**Source**: FreeSASA or Rosetta InterfaceAnalyzer
**Range**: 200–900 Å² (varies by target size)
**Rationale**: BSA is a simple geometric measure of interface size. For nanobody–hapten
complexes, the BSA is inherently limited by the hapten size (300–600 Å² typical for
steroid glucuronides). We include it with lower weight as a sanity check — candidates
with very low BSA are unlikely to achieve meaningful affinity.

**Target-specific normalization**:
- PdG: min=200, max=600 Å²
- E3G: min=180, max=550 Å²

### 6. Developability Meta-Score (w = 0.10)

**Source**: Composite of sequence-based and structure-based filters
**Range**: 0.0–1.0

Sub-components (equally weighted within this meta-score):
- **Aggregation propensity**: 1 - normalized Aggrescan3D score
- **Charge balance**: 1 if CDR3 net charge in [−2, +2], linearly decaying outside
- **Hydrophobicity**: 1 - normalized CDR mean hydrophobicity (Kyte-Doolittle)
- **Sequence liabilities**: 1 - (liability_count / max_liability_count)
- **Predicted Tm**: normalized, min=50°C, max=85°C

## Decision Thresholds

### Hard Gates (must pass ALL)

These are binary pass/fail criteria applied before composite scoring:

| Gate | Threshold | Rationale |
|------|-----------|-----------|
| ipTM | ≥ 0.60 | Below this, AF has low confidence in the interaction |
| pLDDT (interface) | ≥ 65 | Below this, CDR loop conformation is unreliable |
| ΔG_bind | ≤ −15 REU | Minimum energetic favorability for binding |
| Aggregation score | ≤ 0.50 | Above this, high risk of aggregation in LFA conjugate |
| Sequence liabilities | ≤ 3 | More than 3 liabilities = high manufacturing risk |

### Tiered Classification

Candidates passing hard gates are classified into tiers:

| Tier | Composite Score | Action |
|------|----------------|--------|
| Green (Advance) | ≥ 0.70 | Promote to next round; priority for wet lab |
| Yellow (Borderline) | 0.50–0.70 | Keep in pool; may advance if pool is thin |
| Red (Reject) | < 0.50 | Discard |

## Calibration Notes

### Pre-Experimental (v0.1)

The initial weights and thresholds are based on:
- Published benchmarks of AF-Multimer accuracy for antibody–antigen complexes
- Rosetta energy function validation studies
- Expert judgment from antibody engineering literature
- Conservative estimates (we err toward false negatives over false positives,
  since wet-lab capacity is the bottleneck)

### Post-Experimental (v0.2+)

After Phase 6 experimental data:
1. Compute Pearson/Spearman correlation of each metric with experimental KD.
2. Reweight using ridge regression coefficients.
3. Adjust thresholds based on ROC analysis (optimize for recall at 80% precision).
4. If ≥30 data points: train a Gaussian Process surrogate model as a drop-in
   replacement for the linear composite.

## Limitations and Caveats

1. **ipTM is not KD**: ipTM measures structural prediction confidence, not binding
   affinity directly. A high-confidence wrong structure still scores well.
2. **Rosetta energy for haptens**: Rosetta's protein score functions are less
   validated for small-molecule interactions. FoldX may perform better; we average both.
3. **Additive assumption**: the composite score assumes metrics are independent.
   In reality, ipTM and pLDDT are partially correlated. PCA-based scoring or
   learned ensembles may improve in v1.0.
4. **Kinetics are not scored directly**: binding affinity (KD) does not predict
   kinetics (kon, koff). Phase 5 adds kinetic heuristics, but these are approximate.
