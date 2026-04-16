# Contributing to NanoLFA-Design

Thanks for your interest in contributing. This document describes how to
propose changes, what standards the codebase follows, and what to expect
during review.

---

## Scope of contributions

NanoLFA-Design is a research pipeline, not a general-purpose library. Contributions
that align well with the project include:

- New scoring metrics with clear structural or kinetic justification
- Additional developability filters (aggregation, stability, liability motifs)
- Support for new target analytes (small molecules, protein hormones, peptides)
- Wrappers for new structure-prediction or sequence-design models
- Performance improvements for the iterative loop
- Experimental calibration data ingestion (SPR, BLI, LFA signal)
- Documentation, tutorials, and notebook examples

Contributions that are usually out of scope:

- Generic antibody tooling unrelated to lateral flow use cases
- Format conversions or utilities already well-covered by Biopython/RDKit
- Experimental features without published or in-house validation data

If you are unsure whether something fits, open an issue before starting work.

---

## Development setup

```bash
git clone https://github.com/DoctorDean/nanobody-lfa-design.git
cd nanobody-lfa-design
mamba env create -f environment.yml
conda activate nanolfa
pip install -e ".[dev]"
pre-commit install          # optional but recommended
```

Run the full check suite before opening a PR:

```bash
make check                   # lint + typecheck + tests
```

---

## Branching and pull requests

1. **Fork** the repository and create a feature branch from `develop`:
   ```bash
   git checkout -b feature/my-improvement develop
   ```
2. **Keep branches focused**: one logical change per PR. Large PRs that mix
   refactoring, new features, and bug fixes are hard to review and will be
   asked to split.
3. **Rebase, do not merge**: keep your branch clean with
   `git rebase develop` before pushing updates.
4. **Write descriptive commits**: follow the
   [Conventional Commits](https://www.conventionalcommits.org/) format.
   Examples:
   - `feat(scoring): add APBS electrostatic steering term`
   - `fix(mpnn): correct CDR3 position offset for VHH numbering`
   - `docs(protocol): clarify AF3 ligand input format`
   - `test(filters): add edge case for all-Cys sequence`
5. **Open a pull request** against `develop` (not `main`). The
   `main` branch is reserved for tagged releases.

---

## Code standards

### Style

- **Formatting**: `ruff format` (configured in `pyproject.toml`)
- **Linting**: `ruff check` with the rule set defined in `pyproject.toml`
- **Line length**: 100 characters
- **Imports**: sorted by ruff's `I` rules (isort-compatible)

### Types

- **All new public functions must have type hints.**
- `mypy src/nanolfa/` must pass with the settings in `pyproject.toml`.
- Use `from __future__ import annotations` at the top of every module
  to enable forward references without quotes.

### Docstrings

Use Google-style docstrings for all public functions and classes:

```python
def score_complex(self, complex_path: Path) -> dict[str, float]:
    """Extract all raw interface scores from a predicted complex PDB.

    Args:
        complex_path: Path to the relaxed complex PDB file.

    Returns:
        Dictionary of raw metric values keyed by metric name.

    Raises:
        FileNotFoundError: If the complex PDB is missing.
    """
```

### Tests

- **Unit tests are required** for any new scoring function, filter, or
  utility. Integration tests are encouraged for pipeline changes.
- Tests must not require GPU access, AlphaFold databases, Rosetta, or
  FoldX. Mock these heavy dependencies or skip with `pytest.mark.skip`
  and a clear reason.
- Maintain or improve coverage — CI will flag regressions.
- Use the fixtures in `tests/conftest.py` for common test objects
  (sample candidates, minimal configs, VHH sequences).

### Configuration changes

- **New config keys require defaults** in `configs/default.yaml` and
  documentation in `docs/PROTOCOL.md` or `docs/SCORING.md`.
- **Scoring weight changes require justification** in the PR description
  with supporting data (correlation with experimental KD, ROC analysis,
  or literature citation).
- **Threshold changes require a changelog entry** describing the
  rationale and expected impact on candidate selection rates.

---

## Scientific rigor

This pipeline informs real wet-lab decisions, and downstream users (including
this project's maintainers) commit significant experimental resources based on
its output. Contributions are held to scientific as well as software standards:

1. **Cite your sources.** New scoring metrics, thresholds, or algorithms must
   reference the originating literature in code comments and the relevant
   docs file.
2. **Declare limitations.** If a method has known failure modes (e.g., Rosetta
   energy functions for small-molecule interfaces), document them in the
   docstring and in `docs/SCORING.md`.
3. **Separate prediction from calibration.** Do not hard-code experimental
   corrections into predictor modules; use the calibration layer in
   `configs/scoring.yaml`.
4. **Preserve reproducibility.** Any randomness must be seeded and the seed
   exposed via config. Avoid `datetime.now()` or other non-deterministic
   inputs to the design process.

---

## Reviewing and merging

- PRs require at least **one approving review** from a CODEOWNER.
- All CI checks must pass (lint, typecheck, tests, config validation).
- PRs are merged via **squash merge** to keep `develop` history linear.
- Release tags on `main` are created by maintainers after merging a
  batch of PRs into `develop` and running internal validation.

---

## Reporting bugs

Open a GitHub issue using the Bug Report template. Include:

- Pipeline version (`git rev-parse HEAD` or tag)
- Target being run (pdg, e3g, custom)
- Relevant section of the config
- Full traceback
- Minimum steps to reproduce

For scoring anomalies (e.g., a candidate that passes hard gates but clearly
shouldn't), include the candidate sequence, its scores dict, and the
predicted complex PDB if possible.

---

## Security

Do not report security issues via public GitHub issues. Email the
maintainers at the address in `pyproject.toml`.

---

## License

By contributing, you agree that your contributions will be licensed under
the Apache License 2.0, the same license that covers the project.
