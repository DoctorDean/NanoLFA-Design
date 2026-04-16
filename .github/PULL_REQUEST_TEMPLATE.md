<!--
Thanks for contributing to NanoLFA-Design.
Please fill out the sections below. Delete any that don't apply.
-->

## Summary

<!-- One or two sentences describing what this PR does and why. -->


## Type of change

- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that changes existing behavior)
- [ ] Documentation update
- [ ] Refactoring / code quality improvement
- [ ] Scoring function or threshold change (see scientific checklist below)
- [ ] Dependency update

## Related issues

<!-- Link issues with "Closes #123" or "Fixes #456". -->


## Changes

<!-- Bullet list of the specific changes made. Be concrete:
     - Added `X` function in `src/nanolfa/Y.py`
     - Changed default weight for ipTM from 0.20 to 0.25
     - Updated docs/SCORING.md with rationale for new threshold
-->


## Testing

<!-- How did you verify this works? -->

- [ ] New unit tests added
- [ ] Existing tests still pass (`make test`)
- [ ] Linting passes (`make lint`)
- [ ] Type checking passes (`make typecheck`)
- [ ] Config validation passes
- [ ] Manually verified on [target: pdg | e3g | other]

### Test coverage

<!-- Brief description of what the new tests cover. -->


## Scientific justification

<!-- REQUIRED for scoring weight changes, threshold changes, new metrics,
     or changes to filter logic. -->

### Rationale

<!-- Why is this change needed scientifically? -->


### Supporting evidence

<!-- Literature citations, internal validation data, benchmark results,
     correlation coefficients, etc. Attach or link supporting plots. -->


### Expected impact

<!-- How will this change the output of the pipeline?
     - Will it accept more or fewer candidates?
     - Does it change ranking order of candidates from previous rounds?
     - Are there candidates that would now pass that previously failed? -->


## Configuration changes

<!-- List any new config keys or changed defaults. -->

- [ ] No config changes
- [ ] New config keys added (list below)
- [ ] Default values changed (list below with old → new)
- [ ] Config documentation updated in `docs/`

```yaml
# List additions/changes here
```

## Breaking changes

<!-- If this is a breaking change, describe what users need to do to migrate. -->

- [ ] Not a breaking change
- [ ] Breaking change — migration notes below:


## Documentation

- [ ] README updated (if user-facing behavior changed)
- [ ] PROTOCOL.md updated (if computational procedure changed)
- [ ] SCORING.md updated (if scoring logic changed)
- [ ] CHANGELOG.md updated with entry under `[Unreleased]`
- [ ] Docstrings added/updated for all new public functions

## Reviewer checklist

<!-- For the reviewer, not the submitter. -->

- [ ] Code is readable and well-documented
- [ ] Tests adequately cover the changes
- [ ] Scientific justification is sound (if applicable)
- [ ] No hard-coded values that should be in config
- [ ] No breaking changes without migration path
- [ ] Changelog entry is clear and accurate
