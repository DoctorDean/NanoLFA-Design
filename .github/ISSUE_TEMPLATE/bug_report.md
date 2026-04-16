---
name: Bug report
about: Report a problem with the pipeline
title: "[BUG] "
labels: bug, needs-triage
assignees: ''
---

## Description

<!-- A clear description of what went wrong. -->


## Reproduction steps

1.
2.
3.

## Expected behavior

<!-- What did you expect to happen? -->


## Actual behavior

<!-- What actually happened? Include the full traceback if applicable. -->

```
Paste traceback here
```

## Environment

- **Pipeline version**: `<output of git rev-parse HEAD or version tag>`
- **Target**: pdg | e3g | custom
- **Python version**: `<python --version>`
- **CUDA version**: `<nvidia-smi | head -5>`
- **OS**: Ubuntu 22.04 / CentOS 7 / other
- **GPU**: A100 80GB / V100 32GB / other

## Configuration

<!-- Paste the relevant portion of your config YAML. REDACT any paths
     that contain sensitive information or internal hostnames. -->

```yaml
# paste relevant config section
```

## Candidate / round context (if applicable)

<!-- For scoring anomalies, include the candidate ID, its sequence, and
     the full scores dict. -->

- **Round**:
- **Candidate ID**:
- **Sequence**:
- **Scores**:
  ```
  ipTM:                 
  pLDDT (interface):    
  shape_complementarity:
  binding_energy:       
  buried_surface_area:  
  developability:       
  composite:            
  ```

## Additional context

<!-- Logs, screenshots of unexpected structures, plots, or anything else
     that would help diagnose the issue. -->
