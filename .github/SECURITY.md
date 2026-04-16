# Security Policy

## Reporting a vulnerability

Security issues affecting NanoLFA-Design should **not** be reported via public
GitHub issues. Instead, please email the maintainers directly at:

**dean.shez @ gmail.com**

Include:

- A description of the vulnerability
- Steps to reproduce
- The affected version (commit hash or tag)
- Any proof-of-concept code (if applicable)

You should expect an acknowledgement within **3 business days** and a more
detailed response within **10 business days** outlining the next steps.

## Scope

This policy covers vulnerabilities in:

- The NanoLFA-Design codebase itself
- The CI/CD configuration
- Dependency version pins that introduce known CVEs

It does not cover vulnerabilities in upstream dependencies (AlphaFold, ProteinMPNN,
Rosetta, PyTorch, etc.) — those should be reported to the respective projects.

## Supported versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | ✅         |
| < 0.1   | ❌         |

## Disclosure policy

Once a fix is available, we will coordinate public disclosure with the reporter.
We follow a standard 90-day disclosure window unless a shorter or longer period
is agreed upon based on severity and complexity.

## Scientific data integrity

Because this pipeline informs wet-lab decisions, we treat issues that could
cause silently incorrect scoring, ranking, or filtering as security-relevant
even when they don't involve traditional security threats. Examples include:

- Bugs that cause hard gates to be silently bypassed
- Config parsing issues that lead to wrong weights being applied
- Race conditions in batch scoring that produce non-deterministic results
- Scoring function regressions introduced without changelog entries

Please report these through the same channel described above.
