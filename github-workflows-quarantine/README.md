# Quarantined Upstream Workflows

These GitHub Actions workflow files originate from the upstream
[matplotlib/matplotlib](https://github.com/matplotlib/matplotlib) repository.
They were moved here from `.github/workflows/` to prevent them from running
on this fork (MLXPorts/mlx-matplotlib).

## Provenance

- **Upstream repo:** matplotlib/matplotlib
- **Fork point:** `ea40d72fb0` (Merge pull request #30657)
- **Upstream HEAD at quarantine time:** `08fe8bc4ad` (Merge pull request #31111)
- **Quarantine date:** 2026-02-11

## Why quarantined

These workflows are designed for the upstream matplotlib project and cause
problems when running on a fork:

| File | Trigger | Problem |
|---|---|---|
| `nightlies.yml` | cron (daily) | Attempts to upload wheels to Anaconda Cloud; fails because `ANACONDA_TOKEN` is not set. Would publish to `scientific-python-nightly-wheels` if it were. |
| `stale.yml` | cron (3x/week) | Labels fork PRs as inactive on upstream's schedule. |
| `stale-tidy.yml` | cron (3x/week) | Closes fork issues as inactive on upstream's schedule. |
| `cibuildwheel.yml` | push to main | Builds release wheels on every push — expensive CI minutes for a dev fork. |
| `cygwin.yml` | push to main | Runs Cygwin build/test matrix — not relevant for MLX porting work. |
| `circleci.yml` | status events | Requires `CIRCLECI_TOKEN` secret that doesn't exist on the fork. |
| `good-first-issue.yml` | issue labeled | Posts upstream community onboarding text on fork issues. |
| `pr_welcome.yml` | PR opened | Posts upstream contributor greeting on fork PRs. |
| `labeler.yml` | PR opened | Auto-labels PRs using upstream's label rules. |

## Modifications from upstream

All files (except `labeler.yml` which was unmodified) had minimal changes from
upstream — primarily swapping `matplotlib/matplotlib` guards to
`MLXPorts/mlx-matplotlib` and pinning some action versions. The actual workflow
logic is upstream's.

## Restoring

To re-enable any of these workflows, move the file back to `.github/workflows/`:

```sh
git mv github-workflows-quarantine/<file>.yml .github/workflows/<file>.yml
```

## Workflows still active in `.github/workflows/`

The following workflows remain active because they are useful for development:

- `tests.yml` — test suite on push/PR
- `linting.yml` — linting on PRs
- `codeql-analysis.yml` — security scanning
- `mypy-stubtest.yml` — type checking on PRs
- `clean_pr.yml` — PR cleanliness checks
- `do_not_merge.yml` — prevents merging WIP PRs
- `conflictcheck.yml` — merge conflict detection
