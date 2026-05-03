# Quarantined External CI

These files came from upstream Matplotlib CI integrations and are kept here
only as inert provenance. They must not live at their service-discovery paths.

## Quarantined entrypoints

- `circleci/config.yml` was `.circleci/config.yml`; CircleCI would build the
  upstream documentation tree and run `deploy-docs.sh`.
- `circleci/deploy-docs.sh` pushed generated documentation to
  `matplotlib/devdocs`.
- `circleci/fetch_doc_logs.py` was used by upstream CircleCI/GitHub workflow
  glue to retrieve Sphinx logs.
- `azure-pipelines.yml` was the upstream Azure Pipelines configuration.

## Active CI

Only `.github/workflows/tests.yml` is intentionally left in the active workflow
directory for this port.
