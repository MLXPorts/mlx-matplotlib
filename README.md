# mlx-matplotlib

**An MLX-first port of [Matplotlib](https://matplotlib.org/) — plotting with Python, accelerated on Apple Silicon via [MLX](https://github.com/ml-explore/mlx).**

> This repository is a fork of [matplotlib/matplotlib](https://github.com/matplotlib/matplotlib).
> It is **not** the upstream Matplotlib project. For the canonical, production-ready
> plotting library, please use upstream Matplotlib. See [Upstream references](#upstream-references)
> at the end of this document.

---

## About this fork

`mlx-matplotlib` is part of the [MLXPorts](https://github.com/MLXPorts) effort
to bring widely-used scientific Python libraries to a first-class MLX backend
on Apple Silicon. The goal is to keep the familiar `pyplot` / `Figure` / `Axes`
API while progressively replacing NumPy-bound numeric paths with
[MLX](https://github.com/ml-explore/mlx) arrays and kernels where it matters
for performance on Apple GPUs and the Neural Engine.

### MLX-first philosophy

This port is **MLX-first**, not MLX-as-an-afterthought:

- **MLX arrays are first-class inputs.** Plotting functions are intended to
  accept `mlx.core.array` directly — without a forced detour through NumPy —
  wherever the upstream API accepts array-likes.
- **Stay on-device when possible.** Reductions, transforms, colormap
  evaluations, and other vectorized numeric work used by the rendering
  pipeline are migrated to MLX so that data prepared on the GPU does not have
  to round-trip through host memory just to draw a chart.
- **Lazy / unified-memory aware.** The port respects MLX's lazy evaluation
  model and Apple Silicon's unified memory; we avoid eager `.numpy()` /
  `np.asarray(...)` conversions in hot paths.
- **Upstream API compatibility.** Existing Matplotlib code should keep
  working. NumPy inputs are still supported. MLX is an additional, preferred
  path — not a replacement for the public API.
- **Apple Silicon is the primary target.** CI, benchmarks, and the default
  development environment assume `arm64` macOS with MLX available. Other
  platforms are best-effort.

### What has changed vs. upstream

This fork is currently in an early porting / infrastructure phase. The
high-level changes from upstream Matplotlib are:

- **Repository identity.** Project metadata, badges, and CI are scoped to
  `MLXPorts/mlx-matplotlib` rather than `matplotlib/matplotlib`.
- **CI surface trimmed for a fork.** Upstream's release, nightly-wheel,
  Cygwin, stale-bot, labeler, contributor-greeter, CircleCI, Azure Pipelines,
  and AppVeyor integrations are not appropriate for a downstream fork and
  have been moved aside. See
  [`github-workflows-quarantine/README.md`](github-workflows-quarantine/README.md)
  and [`ci-quarantine/README.md`](ci-quarantine/README.md) for the full list
  and rationale. Only the core test workflow (`.github/workflows/tests.yml`)
  is left active.
- **MLX integration work.** Adding MLX as a supported array backend across
  the numeric paths used by the rendering pipeline. This work is ongoing —
  see open pull requests and issues for current status.

### Fork point

- **Upstream repository:** [matplotlib/matplotlib](https://github.com/matplotlib/matplotlib)
- **Forked from upstream commit:** `ea40d72fb0` (Merge pull request #30657)
- **Upstream HEAD at fork-time snapshot:** `08fe8bc4ad` (Merge pull request #31111)

## Install

> ⚠️ This fork is under active development and is **not** published to PyPI or
> conda-forge. Install from source if you want to try it. For a stable
> plotting library, install upstream `matplotlib` instead.

For development, the upstream conda environment file still applies:

```sh
conda env create -f environment.yml
conda activate mpl-dev
pip install --verbose --no-build-isolation --editable ".[dev]"
```

You will additionally need [MLX](https://github.com/ml-explore/mlx) installed
to exercise the MLX-backed code paths:

```sh
pip install mlx
```

## Contribute

Contributions specific to the MLX port (new MLX-backed code paths,
benchmarks, Apple Silicon CI improvements, bug reports against this fork)
are very welcome — please open an issue or pull request on
[`MLXPorts/mlx-matplotlib`](https://github.com/MLXPorts/mlx-matplotlib).

Contributions to **core Matplotlib behavior** that are not specific to MLX or
Apple Silicon should generally go to
[upstream Matplotlib](https://github.com/matplotlib/matplotlib) instead, so
that the wider community benefits and we can pull the changes back in via
merges from upstream.

## Maintainer

- **Sydney Renee** &lt;<sydney@solace.ofharmony.ai>&gt; — [@sydneyrenee](https://github.com/sydneyrenee)

## Acknowledgements

Enormous thanks to John D. Hunter (1968–2012), who created Matplotlib, and to
the hundreds of [Matplotlib
contributors](https://github.com/matplotlib/matplotlib/graphs/contributors)
and the [Matplotlib steering
council](https://matplotlib.org/stable/project/team.html) who have built and
maintained the library for two decades. This port stands entirely on their
work — it is a re-aiming of an excellent existing library at a new hardware
backend, not a from-scratch effort. All credit for the design, API, rendering
architecture, and the bulk of the code in this repository belongs to the
upstream Matplotlib community.

Matplotlib is a [NumFOCUS](https://numfocus.org)-sponsored project; please
consider supporting the upstream project directly.

## Citing

If your work uses plots produced via this port, please **cite upstream
Matplotlib** — not this fork. A ready-made citation entry is available at
<https://matplotlib.org/stable/users/project/citing.html>.

## License

This project inherits Matplotlib's license. See the [`LICENSE/`](LICENSE/)
directory for the full Matplotlib license and the licenses of bundled
third-party components.

---

## Upstream references

All of the following point to the upstream Matplotlib project, which this
fork is derived from. They are the authoritative sources for documentation,
support, and releases of Matplotlib itself.

- **Home page:** <https://matplotlib.org/>
- **Upstream repository:** <https://github.com/matplotlib/matplotlib>
- **Stable documentation:** <https://matplotlib.org/stable/>
- **Installation guide:** <https://matplotlib.org/stable/users/installing/index.html>
- **Contributing guide:** <https://matplotlib.org/devdocs/devel/contribute.html>
- **Citation entry:** <https://matplotlib.org/stable/users/project/citing.html>
- **PyPI:** <https://pypi.org/project/matplotlib/>
- **conda-forge:** <https://anaconda.org/conda-forge/matplotlib>

### Upstream community / discussion

- **Discourse (recommended):** <https://discourse.matplotlib.org/>
- **Gitter (development chat):** <https://gitter.im/matplotlib/matplotlib>
- **Users mailing list:** <https://mail.python.org/mailman/listinfo/matplotlib-users> — <matplotlib-users@python.org>
- **Announcement mailing list:** <https://mail.python.org/mailman/listinfo/matplotlib-announce> — <matplotlib-announce@python.org>
- **Development mailing list:** <https://mail.python.org/mailman/listinfo/matplotlib-devel> — <matplotlib-devel@python.org>

Please **do not** direct upstream Matplotlib questions to this fork's issue
tracker; use the upstream channels above.

### MLX

- **MLX:** <https://github.com/ml-explore/mlx>
- **MLX documentation:** <https://ml-explore.github.io/mlx/>
