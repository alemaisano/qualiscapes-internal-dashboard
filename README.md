# QUALISCAPES Internal Dashboard

This repository contains the public, code-hidden Voilà dashboard for the
precomputed QUALISCAPES Minusio results. The shared notebook preserves every
Markdown cell and every saved output from the main analytical notebook, in the
same order. Original analytical code is removed from this public copy. Each
former analysis cell now only republishes its own saved output, so results stay
in their original position without being recomputed. A final live section
provides cloud-backed widgets using packaged result tables.

[Launch the complete interactive dashboard](https://mybinder.org/v2/gh/alemaisano/qualiscapes-internal-dashboard/HEAD?urlpath=voila%2Frender%2Fqualiscapes_full_results.ipynb)

Visitors need only a browser. Binder installs the environment and runs the
dashboard automatically on cloud infrastructure; no local Python or notebook
execution is required. The first build can take several minutes.

The live dashboard appears near the top and includes interactive overview,
co-occurrence ranking, co-occurrence network, directed Sankey, directed graph,
complete-chain, figure-gallery, and methods panels. Historical widget model IDs
are not reused because their original kernels no longer exist; the public copy
links those locations back to the corresponding live cloud controls.

The co-occurrence network tab uses the same explorer function as the main
notebook's stop-network section. Its Network selector contains Global plus all
ten stops, and retains every original analytical, styling, labeling, hover, and
visibility control.

The launch URL follows the repository's `HEAD`, so it remains valid after future
pushes. An already-running Binder session is pinned to its launch commit: reopen
the launch URL to start an updated session after publishing changes.

The deployment intentionally excludes the original ATLAS.ti quotation exports,
Word documents, and project files. The compact `qualiscapes_internal_dashboard`
notebook remains available as a results-only alternative.
