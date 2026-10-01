# QUALISCAPES Internal Dashboard

This repository contains the public, code-hidden Voilà dashboard for the
precomputed QUALISCAPES Minusio results. The shared notebook preserves every
Markdown cell and every saved output from the main analytical notebook, in the
same order. Original analytical code is removed from this public copy and its
cells are tagged `skip-execution`, so the saved results are displayed without
being recomputed. A final live section provides cloud-backed widgets using
packaged result tables.

[Launch the complete interactive dashboard](https://mybinder.org/v2/gh/alemaisano/qualiscapes-internal-dashboard/HEAD?urlpath=voila%2Frender%2Fqualiscapes_full_results.ipynb)

Visitors need only a browser. Binder installs the environment and runs the
dashboard automatically on cloud infrastructure; no local Python or notebook
execution is required. The first build can take several minutes.

The deployment intentionally excludes the original ATLAS.ti quotation exports,
Word documents, and project files. The compact `qualiscapes_internal_dashboard`
notebook remains available as a results-only alternative.
