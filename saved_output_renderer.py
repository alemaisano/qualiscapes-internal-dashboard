"""Replay the notebook's saved outputs without rerunning its analysis.

Voilà deliberately clears stored outputs before executing a notebook.  The
public notebook therefore calls :func:`replay_saved_outputs` in place of every
original analysis cell.  This module reads the already-saved MIME bundles from
the notebook file and publishes them back to the browser in their original
order.  No analytical calculation or source-data access happens here.
"""

# Keep the parsed notebook in memory so the 35 MB file is read only once per
# visitor session, rather than once for every output-bearing cell.
from functools import lru_cache

# Resolve the notebook reliably even when Voilà starts from another directory.
from pathlib import Path

# Preserve stdout and stderr streams as distinct notebook output channels.
import sys

# Read the notebook with Jupyter's own schema-aware parser.
import nbformat

# Publish complete saved MIME bundles (HTML, PNG, Plotly, text, and widgets).
from IPython.display import display


# The public notebook itself remains the authoritative store for every saved
# output, so no second large results file can silently drift out of sync.
NOTEBOOK_PATH = Path(__file__).with_name("qualiscapes_full_results.ipynb")


@lru_cache(maxsize=1)
def _saved_notebook():
    """Load and cache the on-disk notebook that contains the saved results."""

    # Parse the notebook at v4 so output dictionaries have a stable structure.
    return nbformat.read(NOTEBOOK_PATH, as_version=4)


def replay_saved_outputs(cell_index):
    """Publish every saved output belonging to one original code cell."""

    # Find the cell by its immutable original index.  This keeps replay correct
    # even though the live dashboard is deliberately placed near the top.
    cell = next(
        candidate
        for candidate in _saved_notebook().cells
        if candidate.get("metadata", {}).get("qualiscapes_original_index") == cell_index
    )

    # Track whether this cell has already explained a historical widget output.
    widget_notice_shown = False

    # Re-emit outputs in their original order so multi-part results stay intact.
    for output in cell.get("outputs", []):
        # Identify how Jupyter originally represented this particular output.
        output_type = output.get("output_type")

        # Stream outputs are plain stdout or stderr text, such as progress notes.
        if output_type == "stream":
            # Send stderr back to stderr; all other streams use standard output.
            stream = sys.stderr if output.get("name") == "stderr" else sys.stdout
            # Write the stored text verbatim and avoid adding an extra newline.
            stream.write(output.get("text", ""))
            # Flush immediately so output order is retained by the kernel.
            stream.flush()
            # Continue because stream records do not contain a MIME bundle.
            continue

        # Rich display and expression results both store a complete MIME bundle.
        if output_type in {"display_data", "execute_result"}:
            # Saved widget model IDs belonged to the author's old kernel and
            # cannot retain Python callbacks in a new cloud session.  Rendering
            # those IDs would create controls that look active but do nothing.
            if "application/vnd.jupyter.widget-view+json" in output.get("data", {}):
                # Show one transparent explanation per original widget cell.
                if not widget_notice_shown:
                    display(
                        {
                            "text/html": (
                                "<div style='padding:10px 12px;border-left:4px solid #4c78a8;"
                                "background:#f3f6fa;margin:8px 0'>"
                                "<b>Live replacement available.</b> This historical widget used "
                                "a kernel that no longer exists. Its fully interactive replacement "
                                "is in the <a href='#qualiscapes-live-dashboard'>dashboard at the top</a>."
                                "</div>"
                            )
                        },
                        raw=True,
                    )
                    # Suppress repeated notices for paired controls/output models.
                    widget_notice_shown = True
                else:
                    # Preserve the output slot without publishing a dead widget model.
                    display({"text/html": ""}, raw=True)
                # Continue because the stale widget MIME bundle must not be sent.
                continue

            # Publish the original HTML, PNG, Plotly, and text fallbacks.
            display(
                dict(output.get("data", {})),
                raw=True,
                metadata=dict(output.get("metadata", {})),
            )
            # Continue after the saved rich output has been sent to the browser.
            continue

        # Preserve the one saved exception as an output, without raising it again.
        if output_type == "error":
            # Prefer the complete stored traceback because it includes context.
            traceback = output.get("traceback", [])
            # Fall back to the exception name and value if no traceback was saved.
            message = "\n".join(traceback) or (
                f"{output.get('ename', 'Error')}: {output.get('evalue', '')}"
            )
            # Match notebook error-channel behavior while allowing rendering to go on.
            sys.stderr.write(message + "\n")
            # Flush before the next saved output is published.
            sys.stderr.flush()
