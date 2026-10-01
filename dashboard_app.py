"""Interactive, precomputed QUALISCAPES dashboard served with Voilà."""

from collections import defaultdict
from pathlib import Path

import ipywidgets as widgets
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from IPython.display import HTML, clear_output, display


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FIGURES = ROOT / "figures"


def _read(name):
    """Read one packaged result table without requiring source interviews."""
    return pd.read_csv(DATA / name)


def _clean_label(value):
    """Remove ATLAS.ti marker glyphs and common mojibake from display labels."""
    text = str(value)
    for marker in ("●", "○", "â—", "â—‹"):
        text = text.replace(marker, "")
    return " ".join(text.split())


def _colour(label):
    """Use stable sentiment colours while leaving unmarked concepts blue."""
    text = str(label).casefold()
    if text.endswith("negative"):
        return "#d95f5f"
    if text.endswith("positive"):
        return "#4faf83"
    if text.endswith("neutral"):
        return "#d6a84b"
    return "#4c78a8"


def _empty_figure(message):
    """Return an informative Plotly placeholder for an empty filter result."""
    figure = go.Figure()
    figure.add_annotation(text=message, x=.5, y=.5, showarrow=False)
    figure.update_layout(height=520, xaxis_visible=False, yaxis_visible=False)
    return figure


def _sankey_from_paths(paths, value_column, focus=None, top_n=25, title="Discursive pathways"):
    """Build the historical merged-node Sankey from complete path records."""
    if paths.empty:
        return _empty_figure("No complete chains meet the current filters.")

    working = paths.copy()
    if focus:
        working = working[working["path"].map(lambda p: focus in str(p).split(" -> "))]
    if working.empty:
        return _empty_figure("No complete chains contain the selected code.")

    selected = working.nlargest(top_n, value_column)
    values = defaultdict(float)
    path_counts = defaultdict(int)
    nodes = set()

    for row in selected.itertuples(index=False):
        parts = str(row.path).split(" -> ")
        nodes.update(parts)
        amount = float(getattr(row, value_column))
        for source, target in zip(parts, parts[1:]):
            values[source, target] += amount
            path_counts[source, target] += 1

    ordered_nodes = sorted(nodes, key=str.casefold)
    node_index = {label: index for index, label in enumerate(ordered_nodes)}
    ordered_links = sorted(values, key=lambda pair: (pair[0].casefold(), pair[1].casefold()))

    figure = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            label=ordered_nodes,
            pad=24,
            thickness=20,
            color=["#e45756" if label == focus else _colour(label) for label in ordered_nodes],
            line=dict(color="rgba(40,40,40,.4)", width=.6),
            hovertemplate="<b>%{label}</b><br>visible flow: %{value:.3g}<extra></extra>",
        ),
        link=dict(
            source=[node_index[source] for source, target in ordered_links],
            target=[node_index[target] for source, target in ordered_links],
            value=[values[key] for key in ordered_links],
            customdata=[path_counts[key] for key in ordered_links],
            color=["rgba(228,87,86,.52)" if source == focus else "rgba(76,120,168,.27)"
                   for source, target in ordered_links],
            hovertemplate=(
                "%{source.label} → %{target.label}"
                "<br>aggregated flow: %{value:.3g}"
                "<br>displayed chains using link: %{customdata}<extra></extra>"
            ),
        ),
    ))
    figure.update_layout(title=title, height=720, font=dict(size=11), margin=dict(l=20, r=20, t=75, b=20))
    return figure


def _overview_panel():
    """Create lightweight interactive summaries from precomputed tables."""
    code_summary = _read("global_code_summary.csv")
    code_summary["code"] = code_summary["code"].map(_clean_label)
    indicator = _read("indicator_sentiment_composition.csv")
    domains = _read("stop_domain_composition_matrix.csv").set_index("domain")

    view = widgets.ToggleButtons(
        options=[("Code frequencies", "codes"), ("Indicator sentiment", "sentiment"), ("Domains by stop", "domains")],
        value="codes",
        description="View:",
    )
    top_n = widgets.IntSlider(value=20, min=10, max=50, step=5, description="Top:", continuous_update=False)
    output = widgets.Output()

    def render(*_):
        with output:
            clear_output(wait=True)
            if view.value == "codes":
                table = code_summary.nlargest(top_n.value, "code_frequency").sort_values("code_frequency")
                figure = px.bar(
                    table, x="code_frequency", y="code", orientation="h", color="sentiment",
                    color_discrete_map={"negative": "#d95f5f", "positive": "#4faf83", "neutral": "#d6a84b"},
                    labels={"code_frequency": "Occurrences", "code": "Code"},
                    title="Most frequent coded qualities",
                )
            elif view.value == "sentiment":
                table = indicator.nlargest(top_n.value, "total").sort_values("total")
                figure = go.Figure()
                for column, label, colour in (
                    ("share_positive", "Positive", "#4faf83"),
                    ("share_negative", "Negative", "#d95f5f"),
                    ("share_neutral", "Neutral", "#d6a84b"),
                ):
                    figure.add_bar(
                        x=100 * table[column], y=table["indicator"], orientation="h",
                        name=label, marker_color=colour,
                        customdata=table["total"],
                        hovertemplate="%{y}<br>%{x:.1f}%<br>total: %{customdata}<extra></extra>",
                    )
                figure.update_layout(barmode="stack", title="Sentiment composition within indicators", xaxis_title="Share (%)")
            else:
                matrix = domains.drop(columns=["Total"], errors="ignore") * 100
                figure = px.imshow(
                    matrix, aspect="auto", color_continuous_scale="YlGnBu",
                    labels={"x": "Stop", "y": "Domain", "color": "% within stop"},
                    title="Domain composition by stop",
                )
            figure.update_layout(height=650, margin=dict(l=20, r=20, t=65, b=20))
            display(figure)

    for control in (view, top_n):
        control.observe(render, names="value")
    render()
    return widgets.VBox([widgets.HBox([view, top_n]), output])


def _cooccurrence_panel():
    """Create an inspectable co-occurrence ranking with indicator filtering."""
    pairs = _read("global_pairs_all.csv")
    for column in ("code_a", "code_b"):
        pairs[column] = pairs[column].map(_clean_label)

    indicators = sorted(set(pairs["indicator_a"].dropna()).union(pairs["indicator_b"].dropna()), key=str.casefold)
    focus = widgets.Dropdown(options=[("All indicators", None), *[(x, x) for x in indicators]], description="Focus:")
    metric = widgets.Dropdown(
        options=[("Joint count", "cooccurrence"), ("c coefficient", "c_coefficient")],
        value="cooccurrence", description="Rank by:",
    )
    top_n = widgets.IntSlider(value=20, min=10, max=50, step=5, description="Top:", continuous_update=False)
    output = widgets.Output()

    def render(*_):
        with output:
            clear_output(wait=True)
            selected = pairs.copy()
            if focus.value:
                selected = selected[(selected["indicator_a"] == focus.value) | (selected["indicator_b"] == focus.value)]
            selected = selected.nlargest(top_n.value, metric.value).copy()
            if selected.empty:
                display(_empty_figure("No co-occurrence pairs meet the filter."))
                return
            selected["pair"] = selected["code_a"] + " ↔ " + selected["code_b"]
            selected = selected.sort_values(metric.value)
            figure = px.bar(
                selected, x=metric.value, y="pair", orientation="h", color="cooccurrence",
                color_continuous_scale="YlGnBu",
                hover_data=["cooccurrence", "c_coefficient", "gr_a", "gr_b"],
                title="Co-occurrence pairs",
            )
            figure.update_layout(height=max(550, 26 * len(selected) + 180), margin=dict(l=20, r=20, t=65, b=20))
            display(figure)
            display(selected[["pair", "cooccurrence", "c_coefficient", "gr_a", "gr_b"]].sort_values(metric.value, ascending=False))

    for control in (focus, metric, top_n):
        control.observe(render, names="value")
    render()
    return widgets.VBox([widgets.HBox([focus, metric, top_n]), output])


def _transition_panel():
    """Create a merged-node directed Sankey from precomputed transition edges."""
    edges = _read("directed_transitions_all_levels_by_stop.csv")
    levels = [x for x in ("leaf", "indicator", "domain") if x in set(edges["level"])]
    stops = sorted(edges["stop"].dropna().unique(), key=str.casefold)

    level = widgets.Dropdown(options=levels, value="indicator" if "indicator" in levels else levels[0], description="Level:")
    stop = widgets.Dropdown(options=[("All stops", None), *[(x, x) for x in stops]], description="Stop:")
    metric = widgets.Dropdown(
        options=[("Raw occurrences", "raw_transition_count"), ("Fractional weight", "weight")],
        value="raw_transition_count", description="Flow:",
    )
    focus = widgets.Dropdown(options=[("All codes", None)], description="Focus:", layout=widgets.Layout(width="380px"))
    top_n = widgets.IntSlider(value=35, min=10, max=100, step=5, description="Top links:", continuous_update=False)
    output = widgets.Output()

    def selected_edges():
        selected = edges[edges["level"] == level.value].copy()
        if stop.value:
            selected = selected[selected["stop"] == stop.value]
        keys = ["source_code", "target_code"]
        selected = selected.groupby(keys, as_index=False).agg(
            weight=("weight", "sum"),
            raw_transition_count=("raw_transition_count", "sum"),
            documents=("stop", "nunique"),
            quotation_pairs=("unique_quotation_pairs", "sum"),
        )
        return selected

    def refresh_focus(*_):
        selected = selected_edges()
        nodes = sorted(set(selected["source_code"]).union(selected["target_code"]), key=str.casefold)
        previous = focus.value
        focus.options = [("All codes", None), *[(x, x) for x in nodes]]
        focus.value = previous if previous in nodes else None

    def render(*_):
        with output:
            clear_output(wait=True)
            selected = selected_edges()
            if focus.value:
                selected = selected[(selected["source_code"] == focus.value) | (selected["target_code"] == focus.value)]
            selected = selected.nlargest(top_n.value, metric.value)
            if selected.empty:
                display(_empty_figure("No directed transitions meet the filters."))
                return
            labels = sorted(set(selected["source_code"]).union(selected["target_code"]), key=str.casefold)
            index = {label: i for i, label in enumerate(labels)}
            figure = go.Figure(go.Sankey(
                arrangement="snap",
                node=dict(
                    label=labels, pad=22, thickness=19,
                    color=["#e45756" if label == focus.value else _colour(label) for label in labels],
                ),
                link=dict(
                    source=[index[x] for x in selected["source_code"]],
                    target=[index[x] for x in selected["target_code"]],
                    value=selected[metric.value],
                    customdata=np.stack([selected["documents"], selected["quotation_pairs"]], axis=-1),
                    color="rgba(76,120,168,.28)",
                    hovertemplate=(
                        "%{source.label} → %{target.label}<br>flow: %{value:.3g}"
                        "<br>stops: %{customdata[0]}<br>quotation pairs: %{customdata[1]}<extra></extra>"
                    ),
                ),
            ))
            scope = stop.value or "all stops"
            figure.update_layout(
                title=f"Directed discourse transitions — {level.value} — {scope}",
                height=720, margin=dict(l=20, r=20, t=70, b=20),
            )
            display(figure)
            display(selected[["source_code", "target_code", metric.value, "documents", "quotation_pairs"]])

    for control in (level, stop):
        control.observe(refresh_focus, names="value")
    for control in (level, stop, metric, focus, top_n):
        control.observe(render, names="value")
    refresh_focus()
    render()
    return widgets.VBox([widgets.HBox([level, stop, metric]), widgets.HBox([focus, top_n]), output])


def _pathway_panel():
    """Explore complete consecutive chains of two, three, or four states."""
    cache = {}
    for level in ("leaf", "indicator", "domain"):
        for length in (2, 3, 4):
            cache[level, length] = _read(f"recurrent_paths_{level}_length_{length}.csv")

    level = widgets.Dropdown(options=["leaf", "indicator", "domain"], value="indicator", description="Level:")
    length = widgets.ToggleButtons(options=[2, 3, 4], value=3, description="States:")
    metric = widgets.Dropdown(
        options=[("Raw occurrences", "raw_count"), ("Fractional weight", "weighted_count"), ("Source support", "unique_sources")],
        value="raw_count", description="Flow:",
    )
    focus = widgets.Dropdown(options=[("All codes", None)], description="Focus:", layout=widgets.Layout(width="390px"))
    top_n = widgets.IntSlider(value=25, min=5, max=75, step=5, description="Top chains:", continuous_update=False)
    output = widgets.Output()

    def frame():
        return cache[level.value, length.value]

    def refresh_focus(*_):
        paths = frame()
        nodes = sorted({node for path in paths["path"] for node in str(path).split(" -> ")}, key=str.casefold)
        previous = focus.value
        focus.options = [("All codes", None), *[(x, x) for x in nodes]]
        focus.value = previous if previous in nodes else None

    def render(*_):
        with output:
            clear_output(wait=True)
            paths = frame().copy()
            figure = _sankey_from_paths(
                paths, metric.value, focus.value, top_n.value,
                f"Complete {length.value}-state chains — {level.value} level",
            )
            display(figure)
            if focus.value:
                paths = paths[paths["path"].map(lambda p: focus.value in str(p).split(" -> "))]
            columns = ["path", metric.value, "raw_count", "weighted_count", "unique_sources"]
            columns = list(dict.fromkeys(column for column in columns if column in paths))
            display(paths.nlargest(top_n.value, metric.value)[columns])

    for control in (level, length):
        control.observe(refresh_focus, names="value")
    for control in (level, length, metric, focus, top_n):
        control.observe(render, names="value")
    refresh_focus()
    render()
    return widgets.VBox([widgets.HBox([level, length, metric]), widgets.HBox([focus, top_n]), output])


def _figure_gallery():
    """Display the publication figures generated by the analytical notebook."""
    files = sorted(FIGURES.glob("fig*.png"))
    options = [(path.stem.replace("_", " ").title(), path) for path in files]
    selector = widgets.Dropdown(options=options, description="Figure:", layout=widgets.Layout(width="620px"))
    image = widgets.Image(format="png", layout=widgets.Layout(max_width="100%"))

    def render(*_):
        image.value = selector.value.read_bytes()

    selector.observe(render, names="value")
    render()
    return widgets.VBox([selector, image])


def _methods_panel():
    """Expose compact quality and inferential summaries without recomputation."""
    quality = _read("quotation_data_quality_summary.csv")
    permutation = _read("transition_permutation_tests.csv")
    significant = permutation[permutation["significant_fdr"].fillna(False)].copy()
    nodes = _read("transition_node_metrics.csv")

    output = widgets.Output()
    with output:
        display(HTML(
            "<p><b>Important:</b> transitions and chains describe ordered coded discourse. "
            "They do not demonstrate causality or a participant's internal reasoning.</p>"
        ))
        display(HTML("<h4>Data-quality summary</h4>"))
        display(quality)
        display(HTML(f"<h4>FDR-significant directed transitions ({len(significant)})</h4>"))
        display(significant.sort_values(["fdr_q", "observed_weight"]).head(30))
        display(HTML("<h4>Leading first-order network nodes</h4>"))
        display(nodes.nlargest(20, "pagerank")[["code", "pagerank", "directed_betweenness", "in_strength", "out_strength"]])
    return output


def launch_dashboard():
    """Assemble and display the complete code-hidden public dashboard."""
    header = widgets.HTML(
        """
        <div style="padding:18px 22px;border-radius:12px;background:#f3f6fa;margin-bottom:14px">
          <h1 style="margin:0 0 8px 0">QUALISCAPES — Minusio internal dashboard</h1>
          <p style="margin:0">Precomputed descriptive and inferential results with interactive views.
          No source interview files are included in this deployment.</p>
        </div>
        """
    )
    tabs = widgets.Tab(children=[
        _overview_panel(),
        _cooccurrence_panel(),
        _transition_panel(),
        _pathway_panel(),
        _figure_gallery(),
        _methods_panel(),
    ])
    for index, title in enumerate((
        "Overview", "Co-occurrences", "Directed flows", "Full chains", "Report figures", "Methods & checks"
    )):
        tabs.set_title(index, title)
    display(widgets.VBox([header, tabs], layout=widgets.Layout(width="100%")))

