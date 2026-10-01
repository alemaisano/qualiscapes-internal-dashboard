"""Interactive, precomputed QUALISCAPES dashboard served with Voilà."""

from collections import defaultdict
import json
from pathlib import Path

import ipywidgets as widgets
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from IPython.display import HTML, clear_output, display

# Reuse the exact Global/stop explorer copied from the analytical notebook.
from notebook_network_explorer import interactive_network_explorer


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


def _network_positions(graph, layout_name):
    """Return deterministic two-dimensional positions for an interactive graph."""
    # Circular placement is useful when users want to compare all nodes equally.
    if layout_name == "Circular":
        return nx.circular_layout(graph)
    # Kamada-Kawai emphasizes graph distance and often separates communities well.
    if layout_name == "Kamada-Kawai":
        return nx.kamada_kawai_layout(graph, weight="weight")
    # A fixed seed makes the default spring layout stable across widget changes.
    return nx.spring_layout(graph, weight="weight", seed=17, k=None)


def _scaled_sizes(values, low=13, high=42):
    """Scale non-negative node strengths into a readable marker-size range."""
    # Convert arbitrary numeric iterables into one floating-point array.
    array = np.asarray(list(values), dtype=float)
    # Return an empty array when the filtered graph contains no nodes.
    if array.size == 0:
        return array
    # Compress large differences so a few dominant nodes do not hide the rest.
    array = np.sqrt(np.maximum(array, 0))
    # Use one middle size when every node has the same strength.
    if np.nanmax(array) == np.nanmin(array):
        return np.full(array.size, (low + high) / 2)
    # Map the observed range linearly onto the requested visual range.
    return low + (array - np.nanmin(array)) * (high - low) / (np.nanmax(array) - np.nanmin(array))


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


def _simplified_cooccurrence_network_panel():
    """Retain the compact network implementation for internal fallback use."""
    # Load only precomputed global and stop-level edges; no analysis is rerun.
    global_edges = _read("global_pairs_all.csv")
    stop_edges = _read("stop_edges_long.csv")
    # Remove export marker glyphs from every code shown in the browser.
    for frame in (global_edges, stop_edges):
        for column in ("code_a", "code_b"):
            frame[column] = frame[column].map(_clean_label)

    # Let users switch between the complete corpus and one stop/document group.
    stops = sorted(stop_edges["stop"].dropna().unique(), key=str.casefold)
    scope = widgets.Dropdown(
        options=[("Global", None), *[(name, name) for name in stops]],
        description="Scope:",
        layout=widgets.Layout(width="430px"),
    )
    # A focal code isolates its immediate undirected neighbourhood.
    focus = widgets.Dropdown(
        options=[("All codes", None)],
        description="Focus:",
        layout=widgets.Layout(width="430px"),
    )
    # The edge slider prevents a dense network from becoming unreadable.
    top_n = widgets.IntSlider(
        value=40, min=10, max=100, step=5,
        description="Top edges:", continuous_update=False,
    )
    # Users can compare complementary graph layouts without recalculating data.
    layout = widgets.Dropdown(
        options=["Spring", "Kamada-Kawai", "Circular"],
        value="Spring", description="Layout:",
    )
    # Keep redraws inside one replaceable output area.
    output = widgets.Output()

    def selected_edges():
        """Return the currently selected precomputed co-occurrence table."""
        # Global scope uses the complete matrix-derived pair table.
        if scope.value is None:
            return global_edges.copy()
        # Stop scope uses only edges assigned to that named stop.
        return stop_edges[stop_edges["stop"].eq(scope.value)].copy()

    def refresh_focus(*_):
        """Update focal-code choices whenever the selected scope changes."""
        # Collect every source or target code in the selected edge table.
        edges = selected_edges()
        nodes = sorted(set(edges["code_a"]).union(edges["code_b"]), key=str.casefold)
        # Preserve the previous choice when it also exists in the new scope.
        previous = focus.value
        focus.options = [("All codes", None), *[(node, node) for node in nodes]]
        focus.value = previous if previous in nodes else None

    def render(*_):
        """Redraw the graph and its exact-value table from current controls."""
        # Replace the preceding view rather than appending repeated figures.
        with output:
            clear_output(wait=True)
            # Rank first so the graph stays responsive even at global scale.
            edges = selected_edges().nlargest(top_n.value, "cooccurrence").copy()
            # A focal selection retains only edges touching the chosen code.
            if focus.value:
                edges = edges[
                    edges["code_a"].eq(focus.value)
                    | edges["code_b"].eq(focus.value)
                ]
            # Explain empty combinations instead of displaying a blank canvas.
            if edges.empty:
                display(_empty_figure("No co-occurrence edges meet the current filters."))
                return

            # Construct the displayed graph from the already-ranked edges.
            graph = nx.Graph()
            sentiment = {}
            for row in edges.itertuples(index=False):
                weight = float(row.cooccurrence)
                graph.add_edge(row.code_a, row.code_b, weight=weight)
                sentiment[row.code_a] = getattr(row, "sentiment_a", None)
                sentiment[row.code_b] = getattr(row, "sentiment_b", None)
            # Calculate deterministic coordinates for the selected layout.
            positions = _network_positions(graph, layout.value)
            # Start with a clean Plotly figure and draw one trace per edge.
            figure = go.Figure()
            for source, target, attributes in graph.edges(data=True):
                x0, y0 = positions[source]
                x1, y1 = positions[target]
                weight = float(attributes["weight"])
                figure.add_trace(go.Scatter(
                    x=[x0, x1], y=[y0, y1], mode="lines",
                    line=dict(width=max(1, min(8, .8 + np.sqrt(weight))), color="rgba(110,110,110,.42)"),
                    hoverinfo="skip", showlegend=False,
                ))
            # Weighted degree controls node size and is also shown on hover.
            nodes = list(graph.nodes())
            strengths = np.array([graph.degree(node, weight="weight") for node in nodes], dtype=float)
            sizes = _scaled_sizes(strengths)
            figure.add_trace(go.Scatter(
                x=[positions[node][0] for node in nodes],
                y=[positions[node][1] for node in nodes],
                mode="markers+text",
                text=nodes,
                textposition="top center",
                customdata=np.stack([strengths, [graph.degree(node) for node in nodes]], axis=-1),
                marker=dict(
                    size=sizes,
                    color=["#e45756" if node == focus.value else _colour(sentiment.get(node)) for node in nodes],
                    line=dict(color="white", width=1.2),
                ),
                hovertemplate="<b>%{text}</b><br>weighted degree: %{customdata[0]:.3g}<br>visible neighbours: %{customdata[1]}<extra></extra>",
                showlegend=False,
            ))
            # Label the current scope explicitly to prevent overgeneralization.
            title_scope = scope.value or "global corpus"
            figure.update_layout(
                title=f"Interactive co-occurrence network — {title_scope}",
                height=760,
                margin=dict(l=20, r=20, t=75, b=20),
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                hovermode="closest",
            )
            # Display both the graph and the numerical evidence behind it.
            display(figure)
            display(edges[["code_a", "code_b", "cooccurrence"]].sort_values("cooccurrence", ascending=False))

    # Scope changes also require rebuilding the focal-code list.
    scope.observe(refresh_focus, names="value")
    # Every visible control triggers a redraw using cached CSV results.
    for control in (scope, focus, top_n, layout):
        control.observe(render, names="value")
    # Populate controls and draw the initial global view immediately.
    refresh_focus()
    render()
    # Arrange controls above the replaceable graph output.
    return widgets.VBox([
        widgets.HBox([scope, layout, top_n]),
        focus,
        output,
    ])


def _load_notebook_network_analyses():
    """Rebuild the notebook explorer inputs from packaged aggregate results."""
    # Read the exact Global and stop-specific summaries exported by the pipeline.
    payload = json.loads((DATA / "notebook_network_analyses.json").read_text(encoding="utf-8"))
    # Convert JSON records back to the two DataFrames used by the notebook function.
    analyses = {}
    for analysis_name, values in payload.items():
        # Preserve original code labels, frequencies, matrices, and validity flags.
        analyses[analysis_name] = {
            "code_summary": pd.DataFrame(values["code_summary"]),
            "pairs": pd.DataFrame(values["pairs"]),
            "codes": values["codes"],
            "code_frequency": np.asarray(values["code_frequency"], dtype=float),
            "matrix": np.asarray(values["matrix"], dtype=float),
            "code_frequency_valid": bool(values["code_frequency_valid"]),
        }
    # Return the mapping in its deterministic Global-then-stops JSON order.
    return analyses


def _cooccurrence_network_panel():
    """Embed the exact Global/stop network explorer from the main notebook."""
    # Load one Global analysis and all ten stop analyses from derived public data.
    analyses = _load_notebook_network_analyses()
    # Remove Global before passing the remaining named analyses as stop choices.
    global_analysis = analyses.pop("Global")
    # Build the original controls and FigureWidget without displaying them twice.
    controls, graph = interactive_network_explorer(
        global_analysis,
        analyses,
        default_analysis="Global",
        display_widget=False,
    )
    # Explain that this is the literal notebook explorer, not a reduced dashboard view.
    note = widgets.HTML(
        "<p><b>Exact notebook network explorer.</b> The Network dropdown contains "
        "Global and every stop. All analytical and display controls from the original "
        "notebook cell are retained.</p>"
    )
    # Place the unchanged controls immediately above their live Plotly graph.
    return widgets.VBox([note, controls, graph], layout=widgets.Layout(width="100%"))


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


def _directed_graph_panel():
    """Explore transitions as a force-directed network with visible arrows."""
    # Load the precomputed transition table shared with the Sankey panel.
    edges = _read("directed_transitions_all_levels_by_stop.csv")
    # Expose every hierarchy level and stop already present in the results.
    levels = [value for value in ("leaf", "indicator", "domain") if value in set(edges["level"])]
    stops = sorted(edges["stop"].dropna().unique(), key=str.casefold)
    # Define controls for hierarchy, document scope, weighting, and layout.
    level = widgets.Dropdown(options=levels, value="indicator" if "indicator" in levels else levels[0], description="Level:")
    stop = widgets.Dropdown(options=[("All stops", None), *[(name, name) for name in stops]], description="Stop:", layout=widgets.Layout(width="420px"))
    metric = widgets.Dropdown(options=[("Raw occurrences", "raw_transition_count"), ("Fractional weight", "weight")], value="raw_transition_count", description="Weight:")
    layout = widgets.Dropdown(options=["Spring", "Kamada-Kawai", "Circular"], value="Spring", description="Layout:")
    focus = widgets.Dropdown(options=[("All codes", None)], description="Focus:", layout=widgets.Layout(width="430px"))
    top_n = widgets.IntSlider(value=45, min=10, max=100, step=5, description="Top arrows:", continuous_update=False)
    self_links = widgets.Checkbox(value=True, description="Show self-links")
    # Reserve one output region for redraws and the supporting value table.
    output = widgets.Output()

    def selected_edges():
        """Aggregate the selected precomputed transitions across stops."""
        # Restrict the table to the selected hierarchy level first.
        selected = edges[edges["level"].eq(level.value)].copy()
        # Apply an optional individual-stop filter.
        if stop.value:
            selected = selected[selected["stop"].eq(stop.value)]
        # Merge repeated source-target rows while retaining both weight measures.
        return selected.groupby(["source_code", "target_code"], as_index=False).agg(
            weight=("weight", "sum"),
            raw_transition_count=("raw_transition_count", "sum"),
            stops=("stop", "nunique"),
            quotation_pairs=("unique_quotation_pairs", "sum"),
        )

    def refresh_focus(*_):
        """Refresh the focal-node selector for the chosen level and stop."""
        selected = selected_edges()
        nodes = sorted(set(selected["source_code"]).union(selected["target_code"]), key=str.casefold)
        previous = focus.value
        focus.options = [("All codes", None), *[(node, node) for node in nodes]]
        focus.value = previous if previous in nodes else None

    def render(*_):
        """Draw the current directed neighbourhood and its exact edge values."""
        with output:
            clear_output(wait=True)
            # Rank transitions using the user-selected descriptive measure.
            selected = selected_edges().nlargest(top_n.value, metric.value).copy()
            # Optionally remove self-continuation edges before graph construction.
            if not self_links.value:
                selected = selected[~selected["source_code"].eq(selected["target_code"])]
            # A focal selection shows both incoming and outgoing relationships.
            if focus.value:
                selected = selected[
                    selected["source_code"].eq(focus.value)
                    | selected["target_code"].eq(focus.value)
                ]
            if selected.empty:
                display(_empty_figure("No directed transitions meet the current filters."))
                return

            # Build a directed graph, retaining the selected metric as edge weight.
            graph = nx.DiGraph()
            for row in selected.itertuples(index=False):
                graph.add_edge(row.source_code, row.target_code, weight=float(getattr(row, metric.value)))
            # Layout algorithms use an undirected projection for stable placement.
            positions = _network_positions(graph.to_undirected(), layout.value)
            figure = go.Figure()
            annotations = []
            # Draw non-self edges as lines and add arrowheads separately.
            for source, target, attributes in graph.edges(data=True):
                x0, y0 = positions[source]
                x1, y1 = positions[target]
                weight = float(attributes["weight"])
                if source == target:
                    # A loop symbol beside the node makes repeated references visible.
                    figure.add_trace(go.Scatter(
                        x=[x0 + .035], y=[y0 + .035], mode="text", text=["↻"],
                        textfont=dict(size=24, color="#7a5195"),
                        hovertext=[f"{source} → {target}<br>{metric.label}: {weight:.3g}"],
                        hoverinfo="text", showlegend=False,
                    ))
                    continue
                figure.add_trace(go.Scatter(
                    x=[x0, x1], y=[y0, y1], mode="lines",
                    line=dict(width=max(1, min(7, .8 + np.sqrt(max(weight, 0)))), color="rgba(76,120,168,.38)"),
                    hoverinfo="skip", showlegend=False,
                ))
                # Place an arrowhead near the target while leaving its label clear.
                annotations.append(dict(
                    ax=x0, ay=y0, x=x0 + .88 * (x1 - x0), y=y0 + .88 * (y1 - y0),
                    xref="x", yref="y", axref="x", ayref="y",
                    showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.2,
                    arrowcolor="rgba(76,120,168,.58)", text="",
                ))
            # Size nodes by combined incoming and outgoing selected strength.
            nodes = list(graph.nodes())
            strengths = np.array([
                graph.in_degree(node, weight="weight") + graph.out_degree(node, weight="weight")
                for node in nodes
            ], dtype=float)
            sizes = _scaled_sizes(strengths)
            figure.add_trace(go.Scatter(
                x=[positions[node][0] for node in nodes],
                y=[positions[node][1] for node in nodes],
                mode="markers+text", text=nodes, textposition="top center",
                customdata=np.stack([
                    strengths,
                    [graph.in_degree(node) for node in nodes],
                    [graph.out_degree(node) for node in nodes],
                ], axis=-1),
                marker=dict(
                    size=sizes,
                    color=["#e45756" if node == focus.value else _colour(node) for node in nodes],
                    line=dict(color="white", width=1.2),
                ),
                hovertemplate="<b>%{text}</b><br>total strength: %{customdata[0]:.3g}<br>incoming arrows: %{customdata[1]}<br>outgoing arrows: %{customdata[2]}<extra></extra>",
                showlegend=False,
            ))
            # State explicitly that arrows represent sequence, not causality.
            title_scope = stop.value or "all stops"
            figure.update_layout(
                title=f"Directed discourse network — {level.value} — {title_scope}<br><sup>Arrow direction is observed coded order, not causality.</sup>",
                height=780,
                margin=dict(l=20, r=20, t=95, b=20),
                annotations=annotations,
                xaxis=dict(visible=False), yaxis=dict(visible=False),
                hovermode="closest",
            )
            display(figure)
            display(selected[["source_code", "target_code", metric.value, "stops", "quotation_pairs"]].sort_values(metric.value, ascending=False))

    # Hierarchy and stop changes also alter the available focal nodes.
    for control in (level, stop):
        control.observe(refresh_focus, names="value")
    # Every control updates the displayed graph without rerunning analysis.
    for control in (level, stop, metric, layout, focus, top_n, self_links):
        control.observe(render, names="value")
    refresh_focus()
    render()
    return widgets.VBox([
        widgets.HBox([level, stop, metric]),
        widgets.HBox([layout, top_n, self_links]),
        focus,
        output,
    ])


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
        <div id="qualiscapes-live-dashboard" style="padding:18px 22px;border-radius:12px;background:#f3f6fa;margin-bottom:14px">
          <h1 style="margin:0 0 8px 0">QUALISCAPES — Minusio internal dashboard</h1>
          <p style="margin:0">Precomputed descriptive and inferential results with interactive views.
          No source interview files are included in this deployment.</p>
        </div>
        """
    )
    tabs = widgets.Tab(children=[
        _overview_panel(),
        _cooccurrence_panel(),
        _cooccurrence_network_panel(),
        _transition_panel(),
        _directed_graph_panel(),
        _pathway_panel(),
        _figure_gallery(),
        _methods_panel(),
    ])
    for index, title in enumerate((
        "Overview",
        "Co-occurrence pairs",
        "Co-occurrence network",
        "Directed Sankey",
        "Directed graph",
        "Full chains",
        "Report figures",
        "Methods & checks",
    )):
        tabs.set_title(index, title)
    display(widgets.VBox([header, tabs], layout=widgets.Layout(width="100%")))
