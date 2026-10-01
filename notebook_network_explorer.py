"""Exact public copy of the notebook's Global/stop network explorer.

The visualization code below is mechanically sourced from cell 49 of the main
analytical notebook. Only the optional display_widget argument was added so
the same controls can be embedded inside the public dashboard tab.
"""

# Accumulate indicator-level edges without repeated key checks.
from collections import defaultdict

# Build and analyse the same undirected graphs as the notebook.
import networkx as nx

# Scale node and edge metrics numerically.
import numpy as np

# Render the graph as the same interactive Plotly FigureWidget.
import plotly.graph_objects as go

# Create the original dropdowns, sliders, toggles, and widget containers.
import ipywidgets as widgets

# Publish controls when this function is used directly in a notebook.
from IPython.display import display


# Match the initial minimum-weight threshold in chapter 1 of the notebook.
GLOBAL_NETWORK_MIN_COOCCURRENCE = 3

# Keep the exact ordered domain labels used by the original explorer.
DOMAIN_CANDIDATES = [
    "Functionality",
    "Landscape characteristics (physical properties)",
    "Landscape characteristics (Physical properties)",
    "Memory/temporality",
    "Percieved qualities (experiential dimension)",
    "Percieved qualities (expriential dimension)",
    "Place identity",
    "Social dimension",
    "Governance",
]


def strip_symbol(code):
    """Remove the ATLAS.ti leaf marker exactly as the main notebook does."""
    # Strip either filled or empty circle markers plus surrounding whitespace.
    return code.lstrip("●○ ").strip()


def interactive_network_explorer(global_analysis, stop_analyses=None, default_analysis="Global", display_widget=True):
    """Build the interactive Global/stop network explorer widget.

    Structure of this function (it is long because it owns a full interactive UI, not
    because the underlying logic is complex):
      1. lookup tables (sentiment/domain colors, layout algorithms) and small pure helpers;
      2. per-(analysis, level, layout) graph builders that turn one stored analysis dict into
         plotly node/edge traces, caching results so switching a dropdown never recomputes
         an already-seen combination;
      3. the ipywidgets controls (dropdowns/sliders) and one redraw callback wired to all of
         them, which only ever mutates the existing FigureWidget in place.
    Everything here reads already-computed tables (global_analysis / stop_analyses); it never
    reruns statistics, so moving a slider is always fast.
    """
    # Record in `stop_analyses` whether the required condition is satisfied.
    stop_analyses = stop_analyses or {}

    # Stop keys have already been cleaned by load_stop_analyses; retain them verbatim here.
    analyses = {"Global": global_analysis}
    # Merge the newly derived values into `analyses`.
    analyses.update(stop_analyses)
    # Use a separate execution path when `default_analysis not in analyses`.
    if default_analysis not in analyses:
        # Set `default_analysis` to the fixed configuration value `'Global'`.
        default_analysis = "Global"

    # Initialize `sentiment_colors` as the lookup table used by the following analysis.
    sentiment_colors = {
        "positive": "#2CA02C",
        "negative": "#D62728",
        "neutral": "#F2C14E",
        "mixed": "#9467BD",
        None: "#8A8A8A",
        "other": "#8A8A8A",
    }

    # Declare the ordered values stored in `community_palette` so later logic uses one consistent definition.
    community_palette = [
        "#4C78A8", "#F58518", "#E45756", "#72B7B2", "#54A24B",
        "#EECA3B", "#B279A2", "#FF9DA6", "#9D755D", "#BAB0AC",
    ]

    # Define `scaled` to encapsulate the scaled operation for reuse and testing.
    def scaled(values, low, high, sqrt_scale=False):
        # Map any metric to a stable visual range while handling empty or constant arrays.
        values = np.asarray(values, dtype=float)

        # Use a separate execution path when `len(values) == 0`.
        if len(values) == 0:
            # Return the result produced by `np.array`.
            return np.array([])

        # Square-root compression prevents a few very large counts from dominating the view.
        if sqrt_scale:
            # Store the values returned by `np.sqrt` in `values`.
            values = np.sqrt(np.maximum(values, 0))

        # Declare the ordered values stored in `(vmin, vmax)` so later logic uses one consistent definition.
        vmin, vmax = np.nanmin(values), np.nanmax(values)

        # Use a separate execution path when `not np.isfinite(vmin) or not np.isfinite(vmax) or vmax == vmin`.
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax == vmin:
            # Return the result produced by `np.full`.
            return np.full(len(values), (low + high) / 2)

        # Return `low + (values - vmin) * (high - low) / (vmax - vmin)` as the computed result.
        return low + (values - vmin) * (high - low) / (vmax - vmin)

    # Define `dominant_sentiment` to encapsulate the dominant sentiment operation for reuse and testing.
    def dominant_sentiment(group):
        # Weight the label by groundedness; equal maxima are deliberately marked as mixed.
        sentiment_counts = (
            group.groupby("sentiment")["code_frequency"]
            .sum()
            .to_dict()
        )

        # Retrieve the requested value into `positive`, using the documented fallback when absent.
        positive = sentiment_counts.get("positive", 0)
        # Retrieve the requested value into `negative`, using the documented fallback when absent.
        negative = sentiment_counts.get("negative", 0)
        # Retrieve the requested value into `neutral`, using the documented fallback when absent.
        neutral = sentiment_counts.get("neutral", 0)

        # Use `max` to select the largest candidate value, storing the resulting derived value in `maximum`.
        maximum = max(positive, negative, neutral)

        # Use a separate execution path when `maximum == 0`.
        if maximum == 0:
            # Return `None` as the computed result.
            return None

        # Construct `winners` by filtering and transforming the source records in one comprehension.
        winners = [
            label
            for label, value in {
                "positive": positive,
                "negative": negative,
                "neutral": neutral,
            }.items()
            if value == maximum
        ]

        # Return `winners[0] if len(winners) == 1 else 'mixed'` as the computed result.
        return winners[0] if len(winners) == 1 else "mixed"

    # Define `build_leaf_graph` to reconstruct the full leaf-only network before visualization filtering.
    def build_leaf_graph(analysis):
        """Reconstruct the full leaf-only network before visualization filtering."""
        # Construct `G` as the NetworkX graph used for structural analysis.
        G = nx.Graph()

        # Process each `(_, row)` drawn from `analysis['code_summary'].iterrows()`.
        for _, row in analysis["code_summary"].iterrows():
            # Add the selected topology to the working graph with `add_node`.
            G.add_node(
                row["code"],
                code_frequency=float(row["code_frequency"]),
                sentiment=row.get("sentiment"),
                indicator=row.get("indicator"),
                level="leaf",
            )

        # Do not threshold here: the interactive slider applies thresholds without rebuilding.
        for _, row in analysis["pairs"].iterrows():
            # Use `float` to convert the selected value to a floating-point metric, storing the result in `weight`.
            weight = float(row["cooccurrence"])

            # Use a separate execution path when `weight <= 0`.
            if weight <= 0:
                # Ignore this unusable item and proceed to the next iteration.
                continue

            # Use `float` to convert the selected value to a floating-point metric, storing the result in `c_value`.
            c_value = float(row.get("c_coefficient", np.nan))

            # Add the selected topology to the working graph with `add_edge`.
            G.add_edge(
                row["code_a"],
                row["code_b"],
                weight=weight,
                c_coefficient=c_value,
                distance_raw=1 / weight,
            )

        # Retrieve the requested value into `c_valid`, using the documented fallback when absent.
        c_valid = analysis.get("code_frequency_valid", True)
        # Return the assembled collection to the caller.
        return G, c_valid

    # Define `build_indicator_graph` to aggregate positive, negative and neutral leaf codes by base indicator.
    def build_indicator_graph(analysis):
        """
        Aggregate positive, negative and neutral leaf codes by base indicator.

        Raw edge weights are summed across all sentiment combinations.
        Indicator-level C-coefficients are not reconstructed because summing
        leaf groundedness can double-count quotations.
        """
        # Construct `G` as the NetworkX graph used for structural analysis.
        G = nx.Graph()
        # Copy the selected data into `summary` so later transformations do not mutate the input.
        summary = analysis["code_summary"].copy()

        # Process each `(indicator, group)` drawn from `summary.groupby('indicator', dropna=True)`.
        for indicator, group in summary.groupby("indicator", dropna=True):
            # Add the selected topology to the working graph with `add_node`.
            G.add_node(
                indicator,
                code_frequency=float(group["code_frequency"].sum()),
                sentiment=dominant_sentiment(group),
                indicator=indicator,
                level="indicator",
            )

        # Multiple valence-specific leaf edges can contribute to one indicator-level edge.
        aggregated_edges = defaultdict(float)

        # Process each `(_, row)` drawn from `analysis['pairs'].iterrows()`.
        for _, row in analysis["pairs"].iterrows():
            # Retrieve the requested value into `indicator_a`, using the documented fallback when absent.
            indicator_a = row.get("indicator_a")
            # Retrieve the requested value into `indicator_b`, using the documented fallback when absent.
            indicator_b = row.get("indicator_b")

            # Use a separate execution path when `indicator_a is None or indicator_b is None`.
            if indicator_a is None or indicator_b is None:
                # Ignore this unusable item and proceed to the next iteration.
                continue

            # Declare the ordered values stored in `(indicator_a, indicator_b)` so later logic uses one consistent definition.
            indicator_a, indicator_b = str(indicator_a), str(indicator_b)

            # Internal positive/negative variants become part of the same node.
            if indicator_a == indicator_b:
                # Ignore this unusable item and proceed to the next iteration.
                continue

            # Build `key` as the collection needed by the following loop or calculation.
            key = tuple(sorted(
                (indicator_a, indicator_b),
                key=lambda value: value.casefold(),
            ))

            # Accumulate the new contribution into `aggregated_edges[key]`.
            aggregated_edges[key] += float(row["cooccurrence"])

        # Process each `((indicator_a, indicator_b), weight)` drawn from `aggregated_edges.items()`.
        for (indicator_a, indicator_b), weight in aggregated_edges.items():
            # Use a separate execution path when `weight <= 0`.
            if weight <= 0:
                # Ignore this unusable item and proceed to the next iteration.
                continue

            # Add the selected topology to the working graph with `add_edge`.
            G.add_edge(
                indicator_a,
                indicator_b,
                weight=weight,
                c_coefficient=np.nan,
                distance_raw=1 / weight,
            )

        # Return the assembled collection to the caller.
        return G, False

    # Define `build_domain_graph` to build a graph directly from the higher-level domain codes.
    def build_domain_graph(analysis):
        """Build a graph directly from the higher-level domain codes."""
        # Construct `G` as the NetworkX graph used for structural analysis.
        G = nx.Graph()

        # Build `codes` as the collection needed by the following loop or calculation.
        codes = list(analysis["codes"])
        # Store the derived value returned by `np.asarray` in `frequencies`.
        frequencies = np.asarray(analysis["code_frequency"])
        # Compute the comparison matrix `matrix` from the value returned by `np.asarray`.
        matrix = np.asarray(analysis["matrix"])

        # Match configured domains case-insensitively while keeping export spelling.
        lower_lookup = {str(code).casefold(): index for index, code in enumerate(codes)}
        # Set `selected` to `[]` as the derived value used below.
        selected = []
        # Build `seen` as the collection needed by the following loop or calculation.
        seen = set()
        # Process each `candidate` drawn from `DOMAIN_CANDIDATES`.
        for candidate in DOMAIN_CANDIDATES:
            # Retrieve the requested value into `index`, using the documented fallback when absent.
            index = lower_lookup.get(candidate.casefold())
            # Use a separate execution path when `index is not None and index not in seen`.
            if index is not None and index not in seen:
                # Append `index` to the accumulating result list.
                selected.append(index)
                # Call `seen.add` to register the new rule or graph item.
                seen.add(index)

        # Process each `index` drawn from `selected`.
        for index in selected:
            # Select `codes[index]` and store that subset in `domain`.
            domain = codes[index]

            # Add the selected topology to the working graph with `add_node`.
            G.add_node(
                domain,
                code_frequency=float(frequencies[index]),
                sentiment=None,
                indicator=domain,
                level="domain",
            )

        # Process each `(position, index_a)` drawn from `enumerate(selected)`.
        for position, index_a in enumerate(selected):
            # Process each `index_b` drawn from `selected[position + 1:]`.
            for index_b in selected[position + 1:]:
                # Use `float` to convert the selected value to a floating-point metric, storing the result in `weight`.
                weight = float(matrix[index_a, index_b])

                # Use a separate execution path when `weight <= 0`.
                if weight <= 0:
                    # Ignore this unusable item and proceed to the next iteration.
                    continue

                # Use `float` to convert the selected value to a floating-point metric, storing the result in `frequency_a`.
                frequency_a = float(frequencies[index_a])
                # Use `float` to convert the selected value to a floating-point metric, storing the result in `frequency_b`.
                frequency_b = float(frequencies[index_b])
                # At domain level the original marginals remain valid, so C can be recomputed.
                denominator = frequency_a + frequency_b - weight

                # Set `c_value` from the condition-dependent choice `weight / denominator if denominator > 0 else np.nan`.
                c_value = (
                    weight / denominator
                    if denominator > 0
                    else np.nan
                )

                # Add the selected topology to the working graph with `add_edge`.
                G.add_edge(
                    codes[index_a],
                    codes[index_b],
                    weight=weight,
                    c_coefficient=c_value,
                    distance_raw=1 / weight,
                )

        # Retrieve the requested value into `c_valid`, using the documented fallback when absent.
        c_valid = analysis.get("code_frequency_valid", True)
        # Return the assembled collection to the caller.
        return G, c_valid

    # Define `build_level_graphs` to construct the level graphs result from validated inputs.
    def build_level_graphs(analysis):
        # Return the assembled named results as a dictionary for downstream reuse.
        return {
            "Code + sentiment": build_leaf_graph(analysis),
            "Indicator": build_indicator_graph(analysis),
            "Domain": build_domain_graph(analysis),
        }

    # Build each level once, then copy it during rendering.
    level_graphs = {
        analysis_name: build_level_graphs(analysis)
        for analysis_name, analysis in analyses.items()
    }

    # Define `get_layout` to encapsulate the get layout operation for reuse and testing.
    def get_layout(G, layout_name):
        # Use a separate execution path when `G.number_of_nodes() == 0`.
        if G.number_of_nodes() == 0:
            # Return the assembled named results as a dictionary for downstream reuse.
            return {}

        # A fixed seed makes the default force-directed layout reproducible across reruns.
        if layout_name == "Spring":
            # Return the result produced by `nx.spring_layout`.
            return nx.spring_layout(
                G, seed=42, weight="weight", iterations=100
            )

        # Kamada-Kawai expects distances, so strong co-occurrences become short lengths.
        if layout_name == "Kamada-Kawai":
            # Construct `distances` by filtering and transforming the source records in one comprehension.
            distances = {
                u: {
                    v: 1 / max(float(data.get("weight", 1)), 1e-9)
                    for v, data in G[u].items()
                }
                for u in G.nodes
            }
            # Return the result produced by `nx.kamada_kawai_layout`.
            return nx.kamada_kawai_layout(G, dist=distances)

        # Use a separate execution path when `layout_name == 'Circular'`.
        if layout_name == "Circular":
            # Return the result produced by `nx.circular_layout`.
            return nx.circular_layout(G)

        # Use a separate execution path when `layout_name == 'Spectral'`.
        if layout_name == "Spectral":
            # Return the result produced by `nx.spectral_layout`.
            return nx.spectral_layout(G, weight="weight")

        # Return the result produced by `nx.spring_layout`.
        return nx.spring_layout(G, seed=42, weight="weight")

    # Define `calculate_communities` to encapsulate the calculate communities operation for reuse and testing.
    def calculate_communities(G):
        # Use a separate execution path when `G.number_of_nodes() == 0`.
        if G.number_of_nodes() == 0:
            # Return the assembled named results as a dictionary for downstream reuse.
            return {}

        # With no links, assign each node separately because modularity is undefined.
        if G.number_of_edges() == 0:
            # Return `{node: community_id for (community_id, node) in enumerate(G.nodes, start=1)}` as the computed result.
            return {
                node: community_id
                for community_id, node in enumerate(G.nodes, start=1)
            }

        # Build `communities` as the collection needed by the following loop or calculation.
        communities = list(
            nx.community.greedy_modularity_communities(
                G, weight="weight"
            )
        )
        # Order `communities` so the most relevant or stable records appear predictably.
        communities = sorted(communities, key=len, reverse=True)

        # Return the finalized structured result to the caller.
        return {
            node: community_id
            for community_id, community in enumerate(communities, start=1)
            for node in community
        }

    # Define `render` to encapsulate the render operation for reuse and testing.
    def render(
        analysis_name, analysis_level, minimum_edge_weight,
        layout_name, node_size_mode, node_color_mode,
        edge_width_mode, show_nodes, show_edges, show_isolates,
        show_labels, show_edge_values, show_edge_hover,
        label_quantile, edge_opacity,
    ):
        # Select `level_graphs[analysis_name][analysis_level]` and store that subset in `(G, c_valid)`.
        G, c_valid = level_graphs[analysis_name][analysis_level]
        # Work on a copy so widget filtering never mutates the cached source graph.
        G = G.copy()

        # The slider always filters on raw counts, independent of the chosen width encoding.
        edges_to_remove = [
            (u, v)
            for u, v, data in G.edges(data=True)
            if float(data.get("weight", 0)) < minimum_edge_weight
        ]
        # Call `G.remove_edges_from` to apply its documented side effect.
        G.remove_edges_from(edges_to_remove)

        # Use a separate execution path when `not show_isolates`.
        if not show_isolates:
            # Call `G.remove_nodes_from` to apply its documented side effect.
            G.remove_nodes_from(list(nx.isolates(G)))

        # Use a separate execution path when `G.number_of_nodes() == 0`.
        if G.number_of_nodes() == 0:
            # Create the figure object `fig` that will hold this visualization.
            fig = go.Figure()
            # Call `fig.add_annotation` to apply its documented side effect.
            fig.add_annotation(
                text="No nodes remain with the selected filters.",
                x=0.5, y=0.5, showarrow=False, font=dict(size=16),
            )
            # Call `fig.update_layout` to apply the figure-wide layout and axis settings.
            fig.update_layout(
                width=1100, height=700,
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
            )
            # Return `fig` as the computed result.
            return fig

        # Store the derived value returned by `get_layout` in `pos`.
        pos = get_layout(G, layout_name)

        # Build `weighted_degree` as the collection needed by the following loop or calculation.
        weighted_degree = dict(G.degree(weight="weight"))
        # Build `unweighted_degree` as the collection needed by the following loop or calculation.
        unweighted_degree = dict(G.degree())

        # This ratio highlights connectivity beyond what a code's prevalence alone would predict.
        weighted_degree_per_frequency = {}

        # Process each `node` drawn from `G.nodes`.
        for node in G.nodes:
            # Retrieve the requested value into `frequency`, using the documented fallback when absent.
            frequency = G.nodes[node].get("code_frequency", np.nan)

            # Compute `weighted_degree_per_frequency[node]` as the derived value used below.
            weighted_degree_per_frequency[node] = (
                weighted_degree[node] / frequency
                if frequency is not None
                and np.isfinite(frequency)
                and frequency > 0
                else 0.0
            )

        # Betweenness follows the notebook: largest connected component.
        betweenness = {node: 0.0 for node in G.nodes}

        # Construct `components` by filtering and transforming the source records in one comprehension.
        components = [
            component
            for component in nx.connected_components(G)
            if len(component) > 1
        ]

        # Use a separate execution path when `components`.
        if components:
            # Use `max` to select the largest candidate value, storing the resulting derived value in `largest_component`.
            largest_component = max(components, key=len)
            # Copy the selected data into `H` so later transformations do not mutate the input.
            H = G.subgraph(largest_component).copy()

            # Merge the newly derived values into `betweenness`.
            betweenness.update(
                nx.betweenness_centrality(
                    H, weight="distance_raw", normalized=True
                )
            )

        # Compute the community result `community_map` from the value returned by `calculate_communities`.
        community_map = calculate_communities(G)

        # Node size encodes one selected prominence measure; fixed mode removes that encoding.
        if node_size_mode == "Code frequency":
            # Store the derived value returned by `scaled` in `node_sizes`.
            node_sizes = scaled(
                [G.nodes[n].get("code_frequency", 0) for n in G.nodes],
                14, 55, sqrt_scale=True,
            )

        # Use a separate execution path when `node_size_mode == 'Weighted degree'`.
        elif node_size_mode == "Weighted degree":
            # Store the derived value returned by `scaled` in `node_sizes`.
            node_sizes = scaled(
                [weighted_degree[n] for n in G.nodes],
                14, 55, sqrt_scale=True,
            )

        # Use a separate execution path when `node_size_mode == 'Degree'`.
        elif node_size_mode == "Degree":
            # Store the derived value returned by `scaled` in `node_sizes`.
            node_sizes = scaled(
                [unweighted_degree[n] for n in G.nodes],
                14, 55, sqrt_scale=True,
            )

        else:
            # Store the derived value returned by `np.full` in `node_sizes`.
            node_sizes = np.full(G.number_of_nodes(), 24)

        # Categorical modes use fixed palettes; numeric modes use a shared continuous scale.
        node_colorbar_title = None

        # Use a separate execution path when `node_color_mode == 'Sentiment'`.
        if node_color_mode == "Sentiment":
            # Construct `node_colors` by filtering and transforming the source records in one comprehension.
            node_colors = [
                sentiment_colors.get(
                    G.nodes[n].get("sentiment"),
                    sentiment_colors["other"],
                )
                for n in G.nodes
            ]

        # Use a separate execution path when `node_color_mode == 'Community'`.
        elif node_color_mode == "Community":
            # Construct `node_colors` by filtering and transforming the source records in one comprehension.
            node_colors = [
                community_palette[
                    (community_map.get(n, 1) - 1)
                    % len(community_palette)
                ]
                for n in G.nodes
            ]

        # Use a separate execution path when `node_color_mode == 'Degree'`.
        elif node_color_mode == "Degree":
            # Construct `node_colors` by filtering and transforming the source records in one comprehension.
            node_colors = [unweighted_degree[n] for n in G.nodes]
            # Set `node_colorbar_title` to the fixed configuration value `'Degree'`.
            node_colorbar_title = "Degree"

        # Use a separate execution path when `node_color_mode == 'Weighted degree'`.
        elif node_color_mode == "Weighted degree":
            # Construct `node_colors` by filtering and transforming the source records in one comprehension.
            node_colors = [weighted_degree[n] for n in G.nodes]
            # Set `node_colorbar_title` to the fixed configuration value `'Weighted degree'`.
            node_colorbar_title = "Weighted degree"

        # Use a separate execution path when `node_color_mode == 'Weighted degree / code frequency'`.
        elif node_color_mode == "Weighted degree / code frequency":
            # Construct `node_colors` by filtering and transforming the source records in one comprehension.
            node_colors = [
                weighted_degree_per_frequency[n]
                for n in G.nodes
            ]
            # Set `node_colorbar_title` to the fixed configuration value `'Weighted degree / frequency'`.
            node_colorbar_title = "Weighted degree / frequency"

        # Use a separate execution path when `node_color_mode == 'Betweenness'`.
        elif node_color_mode == "Betweenness":
            # Construct `node_colors` by filtering and transforming the source records in one comprehension.
            node_colors = [betweenness[n] for n in G.nodes]
            # Set `node_colorbar_title` to the fixed configuration value `'Betweenness'`.
            node_colorbar_title = "Betweenness"

        else:
            # Calculate `node_colors` from `['#4C78A8'] * G.number_of_nodes()`.
            node_colors = ["#4C78A8"] * G.number_of_nodes()

        # Set `traces` to `[]` as the derived value used below.
        traces = []
        # Build `edge_data` as the collection needed by the following loop or calculation.
        edge_data = list(G.edges(data=True))
        # Construct `raw_edge_values` by filtering and transforming the source records in one comprehension.
        raw_edge_values = [
            float(data.get("weight", 0))
            for _, _, data in edge_data
        ]
        # Construct `c_edge_values` by filtering and transforming the source records in one comprehension.
        c_edge_values = [
            float(data.get("c_coefficient", np.nan))
            for _, _, data in edge_data
        ]

        # Honor C-coefficient widths only where groundedness supports a valid denominator.
        if edge_width_mode == "C-coefficient" and c_valid:
            # Store the display width returned by `np.nan_to_num` in `width_values`.
            width_values = np.nan_to_num(c_edge_values, nan=0)
            # Set `displayed_edge_measure` to the fixed configuration value `'C-coefficient'`.
            displayed_edge_measure = "C-coefficient"

        # Use a separate execution path when `edge_width_mode == 'Fixed'`.
        elif edge_width_mode == "Fixed":
            # Store the display width returned by `np.ones` in `width_values`.
            width_values = np.ones(len(edge_data))
            # Set `displayed_edge_measure` to the fixed configuration value `'fixed'`.
            displayed_edge_measure = "fixed"

        else:
            # Indicator level automatically falls back to raw weight.
            width_values = raw_edge_values
            # Set `displayed_edge_measure` to the fixed configuration value `'raw co-occurrence'`.
            displayed_edge_measure = "raw co-occurrence"

        # Compute the network edge table `edge_widths` from the value returned by `scaled`.
        edge_widths = scaled(
            width_values, 0.8, 8,
            sqrt_scale=displayed_edge_measure == "raw co-occurrence",
        )

        # Declare the ordered values stored in `(edge_mid_x, edge_mid_y)` so later logic uses one consistent definition.
        edge_mid_x, edge_mid_y = [], []
        # Declare the ordered values stored in `(edge_hover_text, edge_value_text)` so later logic uses one consistent definition.
        edge_hover_text, edge_value_text = [], []

        # Plotly needs one line trace per edge to support independently scaled widths.
        if show_edges:
            # Process each `(edge_index, (u, v, data))` drawn from `enumerate(edge_data)`.
            for edge_index, (u, v, data) in enumerate(edge_data):
                # Select `pos[u]` and store that subset in `(x0, y0)`.
                x0, y0 = pos[u]
                # Select `pos[v]` and store that subset in `(x1, y1)`.
                x1, y1 = pos[v]

                # Use `float` to convert the selected value to a floating-point metric, storing the result in `raw_weight`.
                raw_weight = float(data.get("weight", 0))
                # Retrieve the requested value into `c_value`, using the documented fallback when absent.
                c_value = data.get("c_coefficient", np.nan)

                # Use a separate execution path when `c_valid and np.isfinite(c_value)`.
                if c_valid and np.isfinite(c_value):
                    # Set `c_text` to `f'{float(c_value):.4f}'` as the normalized text used below.
                    c_text = f"{float(c_value):.4f}"
                # Use a separate execution path when `analysis_level == 'Indicator'`.
                elif analysis_level == "Indicator":
                    # Set `c_text` to the fixed configuration value `'not defined after indicator aggregation'`.
                    c_text = "not defined after indicator aggregation"
                else:
                    # Set `c_text` to the fixed configuration value `'not available'`.
                    c_text = "not available"

                # Compute `hover_text` as the normalized text used below.
                hover_text = (
                    f"<b>{strip_symbol(str(u))}</b><br>↔"
                    f"<br><b>{strip_symbol(str(v))}</b>"
                    f"<br><br>Raw co-occurrence: {raw_weight:g}"
                    f"<br>C-coefficient: {c_text}"
                )

                # Append one newly assembled record to `traces`.
                traces.append(
                    go.Scatter(
                        x=[x0, x1], y=[y0, y1], mode="lines",
                        line=dict(
                            width=float(edge_widths[edge_index]),
                            color=f"rgba(90,90,90,{edge_opacity})",
                        ),
                        hoverinfo="skip", showlegend=False,
                    )
                )

                # Append `(x0 + x1) / 2` to the accumulating result list.
                edge_mid_x.append((x0 + x1) / 2)
                # Append `(y0 + y1) / 2` to the accumulating result list.
                edge_mid_y.append((y0 + y1) / 2)
                # Append `hover_text` to the accumulating result list.
                edge_hover_text.append(hover_text)
                # Append `f'{raw_weight:g}'` to the accumulating result list.
                edge_value_text.append(f"{raw_weight:g}")

            # Transparent midpoint markers provide a generous hover target for thin lines.
            if show_edge_hover:
                # Append one newly assembled record to `traces`.
                traces.append(
                    go.Scatter(
                        x=edge_mid_x, y=edge_mid_y, mode="markers",
                        marker=dict(size=16, color="rgba(0,0,0,0)"),
                        text=edge_hover_text,
                        hovertemplate="%{text}<extra></extra>",
                        showlegend=False,
                    )
                )

            # Use a separate execution path when `show_edge_values`.
            if show_edge_values:
                # Append one newly assembled record to `traces`.
                traces.append(
                    go.Scatter(
                        x=edge_mid_x, y=edge_mid_y, mode="text",
                        text=edge_value_text,
                        textfont=dict(size=9, color="#555555"),
                        hoverinfo="skip", showlegend=False,
                    )
                )

        # Store the values returned by `np.asarray` in `degree_values`.
        degree_values = np.asarray(
            [weighted_degree[n] for n in G.nodes],
            dtype=float,
        )

        # Label only nodes at or above the selected weighted-degree quantile.
        label_threshold = (
            np.quantile(degree_values, label_quantile)
            if len(degree_values) else 0
        )

        # Declare the ordered values stored in `(node_x, node_y)` so later logic uses one consistent definition.
        node_x, node_y = [], []
        # Declare the ordered values stored in `(node_hover, node_labels)` so later logic uses one consistent definition.
        node_hover, node_labels = [], []

        # Process each `node` drawn from `G.nodes`.
        for node in G.nodes:
            # Select `pos[node]` and store that subset in `(x, y)`.
            x, y = pos[node]
            # Select `G.nodes[node]` and store that subset in `attrs`.
            attrs = G.nodes[node]

            # Append `x` to the accumulating result list.
            node_x.append(x)
            # Append `y` to the accumulating result list.
            node_y.append(y)

            # Retrieve the requested value into `frequency`, using the documented fallback when absent.
            frequency = attrs.get("code_frequency", np.nan)
            # Set `frequency_text` from the condition-dependent choice `f'{frequency:g}' if frequency is not None and np.isfinite(frequency) else 'not available'`.
            frequency_text = (
                f"{frequency:g}"
                if frequency is not None and np.isfinite(frequency)
                else "not available"
            )

            # Record in `sentiment_value` whether the required condition is satisfied.
            sentiment_value = attrs.get("sentiment") or "not applicable"
            # Record in `indicator_value` whether the required condition is satisfied.
            indicator_value = attrs.get("indicator") or str(node)
            # Retrieve the requested value into `community`, using the documented fallback when absent.
            community = community_map.get(node, "unassigned")

            # Append one newly assembled record to `node_hover`.
            node_hover.append(
                f"<b>{strip_symbol(str(node))}</b>"
                f"<br><br>Analysis level: {analysis_level}"
                f"<br>Code frequency: {frequency_text}"
                f"<br>Sentiment: {sentiment_value}"
                f"<br>Indicator: {indicator_value}"
                f"<br>Degree: {unweighted_degree[node]}"
                f"<br>Weighted degree: {weighted_degree[node]:g}"
                f"<br>Weighted degree / frequency: "
                f"{weighted_degree_per_frequency[node]:.3f}"
                f"<br>Betweenness: {betweenness[node]:.4f}"
                f"<br>Community: {community}"
            )

            # Append one newly assembled record to `node_labels`.
            node_labels.append(
                strip_symbol(str(node))
                if show_labels
                and weighted_degree[node] >= label_threshold
                else ""
            )

        # Build `node_marker` as the collection needed by the following loop or calculation.
        node_marker = dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=1, color="white"),
            opacity=0.92,
        )

        # A colorbar is meaningful only when color represents a numeric metric.
        if node_colorbar_title is not None:
            # Merge the newly derived values into `node_marker`.
            node_marker.update(
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(
                    title=dict(text=node_colorbar_title),
                    thickness=14,
                    len=0.65,
                    x=1.01,
                ),
            )

        # Use a separate execution path when `show_nodes`.
        if show_nodes:
            # Append one newly assembled record to `traces`.
            traces.append(
                go.Scatter(
                    x=node_x, y=node_y,
                    mode="markers+text" if show_labels else "markers",
                    marker=node_marker,
                    text=node_labels,
                    textposition="top center",
                    textfont=dict(size=10),
                    customdata=node_hover,
                    hovertemplate="%{customdata}<extra></extra>",
                    showlegend=False,
                )
            )

        # Compute `subtitle` as the derived value used below.
        subtitle = (
            f"Level: {analysis_level} | "
            f"Nodes: {G.number_of_nodes()} | "
            f"Edges: {G.number_of_edges()} | "
            f"Minimum raw weight: {minimum_edge_weight}"
        )

        # Create the figure object `fig` that will hold this visualization.
        fig = go.Figure(data=traces)

        # Call `fig.update_layout` to apply the figure-wide layout and axis settings.
        fig.update_layout(
            dragmode="pan",
            title=dict(
                text=f"{analysis_name} network<br><sup>{subtitle}</sup>",
                x=0.5,
            ),
            width=1100,
            height=760,
            margin=dict(l=20, r=85, t=75, b=25),
            plot_bgcolor="white",
            paper_bgcolor="white",
            hovermode="closest",
            xaxis=dict(visible=False, showgrid=False, zeroline=False),
            yaxis=dict(
                visible=False, showgrid=False, zeroline=False,
                scaleanchor="x", scaleratio=1,
            ),
        )

        # Surface methodological limitations inside the figure so exported images retain them.
        warnings = []

        # Use a separate execution path when `analysis_level == 'Indicator'`.
        if analysis_level == "Indicator":
            # Append one newly assembled record to `warnings`.
            warnings.append(
                "Indicator edges sum all sentiment-specific co-occurrences; "
                "C-coefficients are therefore not reconstructed."
            )

        # Use a separate execution path when `not analyses[analysis_name].get('code_frequency_valid', True)`.
        if not analyses[analysis_name].get("code_frequency_valid", True):
            # Append `'Stop-specific groundedness is unavailable.'` to the accumulating result list.
            warnings.append(
            "Stop-specific groundedness is unavailable."
     )

        # Use a separate execution path when `warnings`.
        if warnings:
            # Call `fig.add_annotation` to apply its documented side effect.
            fig.add_annotation(
                text="<b>Note:</b> " + " ".join(warnings),
                x=0.5, y=-0.02,
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=10, color="#B22222"),
            )

        # Return `fig` as the computed result.
        return fig

    # Keep controls compact enough to fit above the 1100-pixel-wide figure.
    style = {"description_width": "initial"}
    # Store the derived value returned by `widgets.Layout` in `short`.
    short = widgets.Layout(width="245px")
    # Store the derived value returned by `widgets.Layout` in `medium`.
    medium = widgets.Layout(width="310px")

    # Store the derived value returned by `widgets.Dropdown` in `analysis_selector`.
    analysis_selector = widgets.Dropdown(
        options=list(analyses), value=default_analysis,
        description="Network:", style=style, layout=short,
    )

    # Store the derived value returned by `widgets.ToggleButtons` in `level_selector`.
    level_selector = widgets.ToggleButtons(
        options=["Code + sentiment", "Indicator", "Domain"],
        value="Code + sentiment",
        description="Analysis level:",
        style=style,
        layout=widgets.Layout(width="600px"),
    )

    # Set the threshold ceiling from the largest edge across every analysis and level.
    max_weight = max(
        (
            data.get("weight", 1)
            for analysis_levels in level_graphs.values()
            for graph, _ in analysis_levels.values()
            for _, _, data in graph.edges(data=True)
        ),
        default=1,
    )

    # Store the derived value returned by `widgets.IntSlider` in `threshold_slider`.
    threshold_slider = widgets.IntSlider(
        value=GLOBAL_NETWORK_MIN_COOCCURRENCE,
        min=1, max=max(1, int(max_weight)), step=1,
        description="Min weight:", continuous_update=False,
        style=style, layout=medium,
    )

    # Store the derived value returned by `widgets.Dropdown` in `layout_selector`.
    layout_selector = widgets.Dropdown(
        options=["Spring", "Kamada-Kawai", "Circular", "Spectral"],
        value="Spring", description="Layout:",
        style=style, layout=short,
    )

    # Store the derived value returned by `widgets.Dropdown` in `node_size_selector`.
    node_size_selector = widgets.Dropdown(
        options=["Code frequency", "Weighted degree", "Degree", "Fixed"],
        value="Code frequency", description="Node size:",
        style=style, layout=short,
    )

    # Store the display color returned by `widgets.Dropdown` in `node_color_selector`.
    node_color_selector = widgets.Dropdown(
        options=[
            "Sentiment", "Community", "Degree", "Weighted degree",
            "Weighted degree / code frequency", "Betweenness", "Fixed",
        ],
        value="Sentiment", description="Node colour:",
        style=style, layout=medium,
    )

    # Compute the network edge table `edge_width_selector` from the value returned by `widgets.Dropdown`.
    edge_width_selector = widgets.Dropdown(
        options=["Raw co-occurrence", "C-coefficient", "Fixed"],
        value="Raw co-occurrence", description="Edge width:",
        style=style, layout=medium,
    )

    # Store the display label returned by `widgets.FloatSlider` in `label_quantile_slider`.
    label_quantile_slider = widgets.FloatSlider(
        value=0.70, min=0, max=1, step=0.05,
        description="Label cutoff:", continuous_update=False,
        readout_format=".0%", style=style, layout=medium,
    )

    # Compute the network edge table `edge_opacity_slider` from the value returned by `widgets.FloatSlider`.
    edge_opacity_slider = widgets.FloatSlider(
        value=0.45, min=0.05, max=1, step=0.05,
        description="Edge opacity:", continuous_update=False,
        readout_format=".0%", style=style, layout=medium,
    )

    # Store the derived value returned by `widgets.Checkbox` in `show_nodes_toggle`.
    show_nodes_toggle = widgets.Checkbox(value=True, description="Nodes")
    # Compute the network edge table `show_edges_toggle` from the value returned by `widgets.Checkbox`.
    show_edges_toggle = widgets.Checkbox(value=True, description="Edges")
    # Store the derived value returned by `widgets.Checkbox` in `show_isolates_toggle`.
    show_isolates_toggle = widgets.Checkbox(
        value=False, description="Isolated nodes"
    )
    # Store the display label returned by `widgets.Checkbox` in `show_labels_toggle`.
    show_labels_toggle = widgets.Checkbox(
        value=True, description="Node labels"
    )
    # Compute the network edge table `show_edge_values_toggle` from the value returned by `widgets.Checkbox`.
    show_edge_values_toggle = widgets.Checkbox(
        value=False, description="Edge values"
    )
    # Compute the network edge table `show_edge_hover_toggle` from the value returned by `widgets.Checkbox`.
    show_edge_hover_toggle = widgets.Checkbox(
        value=True, description="Edge hover"
    )

    # Compute the record accumulator `compact_row` from the value returned by `widgets.Layout`.
    compact_row = widgets.Layout(
        display="flex", flex_flow="row wrap",
        align_items="center", gap="4px 8px",
    )

    # Group analytical selectors separately from visibility toggles for faster scanning.
    controls = widgets.VBox(
        [
            widgets.HTML("<b>Interactive network controls</b>"),
            level_selector,
            widgets.Box(
                [
                    analysis_selector, threshold_slider, layout_selector,
                    node_size_selector, node_color_selector,
                    edge_width_selector, label_quantile_slider,
                    edge_opacity_slider,
                ],
                layout=compact_row,
            ),
            widgets.Box(
                [
                    show_nodes_toggle, show_edges_toggle,
                    show_isolates_toggle, show_labels_toggle,
                    show_edge_values_toggle, show_edge_hover_toggle,
                ],
                layout=compact_row,
            ),
        ],
        layout=widgets.Layout(gap="2px"),
    )

    # Keep Plotly as a direct widget. Using interactive_output here can deadlock
    # Voilà while it waits for the Output widget's nested display messages.
    render_inputs = {
            "analysis_name": analysis_selector,
            "analysis_level": level_selector,
            "minimum_edge_weight": threshold_slider,
            "layout_name": layout_selector,
            "node_size_mode": node_size_selector,
            "node_color_mode": node_color_selector,
            "edge_width_mode": edge_width_selector,
            "show_nodes": show_nodes_toggle,
            "show_edges": show_edges_toggle,
            "show_isolates": show_isolates_toggle,
            "show_labels": show_labels_toggle,
            "show_edge_values": show_edge_values_toggle,
            "show_edge_hover": show_edge_hover_toggle,
            "label_quantile": label_quantile_slider,
            "edge_opacity": edge_opacity_slider,
    }

    # Define `current_figure` to encapsulate the current figure operation for reuse and testing.
    def current_figure():
        # Return the result produced by `render`.
        return render(**{name: widget.value for name, widget in render_inputs.items()})

    # Compute the graph representation `graph` from the value returned by `widgets.Box`.
    graph = widgets.Box()

    # Define `refresh_graph` to encapsulate the refresh graph operation for reuse and testing.
    def refresh_graph(change=None):
        # Create the figure object `updated` that will hold this visualization.
        updated = go.FigureWidget(current_figure())
        # Initialize `updated._config` as the lookup table used by the following analysis.
        updated._config = {
            "displaylogo": False, "scrollZoom": True,
            "toImageButtonOptions": {"format": "png", "scale": 2},
        }
        # Declare the ordered values stored in `graph.children` so later logic uses one consistent definition.
        graph.children = (updated,)

    # Process each `widget` drawn from `dict.fromkeys(render_inputs.values())`.
    for widget in dict.fromkeys(render_inputs.values()):
        # Call `widget.observe` to register the redraw callback on this widget.
        widget.observe(refresh_graph, names="value")

    # Call `refresh_graph` to apply its documented side effect.
    refresh_graph()

    # Render `controls` in the notebook for analyst inspection.
    if display_widget:
        # Publish both widget objects exactly as in the analytical notebook.
        display(controls, graph)
    # Return the assembled collection to the caller.
    return controls, graph
