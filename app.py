"""
app.py — Streamlit frontend for the WNBA Player Similarity Engine.

Views:
  Tab 1 — Galaxy Map   : Interactive 3D Plotly graph, MDS-positioned so physical
                         distance directly reflects statistical similarity.
  Tab 2 — Radar Overlay: Spider chart comparing selected player vs their top match
  Tab 3 — Distribution : Histogram of all pairwise similarities to tune the threshold

Run with:
    streamlit run app.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.manifold import MDS

from src.processor import get_top_matches, load_processed
from src.utils import load_config

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WNBA Similarity Engine",
    page_icon="🏀",
    layout="wide",
)

# ── Data loading (cached so it only runs once per session) ────────────────────
@st.cache_resource
def load_data():
    cfg = load_config()
    player_df, sim_matrix = load_processed(cfg)
    return cfg, player_df, sim_matrix


@st.cache_data
def compute_mds(_sim_matrix: np.ndarray) -> np.ndarray:
    """
    Convert similarity matrix to 3D coordinates via MDS.
    Distance = 1 - similarity, so closer in space = more statistically similar.
    Cached because MDS is expensive and the matrix doesn't change per interaction.
    """
    dist_matrix = 1.0 - _sim_matrix
    np.fill_diagonal(dist_matrix, 0)
    mds = MDS(
        n_components=3,
        dissimilarity="precomputed",
        random_state=42,
        n_init=4,
        normalized_stress="auto",
    )
    return mds.fit_transform(dist_matrix)


cfg, player_df, sim_matrix = load_data()
season = cfg["data"]["season"]
default_threshold = cfg["model"]["sim_threshold"]
default_n = cfg["model"]["neighbors_count"]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🏀 WNBA Similarity")
    st.caption(f"Season: **{season}** · {len(player_df)} players")
    st.divider()

    player_names = sorted(player_df["PLAYER_NAME"].tolist())
    selected_player = st.selectbox("Select a player", player_names, index=player_names.index("Satou Sabally") if "Satou Sabally" in player_names else 0)

    threshold = st.slider(
        "Similarity threshold (Galaxy Map)",
        min_value=0.70,
        max_value=0.99,
        value=default_threshold,
        step=0.01,
        help="Only draw edges between players whose similarity exceeds this value.",
    )

    top_n = st.slider(
        "Top-N matches (Radar)",
        min_value=1,
        max_value=10,
        value=default_n,
    )


# ── Top matches for selected player ──────────────────────────────────────────
try:
    top_matches = get_top_matches(selected_player, player_df, sim_matrix, n=top_n)
except ValueError as e:
    st.error(str(e))
    st.stop()

top_match_name = top_matches.iloc[0]["PLAYER_NAME"]
top_match_score = top_matches.iloc[0]["similarity"]

# Append the ranked match list to the sidebar (second with-block appends content)
with st.sidebar:
    st.divider()
    st.markdown("**Top matches**")
    for rank, row in top_matches.iterrows():
        st.markdown(f"`{rank + 1}.` **{row['PLAYER_NAME']}** · {row['TEAM_ABBREVIATION']}")
        st.progress(float(row["similarity"]), text=f"{row['similarity']:.3f}")
    st.divider()
    st.caption("Data: nba_api · Built with Streamlit")

st.header(f"{selected_player}")

tab1, tab2, tab3 = st.tabs(["🌌 Galaxy Map", "📡 Radar Overlay", "📊 Distribution"])

# ── Tab 1: Galaxy Map ─────────────────────────────────────────────────────────
with tab1:
    st.subheader("Player Similarity Galaxy")
    st.caption(
        f"MDS-positioned in 3D — physical distance = statistical distance. "
        f"Edges drawn where similarity > **{threshold:.2f}**. Rotate, zoom, and hover freely."
    )

    coords = compute_mds(sim_matrix)  # shape (N, 3)
    names  = player_df["PLAYER_NAME"].tolist()

    sel_idx = player_df[player_df["PLAYER_NAME"] == selected_player].index[0]
    top_idx = player_df[player_df["PLAYER_NAME"] == top_match_name].index[0]
    top_match_indices = set(
        player_df[player_df["PLAYER_NAME"] == r["PLAYER_NAME"]].index[0]
        for _, r in top_matches.iterrows()
    )

    # ── Edge traces (batched into one trace using None separators) ────────────
    ex, ey, ez = [], [], []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if sim_matrix[i, j] > threshold:
                ex += [coords[i, 0], coords[j, 0], None]
                ey += [coords[i, 1], coords[j, 1], None]
                ez += [coords[i, 2], coords[j, 2], None]

    edge_trace = go.Scatter3d(
        x=ex, y=ey, z=ez,
        mode="lines",
        line=dict(color="rgba(255,255,255,0.12)", width=1),
        hoverinfo="none",
        showlegend=False,
    )

    # ── Node traces (one per group for independent legend entries) ───────────
    def make_node_trace(indices, label, color, size):
        return go.Scatter3d(
            x=coords[indices, 0],
            y=coords[indices, 1],
            z=coords[indices, 2],
            mode="markers",
            name=label,
            marker=dict(color=color, size=size, opacity=0.9,
                        line=dict(color="white", width=0.5)),
            text=[names[i] for i in indices],
            customdata=[
                player_df.iloc[i]["TEAM_ABBREVIATION"] for i in indices
            ],
            hovertemplate="<b>%{text}</b><br>%{customdata}<extra></extra>",
        )

    rest_idx    = [i for i in range(len(names))
                   if i != sel_idx and i not in top_match_indices]
    match_idx   = [i for i in top_match_indices if i != sel_idx and i != top_idx]

    traces = [
        edge_trace,
        make_node_trace(rest_idx,    "League",       "#4A4A6A", 5),
        make_node_trace(match_idx,   "Top matches",  "#7EB8F7", 9),
        make_node_trace([top_idx],   "Top match",    "#00C2FF", 14),
        make_node_trace([sel_idx],   selected_player,"#FF6B35", 16),
    ]

    fig3d = go.Figure(
        data=traces,
        layout=go.Layout(
            height=650,
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            scene=dict(
                bgcolor="#0e1117",
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
            ),
            legend=dict(
                font=dict(color="white"),
                bgcolor="rgba(26,26,46,0.8)",
                bordercolor="#444466",
                borderwidth=1,
                itemsizing="constant",
            ),
            margin=dict(l=0, r=0, t=0, b=0),
        ),
    )

    st.plotly_chart(fig3d, use_container_width=True)

# ── Tab 2: Radar Overlay ──────────────────────────────────────────────────────
with tab2:
    st.subheader(f"{selected_player}  vs  {top_match_name}")
    st.caption("Comparison on normalized Per-40 and advanced metrics.")

    # Features and display labels for the radar
    RADAR_FEATURES = ["PTS_P40", "REB_P40", "AST_P40", "STL_P40", "BLK_P40", "FG3_PCT", "USG_PCT", "TS_PCT"]
    RADAR_LABELS   = ["PTS/40", "REB/40", "AST/40", "STL/40", "BLK/40", "FG3%", "USG%", "TS%"]

    raw_path = Path(cfg["data"]["raw_path"])
    import pandas as pd
    raw_df = pd.read_csv(raw_path)

    # Recompute Per-40 for display (raw values, not scaled)
    for stat in ["PTS", "REB", "AST", "STL", "BLK"]:
        raw_df[f"{stat}_P40"] = (raw_df[stat] / raw_df["MIN"].replace(0, np.nan)) * 40

    def get_radar_values(name: str) -> np.ndarray:
        row = raw_df[raw_df["PLAYER_NAME"] == name]
        if row.empty:
            return np.zeros(len(RADAR_FEATURES))
        return row.iloc[0][RADAR_FEATURES].fillna(0).values.astype(float)

    vals_sel = get_radar_values(selected_player)
    vals_top = get_radar_values(top_match_name)

    # Normalize each feature to [0, 1] across all players for readability
    all_vals = np.array([get_radar_values(n) for n in player_df["PLAYER_NAME"]])
    col_min = all_vals.min(axis=0)
    col_max = all_vals.max(axis=0)
    col_range = np.where(col_max - col_min == 0, 1, col_max - col_min)

    vals_sel_norm = (vals_sel - col_min) / col_range
    vals_top_norm = (vals_top - col_min) / col_range

    # Polar plot
    num_vars = len(RADAR_LABELS)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    v_sel = vals_sel_norm.tolist() + vals_sel_norm[:1].tolist()
    v_top = vals_top_norm.tolist() + vals_top_norm[:1].tolist()

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    ax.plot(angles, v_sel, color="#FF6B35", linewidth=2, label=selected_player)
    ax.fill(angles, v_sel, color="#FF6B35", alpha=0.25)

    ax.plot(angles, v_top, color="#00C2FF", linewidth=2, label=top_match_name)
    ax.fill(angles, v_top, color="#00C2FF", alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(RADAR_LABELS, color="white", size=11)
    ax.set_yticklabels([])
    ax.grid(color="#444466", linestyle="--", linewidth=0.5)
    ax.spines["polar"].set_color("#444466")

    legend = ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.15))
    for text in legend.get_texts():
        text.set_color("white")
    legend.get_frame().set_facecolor("#1a1a2e")

    st.pyplot(fig, use_container_width=False)

    # Raw stat table beneath the chart
    st.divider()
    display_labels = ["PTS/40", "REB/40", "AST/40", "STL/40", "BLK/40", "FG3%", "USG%", "TS%"]
    comparison = pd.DataFrame(
        {selected_player: np.round(vals_sel, 3), top_match_name: np.round(vals_top, 3)},
        index=display_labels,
    )
    st.dataframe(comparison, use_container_width=True)

# ── Tab 3: Similarity Distribution ───────────────────────────────────────────
with tab3:
    st.subheader("Pairwise Similarity Distribution")
    st.caption(
        "Distribution of all player-pair similarity scores. "
        "Use this to calibrate the Galaxy Map threshold — moving it left reveals more connections, right shows only the tightest clusters."
    )

    # Extract upper triangle (no diagonal, no duplicates)
    upper = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]

    fig2, ax2 = plt.subplots(figsize=(9, 4))
    fig2.patch.set_facecolor("#0e1117")
    ax2.set_facecolor("#0e1117")

    ax2.hist(upper, bins=60, color="#7EB8F7", edgecolor="#0e1117", linewidth=0.3)
    ax2.axvline(threshold, color="#FF6B35", linewidth=2, label=f"Threshold: {threshold:.2f}")

    pairs_above = int((upper > threshold).sum())
    ax2.text(
        threshold + 0.005, ax2.get_ylim()[1] * 0.85,
        f"{pairs_above} pairs\nabove threshold",
        color="#FF6B35", fontsize=9,
    )

    ax2.set_xlabel("Cosine Similarity", color="white")
    ax2.set_ylabel("Number of Player Pairs", color="white")
    ax2.tick_params(colors="white")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#444466")
    legend2 = ax2.legend()
    for text in legend2.get_texts():
        text.set_color("white")
    legend2.get_frame().set_facecolor("#1a1a2e")

    st.pyplot(fig2, use_container_width=True)

    st.metric("Total player pairs", f"{len(upper):,}")
    col_a, col_b = st.columns(2)
    col_a.metric("Pairs above threshold", f"{pairs_above:,}")
    col_b.metric("Median similarity", f"{np.median(upper):.3f}")
