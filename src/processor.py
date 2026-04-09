"""
processor.py — Clean, normalize, weight, and compute the similarity matrix.

Pipeline:
  1. Load raw merged CSV
  2. Filter players below min_minutes threshold
  3. Compute Per-40 counting stats
  4. Apply config weights to feature columns
  5. Z-score scale (StandardScaler)
  6. Compute cosine similarity matrix (N x N)
  7. Save processed matrix + player index to disk

Run directly:
    python -m src.processor
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from src.utils import load_config

# Counting stats to convert to Per-40 basis
PER40_STATS = ["PTS", "REB", "AST", "STL", "BLK", "TOV"]

# All features that enter the similarity vector (after Per-40 conversion)
FEATURES = [
    "PTS_P40", "REB_P40", "AST_P40", "STL_P40", "BLK_P40", "TOV_P40",
    "FG3_PCT", "USG_PCT", "TS_PCT", "PIE", "WNBA_EXPERIENCE",
]

# Map from feature name to config weight key
WEIGHT_MAP = {
    "PTS_P40":        "pts",
    "REB_P40":        "reb",
    "AST_P40":        "ast",
    "STL_P40":        "stl",
    "BLK_P40":        "blk",
    "TOV_P40":        "tov",
    "FG3_PCT":        "fg3_pct",
    "USG_PCT":        "usg_pct",
    "TS_PCT":         "ts_pct",
    "PIE":            "pie",
    "WNBA_EXPERIENCE": "wnba_experience",
}


def _per40(df: pd.DataFrame) -> pd.DataFrame:
    """Add Per-40 columns for all counting stats. Avoids division by zero."""
    for stat in PER40_STATS:
        if stat in df.columns:
            df[f"{stat}_P40"] = (df[stat] / df["MIN"].replace(0, np.nan)) * 40
    return df


def build_similarity_matrix(config: dict) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Full processing pipeline. Returns:
        player_df  : DataFrame with PLAYER_ID, PLAYER_NAME, TEAM_ABBREVIATION, name_clean
        sim_matrix : np.ndarray of shape (N, N), cosine similarities
    """
    raw_path = Path(config["data"]["raw_path"])
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {raw_path}. Run `python -m src.data_fetch` first."
        )

    df = pd.read_csv(raw_path)

    # 1. Filter minimum minutes
    min_min = config["data"].get("min_minutes", 100)
    df = df[df["MIN"] >= min_min].copy()
    print(f"Players after {min_min}-minute filter: {len(df)}")

    # 2. Per-40 normalization
    df = _per40(df)

    # 3. Handle WNBA_EXPERIENCE:
    #    - Undrafted players have NaN — fill with 0 so they aren't silently dropped.
    #      Their performance stats are still valid; we just don't know their draft year.
    #    - drop_rookies=True removes WNBA_EXPERIENCE=0 players (useful early in a new
    #      season when rookies have only a handful of games). For a completed season
    #      like 2025, set drop_rookies=false to keep rookies with a full year of data.
    df["WNBA_EXPERIENCE"] = df["WNBA_EXPERIENCE"].fillna(0)
    if config["data"].get("drop_rookies", False):
        before = len(df)
        df = df[df["WNBA_EXPERIENCE"] > 0]
        print(f"Dropped {before - len(df)} rookies (drop_rookies=true)")

    # 4. Drop rows with NaN in any remaining feature
    df = df.dropna(subset=FEATURES)
    print(f"Players after dropping NaN features: {len(df)}")
    df = df.reset_index(drop=True)

    # 5. Apply weights — multiply each feature column by its config weight before scaling
    weights_cfg = config["weights"]
    feature_matrix = df[FEATURES].copy()
    for feat, weight_key in WEIGHT_MAP.items():
        if feat in feature_matrix.columns:
            feature_matrix[feat] *= weights_cfg.get(weight_key, 1.0)

    # 6. Z-score scaling
    scaler = StandardScaler()
    scaled = scaler.fit_transform(feature_matrix)

    # 7. Cosine similarity
    sim_matrix = cosine_similarity(scaled)

    # 8. Player index DataFrame
    id_cols = ["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "name_clean"]
    player_df = df[[c for c in id_cols if c in df.columns]].copy()

    return player_df, sim_matrix


def save_processed(config: dict, player_df: pd.DataFrame, sim_matrix: np.ndarray) -> None:
    out_path = Path(config["data"]["processed_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({"players": player_df, "matrix": sim_matrix}, f)
    print(f"Saved similarity matrix ({sim_matrix.shape}) to {out_path}")


def load_processed(config: dict) -> tuple[pd.DataFrame, np.ndarray]:
    out_path = Path(config["data"]["processed_path"])
    if not out_path.exists():
        raise FileNotFoundError(
            f"Processed data not found at {out_path}. Run `python -m src.processor` first."
        )
    with open(out_path, "rb") as f:
        data = pickle.load(f)
    return data["players"], data["matrix"]


def get_top_matches(
    player_name: str,
    player_df: pd.DataFrame,
    sim_matrix: np.ndarray,
    n: int = 5,
) -> pd.DataFrame:
    """
    Return the top-N most similar players to player_name.
    Result is a DataFrame with columns: PLAYER_NAME, TEAM_ABBREVIATION, similarity.
    """
    name_clean = player_name.lower().strip()
    matches = player_df[player_df["name_clean"] == name_clean]
    if matches.empty:
        matches = player_df[player_df["name_clean"].str.contains(name_clean, na=False)]
    if matches.empty:
        raise ValueError(f"Player '{player_name}' not found in dataset.")

    idx = matches.index[0]
    scores = sim_matrix[idx]

    top_idx = np.argsort(scores)[::-1]
    top_idx = [i for i in top_idx if i != idx][:n]

    result = player_df.iloc[top_idx][["PLAYER_NAME", "TEAM_ABBREVIATION"]].copy()
    result["similarity"] = scores[top_idx]
    result = result.reset_index(drop=True)
    return result


if __name__ == "__main__":
    cfg = load_config()
    player_df, sim_matrix = build_similarity_matrix(cfg)
    save_processed(cfg, player_df, sim_matrix)

    print(f"\nSimilarity matrix range: [{sim_matrix.min():.3f}, {sim_matrix.max():.3f}]")
    print(f"Pairs above 0.92 threshold: {(sim_matrix > 0.92).sum() // 2}")

    try:
        matches = get_top_matches("Satou Sabally", player_df, sim_matrix, n=5)
        print("\nTop 5 matches for Satou Sabally:")
        print(matches.to_string(index=False))
    except ValueError as e:
        print(f"\n{e}")
