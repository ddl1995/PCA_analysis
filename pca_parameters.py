"""
pca_parameters.py
-----------------
Perform PCA on a set of water-quality (or any) parameters across sampling sites, and
visualize results as:
1) Scree plot (explained variance)
2) PCA biplot (PC1 vs PC2) with:
   - Site scores colored by WQI (or any numeric index)
   - Loading vectors (arrows) for the input parameters

This script is written to be "GitHub-ready":
- No machine-specific absolute paths (use a relative path by default).
- Clear "USER SETTINGS" section.
- Saves tidy CSV outputs + publication-ready PNG figures.

Expected CSV format
-------------------
Your input CSV should contain:
- One ID column (e.g., Site ID)
- One numeric index column for coloring points (e.g., WQI)
- Parameter columns (numeric) to compute PCA, named as: parameter1, parameter2, ...

Example columns:
Site ID, WQI, parameter1, parameter2, parameter3, ...

How to run
----------
1) Install dependencies:
   pip install numpy pandas matplotlib scikit-learn

2) Update USER SETTINGS below (CSV_PATH and column names), then run:
   python pca_parameters.py

Outputs are saved to the output folder (see OUT_DIR in USER SETTINGS).

Notes
-----
- PCA is computed ONLY using PARAM_COLS.
- WQI (or your chosen index column) is NOT used in PCA; it is only used for coloring.
- Missing parameter values are filled with column means (simple but robust for small datasets).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# =============================================================================
# USER SETTINGS (edit these for your dataset)
# =============================================================================

# Path to your input CSV (use a relative path for GitHub projects).
# Example: "data/site_vs_parameters_PCA.csv"
CSV_PATH = "data/site_vs_parameters_PCA.csv"

# Column names in the CSV
ID_COL = "Site ID"      # site identifier (string/number)
INDEX_COL = "WQI"       # numeric index for coloring points (e.g., WQI)

# PCA variables (computed only on these columns)
# Keep generic names for GitHub: parameter1, parameter2, ...
PARAM_COLS = ["parameter1", "parameter2", "parameter3", "parameter4", "parameter5", "parameter6"]

# Which PCA plane to plot (0-based indexing): 0 -> PC1, 1 -> PC2, ...
PCX, PCY = 0, 1

# Output directory:
# - If None, an "outputs" folder will be created next to the input CSV.
# - Or set a custom relative path like "outputs".
OUT_DIR: Optional[str] = None

# Figures
SAVE_FIGS = True
SHOW_FIGS = True

# Plot options
LABEL_POINTS = True        # adds site IDs next to points (can be cluttered for many sites)
POINT_SIZE = 85
ARROW_SCALE = 2.0          # visual scaling of loading arrows

# =============================================================================
# STYLE SETTINGS (optional)
# =============================================================================
FONT_TITLE = 18
FONT_AXES = 15
FONT_TICKS = 13
FONT_SITE_LABELS = 12
FONT_VAR_LABELS = 13
FONT_CBAR = 13

PARAM_ARROW_COLOR = "black"
PARAM_ARROW_LW = 1.6

# Apply default font
plt.rcParams.update({"font.size": FONT_TICKS})


# =============================================================================
# Helper: find columns robustly (case/space tolerant)
# =============================================================================
def _find_column(df_cols: List[str], target: str) -> Optional[str]:
    """Return the real column name from df_cols that matches `target` (tolerant of spaces/case)."""
    stripped_map = {c.strip(): c for c in df_cols}
    if target.strip() in stripped_map:
        return stripped_map[target.strip()]
    for c in df_cols:
        if c.strip().lower() == target.strip().lower():
            return c
    return None


def _resolve_columns(df: pd.DataFrame, names: List[str], kind: str) -> List[str]:
    """Resolve a list of requested column names against df.columns using tolerant matching."""
    resolved, missing = [], []
    for name in names:
        col = _find_column(list(df.columns), name)
        if col is None:
            missing.append(name)
        else:
            resolved.append(col)
    if missing:
        raise ValueError(
            f"Missing {kind} column(s): {missing}\n"
            f"Available columns:\n{list(df.columns)}\n\n"
            f"Fix: update ID_COL / INDEX_COL / PARAM_COLS to match your CSV headers."
        )
    return resolved


def _ensure_out_dir(csv_path: Path, out_dir: Optional[str]) -> Path:
    """Return a valid output directory path and create it if needed."""
    if out_dir is None:
        out = csv_path.parent / "outputs"
    else:
        out = Path(out_dir)
        # If user provides a relative folder, make it relative to the CSV location for convenience
        if not out.is_absolute():
            out = csv_path.parent / out
    out.mkdir(parents=True, exist_ok=True)
    return out


def _corr_with_pc(x_series: pd.Series, pc_scores: np.ndarray, pc_index: int) -> float:
    """Correlation between a variable and a given PC score vector (used here for INDEX_COL only)."""
    x = pd.to_numeric(x_series, errors="coerce")
    x = x.fillna(x.mean() if np.isfinite(x.mean()) else 0.0).values
    return float(np.corrcoef(x, pc_scores[:, pc_index])[0, 1])


def main() -> None:
    # -------------------------------------------------------------------------
    # 1) Load data
    # -------------------------------------------------------------------------
    csv_path = Path(CSV_PATH)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV_PATH not found: {csv_path.resolve()}\n"
            f"Tip: Put your CSV under a 'data/' folder (recommended), or update CSV_PATH."
        )

    df = pd.read_csv(csv_path)

    # Resolve tolerant matches (handles extra spaces or case mismatches)
    id_col = _resolve_columns(df, [ID_COL], "ID")[0]
    index_col = _resolve_columns(df, [INDEX_COL], "index (e.g., WQI)")[0]
    param_cols = _resolve_columns(df, PARAM_COLS, "parameter")

    # Convert to numeric safely
    df[param_cols] = df[param_cols].apply(pd.to_numeric, errors="coerce")
    df[index_col] = pd.to_numeric(df[index_col], errors="coerce")

    # Fill missing parameter values with column means (simple, stable default)
    if df[param_cols].isna().any().any():
        df[param_cols] = df[param_cols].fillna(df[param_cols].mean(numeric_only=True))

    # If INDEX_COL has missing values, fill with its mean (keeps color mapping stable)
    if df[index_col].isna().any():
        m = df[index_col].mean()
        df[index_col] = df[index_col].fillna(m if np.isfinite(m) else 0.0)

    out_dir = _ensure_out_dir(csv_path, OUT_DIR)

    # -------------------------------------------------------------------------
    # 2) Standardize + PCA (parameters only)
    # -------------------------------------------------------------------------
    X = df[param_cols].values
    X_z = StandardScaler().fit_transform(X)

    # PCA components cannot exceed min(n_samples, n_features)
    n_samples, n_features = X_z.shape
    n_components = min(n_samples, n_features)

    if n_components < 2:
        raise ValueError(
            f"Need at least 2 samples and 2 parameters to plot PC1 vs PC2.\n"
            f"Got n_samples={n_samples}, n_features={n_features}."
        )

    if max(PCX, PCY) >= n_components:
        raise ValueError(
            f"PCX/PCY out of range for this dataset.\n"
            f"Computed n_components={n_components}, but requested PCX={PCX}, PCY={PCY}."
        )

    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X_z)

    explained_var = pca.explained_variance_ratio_
    cum_explained = np.cumsum(explained_var)

    loadings = pca.components_.T  # (n_vars, n_pcs)
    pc_names = [f"PC{i+1}" for i in range(n_components)]

    # -------------------------------------------------------------------------
    # 3) Save PCA outputs (CSV)
    # -------------------------------------------------------------------------
    scores_df = pd.DataFrame(scores, columns=pc_names)
    scores_df.insert(0, id_col, df[id_col].values)
    scores_df[index_col] = df[index_col].values
    scores_df.to_csv(out_dir / "PCA_scores.csv", index=False)

    loadings_df = pd.DataFrame(loadings, index=param_cols, columns=pc_names)
    loadings_df.to_csv(out_dir / "PCA_loadings.csv")

    explained_df = pd.DataFrame(
        {"PC": pc_names, "Explained_Variance_Ratio": explained_var, "Cumulative_Explained_Variance": cum_explained}
    )
    explained_df.to_csv(out_dir / "PCA_explained_variance.csv", index=False)

    # Useful quick diagnostic: correlation between INDEX_COL and PC axes
    index_rx = _corr_with_pc(df[index_col], scores, PCX)
    index_ry = _corr_with_pc(df[index_col], scores, PCY)

    print(f"Saved PCA tables in: {out_dir.resolve()}")
    print(f"{INDEX_COL} correlation with PC{PCX+1} = {index_rx:.3f}, PC{PCY+1} = {index_ry:.3f}")

    # -------------------------------------------------------------------------
    # 4) Scree plot
    # -------------------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    x = np.arange(1, len(explained_var) + 1)

    plt.bar(x, explained_var)
    plt.plot(x, cum_explained, marker="o")

    plt.xticks(x, pc_names, fontsize=FONT_TICKS)
    plt.yticks(fontsize=FONT_TICKS)

    plt.xlabel("Principal Components", fontsize=FONT_AXES)
    plt.ylabel("Explained variance ratio / Cumulative", fontsize=FONT_AXES)
    plt.title("PCA Scree Plot (Explained Variance)", fontsize=FONT_TITLE)

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if SAVE_FIGS:
        plt.savefig(out_dir / "PCA_scree.png", dpi=300)
    if SHOW_FIGS:
        plt.show()
    plt.close()

    # -------------------------------------------------------------------------
    # 5) Biplot (PCX vs PCY)
    #    - points colored by INDEX_COL (e.g., WQI)
    #    - parameter arrows (loadings)
    # -------------------------------------------------------------------------
    plt.figure(figsize=(12, 8))

    # Points (colored by index)
    idx_vals = df[index_col].values
    sc = plt.scatter(scores[:, PCX], scores[:, PCY], c=idx_vals, s=POINT_SIZE)

    # Colorbar
    cbar = plt.colorbar(sc)
    cbar.set_label(INDEX_COL, fontsize=FONT_CBAR)
    cbar.ax.tick_params(labelsize=FONT_TICKS)

    # Site labels
    if LABEL_POINTS:
        for i, sid in enumerate(df[id_col].values):
            plt.text(scores[i, PCX], scores[i, PCY], str(sid), fontsize=FONT_SITE_LABELS)

    # Parameter arrows (active loadings)
    for i, var in enumerate(param_cols):
        x_end = loadings[i, PCX] * ARROW_SCALE
        y_end = loadings[i, PCY] * ARROW_SCALE

        plt.annotate(
            "",
            xy=(x_end, y_end),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>", color=PARAM_ARROW_COLOR, lw=PARAM_ARROW_LW),
        )
        plt.text(x_end * 1.12, y_end * 1.12, var, fontsize=FONT_VAR_LABELS, color=PARAM_ARROW_COLOR)

    plt.xlabel(f"PC{PCX+1} ({explained_var[PCX]*100:.1f}%)", fontsize=FONT_AXES)
    plt.ylabel(f"PC{PCY+1} ({explained_var[PCY]*100:.1f}%)", fontsize=FONT_AXES)
    plt.xticks(fontsize=FONT_TICKS)
    plt.yticks(fontsize=FONT_TICKS)

    plt.title(
        f"PCA Biplot (parameters only) + {INDEX_COL} color\nPC{PCX+1} vs PC{PCY+1}",
        fontsize=FONT_TITLE,
    )

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if SAVE_FIGS:
        plt.savefig(out_dir / f"PCA_biplot_{INDEX_COL}_PC{PCX+1}_PC{PCY+1}.png", dpi=300)
    if SHOW_FIGS:
        plt.show()
    plt.close()

    # -------------------------------------------------------------------------
    # 6) Quick loading summary (console)
    # -------------------------------------------------------------------------
    print("\nTop contributors by |loading|:")
    abs_pcx = loadings_df[f"PC{PCX+1}"].abs().sort_values(ascending=False)
    abs_pcy = loadings_df[f"PC{PCY+1}"].abs().sort_values(ascending=False)
    print(f"\nPC{PCX+1}:\n{abs_pcx}")
    print(f"\nPC{PCY+1}:\n{abs_pcy}")


if __name__ == "__main__":
    main()
