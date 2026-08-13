from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from scripts.config_utils import load_project_config, project_path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = project_path(load_project_config(), "paths.experiments_dir")


def list_oof_files() -> list[Path]:
    patterns = [
        "*/artifacts/*oof*.csv",
        "*/features/*oof*.csv",
        "*/*oof*.csv",
    ]
    files: list[Path] = []
    for pattern in patterns:
        files.extend(EXPERIMENTS_DIR.glob(pattern))
    return sorted(set(files))


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def main() -> None:
    st.set_page_config(page_title="OOF Analysis", layout="wide")
    st.title("OOF Analysis")

    oof_files = list_oof_files()
    if not oof_files:
        st.info(f"No OOF CSV files found under {EXPERIMENTS_DIR.relative_to(ROOT)}/.")
        return

    selected = st.sidebar.selectbox(
        "OOF file",
        oof_files,
        format_func=lambda path: str(path.relative_to(ROOT)),
    )
    df = load_csv(selected)

    st.subheader(str(selected.relative_to(ROOT)))
    st.dataframe(df.head(200), use_container_width=True)

    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    if numeric_columns:
        st.subheader("Numeric Summary")
        st.dataframe(df[numeric_columns].describe().T, use_container_width=True)

        selected_column = st.selectbox("Distribution column", numeric_columns)
        st.bar_chart(df[selected_column].value_counts(dropna=False).sort_index())

    st.subheader("Missing Values")
    missing = df.isna().sum().sort_values(ascending=False)
    st.dataframe(missing[missing > 0].rename("missing_count"), use_container_width=True)


if __name__ == "__main__":
    main()
