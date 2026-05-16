import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

MAX_EDA_ROWS = 10_000

def show_plot_insight(title: str, insight: str) -> None:
    st.markdown(f"**{title}:** {insight}")

def _skewness(arr: np.ndarray) -> float:
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return 0.0
    m = arr.mean()
    s = arr.std(ddof=0)
    if s == 0:
        return 0.0
    return float(((arr - m) ** 3).mean() / (s**3))

def _hist_modality(arr: np.ndarray, bins: int = 30) -> str:
    arr = arr[~np.isnan(arr)]
    if arr.size < 10:
        return "insufficient data to judge modality"
    counts, _ = np.histogram(arr, bins=bins)
    peaks = 0
    for i in range(1, len(counts) - 1):
        if counts[i] > counts[i - 1] and counts[i] > counts[i + 1]:
            peaks += 1
    if peaks <= 1:
        return "appears unimodal"
    if peaks == 2:
        return "may be bimodal"
    return "appears multimodal"

def _outlier_stats(arr: np.ndarray) -> (int, float):
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return 0, 0.0
    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    iqr = q3 - q1
    if iqr == 0:
        return 0, 0.0
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = ((arr < lower) | (arr > upper)).sum()
    return int(outliers), float(outliers / arr.size)

def _top_categorical_info(series: pd.Series) -> (str, float, int):
    vc = series.value_counts(dropna=False)
    if vc.empty:
        return "no values", 0.0, 0
    top = vc.index[0]
    freq = int(vc.iloc[0])
    share = float(freq / series.size)
    unique = series.nunique(dropna=True)
    return str(top), share, int(unique)

def dataset_not_available():
    st.info("Upload a dataset in step 1 (Dataset Upload) first.")
    return

def dataset_eda(df: pd.DataFrame, numeric_cols, categorical_cols):
    # Sampling for performance
    if len(df) > MAX_EDA_ROWS:
        df_sample = df.sample(MAX_EDA_ROWS, random_state=42)
        st.caption(
            f"Showing EDA on a random sample of {MAX_EDA_ROWS} rows "
            f"out of {len(df)} to keep things responsive."
        )
    else:
        df_sample = df

    st.subheader("Summary")
    st.write(f"- Rows: {df.shape[0]}")
    st.write(f"- Columns: {df.shape[1]}")

    # Numeric summary
    if numeric_cols:
        st.markdown("**Numeric columns summary**")
        st.dataframe(df[numeric_cols].describe().T)
    else:
        st.write("No numeric columns detected.")

    # Categorical summary (per-column value counts on demand)
    if categorical_cols:
        st.markdown("**Categorical column value counts**")
        cat_col = st.selectbox(
            "Choose a categorical column to see its most common values",
            options=categorical_cols,
        )
        vc = df[cat_col].value_counts(dropna=False).head(20)
        st.write(vc)
        top, share, unique = _top_categorical_info(df[cat_col])
        # Intuitive interpretation
        if df[cat_col].size > 0:
            if share > 0.75:
                intuition = (
                    f"Most rows ({share:.0%}) belong to '{top}'. The model may learn this category easily — "
                    "but check for imbalance before training."
                )
            elif unique > 30:
                intuition = (
                    f"There are many different values ({unique}) in {cat_col}. Consider grouping rare values."
                )
            else:
                intuition = f"Top category is '{top}' ({share:.0%}). Watch for class imbalance."
            show_plot_insight("Interpretation", intuition)

            # Mathematical details behind an expander
            with st.expander("Click for mathematical interpretation"):
                st.write(f"Top category: `{top}`")
                st.write(f"Share of rows: {share:.3f}")
                st.write(f"Unique non-null categories: {unique}")

    st.markdown("---")
    st.subheader("Missing values")

    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if missing.empty:
        st.write("No missing values detected.")
        show_plot_insight("Interpretation", "No missing values detected — no imputation needed.")
    else:
        fig, ax = plt.subplots()
        missing.plot(kind="bar", ax=ax)
        ax.set_ylabel("Number of missing values")
        ax.set_title("Missing values by column")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)

        pct_missing = (missing / len(df)).sort_values(ascending=False)
        high_missing = pct_missing[pct_missing > 0.2].index.tolist()
        moderate_missing = pct_missing[(pct_missing > 0.05) & (pct_missing <= 0.2)].index.tolist()

        if high_missing:
            intuition = (
                f"Columns {', '.join(high_missing)} have a lot of missing values (>20%). "
                "They may need dropping or careful imputation."
            )
        elif moderate_missing:
            intuition = (
                f"Columns {', '.join(moderate_missing)} have moderate missingness (5–20%). Consider imputation."
            )
        else:
            intuition = "Missing values exist but are relatively low; simple imputation may be sufficient."
        show_plot_insight("Interpretation", intuition)

        with st.expander("Click for mathematical interpretation"):
            st.write("Missing values per column (counts and percent):")
            for col, cnt in missing.items():
                st.write(f"- `{col}`: {cnt} missing ({cnt/len(df):.2%})")

    # Numeric distributions
    if numeric_cols:
        st.markdown("---")
        st.subheader("Numeric distributions")

        default_numeric = numeric_cols[:4]
        selected_numeric = st.multiselect(
            "Select numeric columns to plot",
            options=numeric_cols,
            default=default_numeric,
        )

        for col in selected_numeric:
            arr = df_sample[col].dropna().to_numpy(dtype=float)
            fig, ax = plt.subplots()
            sns.histplot(arr, kde=True, ax=ax)
            ax.set_title(f"Distribution of {col}")
            st.pyplot(fig)

            skew = _skewness(arr)
            modality = _hist_modality(arr)
            # Intuitive interpretation (non-technical)
            if abs(skew) < 0.5:
                skew_msg = "looks fairly symmetric"
            elif skew > 0:
                skew_msg = "leans right (longer tail to larger values)"
            else:
                skew_msg = "leans left (longer tail to smaller values)"

            intuition = (
                f"The distribution of {col} {skew_msg}. {modality}. "
                "If needed, consider simple transforms (log) or scaling before modeling."
            )
            show_plot_insight("Interpretation", intuition)

            with st.expander("Click for mathematical interpretation"):
                st.write(f"Skewness (Pearson): {skew:.3f}")
                st.write(f"Modality assessment: {modality}")

        # Boxplots for outlier inspection
        st.subheader("Boxplots (outlier inspection)")
        box_cols = st.multiselect(
            "Select numeric columns for boxplots",
            options=numeric_cols,
            default=default_numeric,
            key="boxplot_columns",
        )
        for col in box_cols:
            arr = df_sample[col].dropna().to_numpy(dtype=float)
            fig, ax = plt.subplots()
            sns.boxplot(x=arr, ax=ax)
            ax.set_title(f"Boxplot of {col}")
            st.pyplot(fig)

            outliers, outlier_pct = _outlier_stats(arr)
            if outliers == 0:
                intuition = f"No strong outliers detected in {col}."
            else:
                intuition = (
                    f"There are {outliers} values that differ noticeably from the bulk of {col} ({outlier_pct:.1%}). "
                    "Inspect those rows — they may be data errors or rare but valid cases."
                )
            show_plot_insight("Interpretation", intuition)

            with st.expander("Click for mathematical interpretation"):
                st.write(f"Outliers by 1.5*IQR rule: {outliers}")
                st.write(f"Outlier share: {outlier_pct:.3f}")

        # Correlation heatmap
        if len(numeric_cols) >= 2:
            st.subheader("Correlation heatmap (numeric columns)")
            corr = df_sample[numeric_cols].corr()
            fig, ax = plt.subplots(
                figsize=(
                    min(0.6 * len(numeric_cols), 6),
                    min(0.6 * len(numeric_cols), 6),
                )
            )
            sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
            ax.set_title("Correlation heatmap")
            st.pyplot(fig)

            corr_pairs = (
                corr.abs()
                .where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                .stack()
                .sort_values(ascending=False)
            )
            strong = corr_pairs[corr_pairs >= 0.7]
            if not strong.empty:
                pairs = [f"{a} & {b} ({r:.2f})" for (a, b), r in strong.items()]
                intuition = (
                    f"Some features show strong linear relationships: {', '.join(pairs)}. "
                    "Consider removing or combining redundant columns."
                )
            else:
                intuition = "No very strong linear correlations detected between numeric features."
            show_plot_insight("Interpretation", intuition)

            with st.expander("Click for mathematical interpretation"):
                st.write("Top absolute correlation pairs:")
                for (a, b), r in corr_pairs.head(10).items():
                    st.write(f"- {a} & {b}: {r:.3f}")

def eda_section():
    with st.expander("2. Exploratory Data Analysis (EDA)"):
        st.subheader("Explore your data")

        if "dataset" not in st.session_state:
            dataset_not_available()
            return

        df = st.session_state["dataset"]
        numeric_cols = st.session_state.get("numeric_columns", [])
        categorical_cols = st.session_state.get("categorical_columns", [])

        st.write(
            "Click the button below to generate summary statistics and visualizations "
            "for your dataset."
        )

        run_eda = st.button("Run EDA")

        if not run_eda:
            return

        dataset_eda(df, numeric_cols, categorical_cols)