import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

MAX_EDA_ROWS = 10_000


def show_eda_card(title: str, insight: str, action: str = None):
    st.info(f"📌 {title}\n\n{insight}")
    if action:
        st.success(f"💡 Recommended Action: {action}")


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


def _outlier_stats(arr: np.ndarray):
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


def _top_categorical_info(series: pd.Series):
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

    if len(df) > MAX_EDA_ROWS:
        df_sample = df.sample(MAX_EDA_ROWS, random_state=42)
        st.caption(
            f"Showing EDA on a random sample of {MAX_EDA_ROWS} rows out of {len(df)} rows."
        )
    else:
        df_sample = df

    st.subheader("🏥 Dataset Health Report")

    total_missing = df.isna().sum().sum()
    missing_ratio = total_missing / (df.shape[0] * df.shape[1])

    if missing_ratio < 0.05:
        st.success("🟢 Missing Values: Low")
    elif missing_ratio < 0.20:
        st.warning("🟡 Missing Values: Moderate")
    else:
        st.error("🔴 Missing Values: High")

    st.info(f"📊 Numeric Features: {len(numeric_cols)}")
    st.info(f"📋 Categorical Features: {len(categorical_cols)}")

    st.subheader("Summary")
    st.write(f"- Rows: {df.shape[0]}")
    st.write(f"- Columns: {df.shape[1]}")

    with st.expander("📑 View Statistical Summary"):
        if numeric_cols:
            st.dataframe(df[numeric_cols].describe().T)

    if categorical_cols:
        st.markdown("### Categorical Analysis")

        cat_col = st.selectbox(
            "Choose a categorical column",
            options=categorical_cols,
        )

        vc = df[cat_col].value_counts(dropna=False).head(20)
        st.write(vc)

        top, share, unique = _top_categorical_info(df[cat_col])

        if share > 0.75:
            insight = (
                f"Most rows ({share:.0%}) belong to '{top}'. "
                f"This feature may be highly imbalanced."
            )
        elif unique > 30:
            insight = (
                f"{cat_col} contains many unique values ({unique}). "
                f"Rare categories may need grouping."
            )
        else:
            insight = (
                f"Top category is '{top}' ({share:.0%} of rows)."
            )

        show_eda_card(
            "Categorical Feature Insights",
            insight,
            "Check class balance before training."
        )

        target_column = st.session_state.get("target_column")
        if target_column == cat_col and share > 0.7:
            st.warning(
                f"⚠️ Class imbalance detected. {share:.0%} of samples belong to '{top}'."
            )

        with st.expander("🔬 Advanced Statistics"):
            st.write(f"Top category: {top}")
            st.write(f"Share: {share:.3f}")
            st.write(f"Unique categories: {unique}")

    st.markdown("---")
    st.subheader("Missing Values")

    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if missing.empty:
        show_eda_card(
            "Missing Value Analysis",
            "No missing values detected.",
            "No imputation required."
        )
    else:

        pct_missing = (missing / len(df)).sort_values(ascending=False)

        high_missing = pct_missing[pct_missing > 0.2].index.tolist()
        moderate_missing = pct_missing[
            (pct_missing > 0.05) & (pct_missing <= 0.2)
        ].index.tolist()

        if high_missing:
            intuition = (
                f"Columns {', '.join(high_missing)} contain high missingness."
            )
        elif moderate_missing:
            intuition = (
                f"Columns {', '.join(moderate_missing)} contain moderate missingness."
            )
        else:
            intuition = "Missing values are present but relatively low."

        show_eda_card(
            "Missing Value Analysis",
            intuition,
            "Apply imputation before model training."
        )

        with st.expander("📊 View Missing Value Graph"):
            fig, ax = plt.subplots()
            missing.plot(kind="bar", ax=ax)
            ax.set_ylabel("Missing Values")
            ax.set_title("Missing Values by Column")
            plt.xticks(rotation=45, ha="right")
            st.pyplot(fig)

    if numeric_cols:

        st.markdown("---")
        st.subheader("📈 Feature Distributions")

        default_numeric = numeric_cols[:4]

        selected_numeric = st.multiselect(
            "Select numeric columns",
            options=numeric_cols,
            default=default_numeric,
        )

        for col in selected_numeric:

            arr = df_sample[col].dropna().to_numpy(dtype=float)

            skew = _skewness(arr)
            modality = _hist_modality(arr)

            if abs(skew) < 0.5:
                skew_msg = "looks fairly symmetric"
            elif skew > 0:
                skew_msg = "leans right"
            else:
                skew_msg = "leans left"

            intuition = (
                f"{col} {skew_msg} and {modality}."
            )

            action = (
                "No transformation needed."
                if abs(skew) < 0.5
                else "Consider scaling or log transformation."
            )

            show_eda_card(
                f"{col} Distribution",
                intuition,
                action
            )

            with st.expander(f"📊 View Distribution Graph ({col})"):
                fig, ax = plt.subplots()
                sns.histplot(arr, kde=True, ax=ax)
                ax.set_title(f"Distribution of {col}")
                st.pyplot(fig)

            with st.expander(f"🔬 Advanced Statistics ({col})"):
                st.write(f"Skewness: {skew:.3f}")
                st.write(f"Modality: {modality}")

        st.markdown("---")
        st.subheader("🎯 Outlier Analysis")

        box_cols = st.multiselect(
            "Select numeric columns for outlier inspection",
            options=numeric_cols,
            default=default_numeric,
            key="boxplot_columns",
        )

        for col in box_cols:

            arr = df_sample[col].dropna().to_numpy(dtype=float)

            outliers, outlier_pct = _outlier_stats(arr)

            if outliers == 0:
                insight = f"No strong outliers detected in {col}."
            else:
                insight = (
                    f"{outliers} potential outliers detected "
                    f"({outlier_pct:.1%} of data)."
                )

            show_eda_card(
                f"{col} Outlier Analysis",
                insight,
                "Inspect extreme values before training."
            )

            with st.expander(f"📊 View Boxplot ({col})"):
                fig, ax = plt.subplots()
                sns.boxplot(x=arr, ax=ax)
                ax.set_title(f"Boxplot of {col}")
                st.pyplot(fig)

        if len(numeric_cols) >= 2:

            st.markdown("---")
            st.subheader("🔗 Feature Relationships")

            corr = df_sample[numeric_cols].corr()

            corr_pairs = (
                corr.abs()
                .where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                .stack()
                .sort_values(ascending=False)
            )

            strong = corr_pairs[corr_pairs >= 0.7]

            if not strong.empty:
                pairs = [
                    f"{a} & {b} ({r:.2f})"
                    for (a, b), r in strong.head(5).items()
                ]

                insight = (
                    "Strong correlations detected: "
                    + ", ".join(pairs)
                )
            else:
                insight = "No strong feature correlations detected."

            show_eda_card(
                "Correlation Analysis",
                insight,
                "Remove redundant features if model performance suffers."
            )

            with st.expander("📊 View Correlation Heatmap"):
                fig, ax = plt.subplots(
                    figsize=(
                        min(0.6 * len(numeric_cols), 6),
                        min(0.6 * len(numeric_cols), 6),
                    )
                )

                sns.heatmap(
                    corr,
                    cmap="coolwarm",
                    center=0,
                    ax=ax
                )

                ax.set_title("Correlation Heatmap")
                st.pyplot(fig)

    st.markdown("---")
    st.subheader("🤖 ModelCraft Recommendations")

    recommendations = []

    if total_missing > 0:
        recommendations.append("Use imputation for missing values.")

    if numeric_cols:
        recommendations.append("Apply feature scaling before training.")

    if recommendations:
        for rec in recommendations:
            st.success(f"✓ {rec}")
    else:
        st.success("Your dataset looks ready for training.")


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
            "Generate beginner-friendly insights about your dataset."
        )

        run_eda = st.button("Run EDA")

        if not run_eda:
            return

        dataset_eda(df, numeric_cols, categorical_cols)
