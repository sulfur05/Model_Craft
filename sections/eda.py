
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

MAX_EDA_ROWS = 10000


def dataset_not_available():
    st.info("Upload a dataset first.")
    return


def _outlier_ratio(series):
    arr = series.dropna().values
    if len(arr) == 0:
        return 0.0
    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    iqr = q3 - q1
    if iqr == 0:
        return 0.0
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return ((arr < lower) | (arr > upper)).mean()


def dataset_eda(df, numeric_cols, categorical_cols):

    if len(df) > MAX_EDA_ROWS:
        df_sample = df.sample(MAX_EDA_ROWS, random_state=42)
    else:
        df_sample = df

    # -----------------------------
    # DATASET REPORT CARD
    # -----------------------------

    total_cells = df.shape[0] * df.shape[1]
    missing_ratio = df.isna().sum().sum() / max(total_cells, 1)

    score = 100

    if missing_ratio > 0.20:
        score -= 25
    elif missing_ratio > 0.05:
        score -= 10

    strong_corr_count = 0
    if len(numeric_cols) >= 2:
        corr = df_sample[numeric_cols].corr().abs()
        strong_corr_count = (
            corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            .stack()
            .ge(0.8)
            .sum()
        )
        score -= min(int(strong_corr_count) * 5, 20)

    severe_outliers = 0
    for col in numeric_cols[:]:
        if _outlier_ratio(df_sample[col]) > 0.1:
            severe_outliers += 1

    score -= min(severe_outliers * 3, 15)
    score = max(score, 0)

    st.subheader("+ Dataset Report Card")

    if score >= 85:
        st.success(f"Dataset Readiness Score: {score}/100")
    elif score >= 65:
        st.warning(f"Dataset Readiness Score: {score}/100")
    else:
        st.error(f"Dataset Readiness Score: {score}/100")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Rows", df.shape[0])
        st.metric("Numeric Features", len(numeric_cols))

    with col2:
        st.metric("Columns", df.shape[1])
        st.metric("Categorical Features", len(categorical_cols))

    # -----------------------------
    # KEY FINDINGS
    # -----------------------------

    st.subheader("Key Findings")

    findings = []

    if missing_ratio == 0:
        findings.append("-> No missing values detected.")
    elif missing_ratio < 0.05:
        findings.append("-> Small amount of missing data detected.")
    else:
        findings.append("-> Significant missing data detected.")

    if strong_corr_count > 0:
        findings.append(
            f"-> {strong_corr_count} strongly correlated feature pairs found."
        )
    else:
        findings.append("-> No strong feature redundancy detected.")

    if severe_outliers > 0:
        findings.append(
            f"-> {severe_outliers} feature(s) contain many outliers."
        )
    else:
        findings.append("-> No major outlier issues detected.")

    target_column = st.session_state.get("target_column")

    if target_column and target_column in df.columns:
        if df[target_column].dtype == "object":
            share = df[target_column].value_counts(normalize=True).iloc[0]
            if share > 0.70:
                findings.append(
                    "-> Target column appears heavily imbalanced."
                )
            else:
                findings.append(
                    "-> Target distribution appears healthy."
                )

    for item in findings:
        st.write(item)

    # -----------------------------
    # RECOMMENDATIONS
    # -----------------------------

    st.subheader("ModelCraft Recommends you to (optional)")

    recommendations = []

    if missing_ratio > 0:
        recommendations.append(
            "Use imputation before training."
        )

    if numeric_cols:
        recommendations.append(
            "Apply feature scaling."
        )

    if strong_corr_count > 0:
        recommendations.append(
            "Consider removing redundant correlated features."
        )

    if severe_outliers > 0:
        recommendations.append(
            "Inspect extreme values before training."
        )

    if not recommendations:
        recommendations.append(
            "Your dataset is ready for training."
        )

    for rec in recommendations:
        st.success(f"✓ {rec}")

    # -----------------------------
    # WHY THESE RECOMMENDATIONS?
    # -----------------------------

    with st.expander("Why did ModelCraft make these recommendations?"):

        st.write(
            """
            ModelCraft looks for:
            - Missing values
            - Outliers
            - Feature redundancy
            - Dataset balance
            - Data quality issues

            Based on these checks, it suggests preprocessing
            steps that are commonly used before training ML models.
            """
        )

    # -----------------------------
    # FEATURE EXPLORER
    # -----------------------------

    if numeric_cols:

        with st.expander("Explore Individual Features"):

            feature = st.selectbox(
                "Choose a feature",
                numeric_cols
            )

            series = df_sample[feature]

            outlier_pct = _outlier_ratio(series)

            if outlier_pct < 0.02:
                outlier_health = "Good"
            elif outlier_pct < 0.10:
                outlier_health = "Moderate"
            else:
                outlier_health = "Needs Attention"

            missing_pct = series.isna().mean()

            st.write(f"**Feature:** {feature}")
            st.write(f"**Missing Values:** {missing_pct:.1%}")
            st.write(f"**Outlier Level:** {outlier_health}")

            if missing_pct > 0:
                st.info("Recommendation: Impute missing values.")
            elif outlier_pct > 0.10:
                st.info("Recommendation: Inspect outliers.")
            else:
                st.success("No immediate action required.")

    # -----------------------------
    # ADVANCED VISUALIZATIONS
    # -----------------------------

    with st.expander("📊 Advanced Visualizations (Optional)"):

        st.caption(
            "These visualizations are optional and intended for advanced users."
        )

        if df.isna().sum().sum() > 0:

            st.markdown("### Missing Values")

            missing = df.isna().sum()
            missing = missing[missing > 0]

            fig, ax = plt.subplots()
            missing.sort_values(ascending=False).plot(
                kind="bar",
                ax=ax
            )
            st.pyplot(fig)

        if numeric_cols:

            st.markdown("### Distribution Plots")

            selected = st.multiselect(
                "Select features",
                numeric_cols,
                default=numeric_cols[:2]
            )

            for col in selected:

                fig, ax = plt.subplots()
                sns.histplot(
                    df_sample[col].dropna(),
                    kde=True,
                    ax=ax
                )
                ax.set_title(col)
                st.pyplot(fig)

                fig, ax = plt.subplots()
                sns.boxplot(
                    x=df_sample[col].dropna(),
                    ax=ax
                )
                ax.set_title(f"{col} Outliers")
                st.pyplot(fig)

        if len(numeric_cols) >= 2:

            st.markdown("### Correlation Heatmap")

            corr = df_sample[numeric_cols].corr()

            fig, ax = plt.subplots(
                figsize=(6, 6)
            )

            sns.heatmap(
                corr,
                cmap="coolwarm",
                center=0,
                ax=ax
            )

            st.pyplot(fig)


def eda_section():

    with st.expander("2. Exploratory Data Analysis (EDA)"):

        st.subheader("Explore Your Dataset")

        if "dataset" not in st.session_state:
            dataset_not_available()
            return

        df = st.session_state["dataset"]
        numeric_cols = st.session_state.get(
            "numeric_columns",
            []
        )
        categorical_cols = st.session_state.get(
            "categorical_columns",
            []
        )

        if st.button("Run EDA"):
            dataset_eda(
                df,
                numeric_cols,
                categorical_cols
            )

            st.session_state["eda_complete"] = True
