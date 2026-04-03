import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


MAX_EDA_ROWS = 10_000


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
    else:
        st.write("No categorical columns detected.")

    st.markdown("---")
    st.subheader("Missing values")

    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if missing.empty:
        st.write("No missing values detected.")
    else:
        fig, ax = plt.subplots()
        missing.plot(kind="bar", ax=ax)
        ax.set_ylabel("Number of missing values")
        ax.set_title("Missing values by column")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)

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
            fig, ax = plt.subplots()
            sns.histplot(df_sample[col].dropna(), kde=True, ax=ax)
            ax.set_title(f"Distribution of {col}")
            st.pyplot(fig)

        # Boxplots for outlier inspection
        st.subheader("Boxplots (outlier inspection)")
        box_cols = st.multiselect(
            "Select numeric columns for boxplots",
            options=numeric_cols,
            default=default_numeric,
            key="boxplot_columns",
        )
        for col in box_cols:
            fig, ax = plt.subplots()
            sns.boxplot(x=df_sample[col].dropna(), ax=ax)
            ax.set_title(f"Boxplot of {col}")
            st.pyplot(fig)

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
            st.info(
                "Darker red or blue squares mean stronger relationships between two numeric columns. "
                "Values near 1 or -1 indicate a strong link; values near 0 mean little or no linear relationship."
            )


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