import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap as sp
import sklearn
from sklearn.linear_model import LinearRegression

def explainability_section():
    #here we will include the fucntionality

    with st.expander("5. Explainability section"):

        if "trained_model" not in st.session_state:
            st.info("No trained model in session_state")
        else:
            model = st.session_state["trained_model"]
            st.write("Model type:", model)

            if isinstance(model, LinearRegression):
                st.write("this is Linear Regression")

                if "preprocessor" not in st.session_state or "X_train" not in st.session_state:
                    st.info("Run preprocessing before using explainability.")
                    return

                X_train = st.session_state["X_train"]
                preprocessor = st.session_state["preprocessor"]
                X_train_proc = preprocessor.transform(X_train)
                
                feature_names = preprocessor.get_feature_names_out()
                X_train_proc = pd.DataFrame(X_train_proc, columns=feature_names)
                
                explainer = sp.LinearExplainer(model, X_train_proc)
                shap_values = explainer(X_train_proc)
                
                fig1 = plt.figure()
                sp.plots.bar(shap_values, max_display=15, show=False)
                st.pyplot(fig1, clear_figure=True)
                
                fig2 = plt.figure()
                sp.plots.beeswarm(shap_values, max_display=15, show=False)
                st.pyplot(fig2, clear_figure=True)
            else:
                return
 
    



explainability_section()