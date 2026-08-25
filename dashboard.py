"""
Customer Churn Risk Dashboard
Run with:  streamlit run dashboard.py

Requires: streamlit, shap  (pip install streamlit shap)
Expects model.pkl (produced by the notebook) in the same folder.
"""

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Churn Risk Dashboard", layout="wide")

NUMERICAL_COLS = [
    "Number vmail messages", "Total day minutes", "Total day calls",
    "Total eve minutes", "Total eve calls", "Total night minutes",
    "Total night calls", "Total intl minutes", "Total intl calls",
    "Customer service calls",
]
CATEGORICAL_COLS = ["International plan", "Voice mail plan"]
ALL_COLS = NUMERICAL_COLS + CATEGORICAL_COLS


@st.cache_resource
def load_model():
    return joblib.load("model.pkl")


@st.cache_data
def load_reference_data():
    # Used only to set sensible slider ranges / defaults and as a SHAP background sample.
    df = pd.read_csv("churn-bigml-80-checkpoint.csv")
    return df


@st.cache_resource
def get_explainer(_model):
    # TreeExplainer works directly on the fitted RandomForestClassifier inside the pipeline.
    rf_model = _model.named_steps["model"]
    return shap.TreeExplainer(rf_model)


def build_input_row(values: dict) -> pd.DataFrame:
    return pd.DataFrame([values], columns=ALL_COLS)


def main():
    st.title("Customer Churn Risk Dashboard")
    st.caption("Enter a customer's usage profile to get a churn risk score and see which factors drove it.")

    model = load_model()
    ref = load_reference_data()
    explainer = get_explainer(model)

    st.sidebar.header("Customer profile")
    values = {}
    values["International plan"] = st.sidebar.selectbox("International plan", ["No", "Yes"])
    values["Voice mail plan"] = st.sidebar.selectbox("Voice mail plan", ["No", "Yes"])
    values["Customer service calls"] = st.sidebar.slider("Customer service calls", 0, 10, 1)
    values["Number vmail messages"] = st.sidebar.slider("Number vmail messages", 0, 60, 0)
    values["Total day minutes"] = st.sidebar.slider("Total day minutes", 0.0, 400.0, 180.0)
    values["Total day calls"] = st.sidebar.slider("Total day calls", 0, 170, 100)
    values["Total eve minutes"] = st.sidebar.slider("Total eve minutes", 0.0, 400.0, 200.0)
    values["Total eve calls"] = st.sidebar.slider("Total eve calls", 0, 170, 100)
    values["Total night minutes"] = st.sidebar.slider("Total night minutes", 0.0, 400.0, 200.0)
    values["Total night calls"] = st.sidebar.slider("Total night calls", 0, 170, 100)
    values["Total intl minutes"] = st.sidebar.slider("Total intl minutes", 0.0, 20.0, 10.0)
    values["Total intl calls"] = st.sidebar.slider("Total intl calls", 0, 20, 4)

    input_row = build_input_row(values)

    proba = model.predict_proba(input_row)[0, 1]
    pred = model.predict(input_row)[0]

    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Churn risk score", f"{proba:.0%}")
        st.write("**Predicted label:**", "Likely to churn" if pred == 1 else "Likely to stay")
        if proba >= 0.5:
            st.warning("High risk — flag for retention outreach.")
        else:
            st.success("Low risk.")

    with col2:
        st.subheader("Why this score? (SHAP)")
        # Transform the single row through the pipeline's preprocessor,
        # then explain the tree model's output on the transformed features.
        preprocessor = model.named_steps["preprocessor"]
        transformed = preprocessor.transform(input_row)
        cat_names = preprocessor.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_COLS)
        feature_names = NUMERICAL_COLS + list(cat_names)

        shap_values = explainer.shap_values(transformed)
        # shap's return shape has changed across versions:
        #  - older versions: list [class0_array, class1_array], each (n_samples, n_features)
        #  - newer versions: single array (n_samples, n_features, n_classes)
        #  - binary-only versions: single array (n_samples, n_features)
        # Normalize all of these down to a 1D array of per-feature SHAP values
        # for the single input row, class 1 (churn).
        if isinstance(shap_values, list):
            sv = np.asarray(shap_values[1])[0]
        else:
            arr = np.asarray(shap_values)
            if arr.ndim == 3:
                sv = arr[0, :, 1]
            else:
                sv = arr[0]

        fig, ax = plt.subplots(figsize=(7, 5))
        order = np.argsort(np.abs(sv))[::-1][:8]
        colors = ["#C44E52" if sv[i] > 0 else "#4C72B0" for i in order]
        ax.barh([feature_names[i] for i in order][::-1], sv[order][::-1], color=colors[::-1])
        ax.set_xlabel("Impact on churn probability (SHAP value)")
        ax.set_title("Top factors driving this prediction")
        st.pyplot(fig)
        st.caption("Red = pushes risk up. Blue = pushes risk down.")


if __name__ == "__main__":
    main()