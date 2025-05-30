import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import plotly.express as px
from Bio import Entrez

# --- Set page config (must be first Streamlit command) ---
st.set_page_config(page_title="Gene Expression Explorer", layout="wide")

# --- Load model and label encoder ---
model = load("model.joblib")
le = load("label_encoder.joblib")

# --- Load gene expression dataset for reference ---
df = pd.read_csv("sample_gene_expression.csv")
genes = [col for col in df.columns if col != "label"]

# --- NCBI Entrez email (required by NCBI API) ---
Entrez.email = "your_email@example.com"  # Change to your email!

# --- Helper: fetch gene info from NCBI ---
def fetch_gene_info(gene_symbol):
    try:
        handle = Entrez.esearch(db="gene", term=gene_symbol + "[sym]", retmax=1)
        record = Entrez.read(handle)
        handle.close()
        if record["IdList"]:
            gene_id = record["IdList"][0]
            handle = Entrez.efetch(db="gene", id=gene_id, retmode="xml")
            records = Entrez.read(handle)
            handle.close()
            summary = records[0]["Entrezgene_summary"]
            return summary
        else:
            return "No gene information found."
    except Exception as e:
        return f"Error fetching gene info: {e}"

# --- Title ---
st.title("🧬 Gene Expression Explorer")
st.markdown(
    """
    Enter gene expression values or a gene symbol to check your disease risk prediction 
    based on our trained classifier on synthetic data.
    """
)

# --- Sidebar for inputs ---
st.sidebar.header("Input Options")

option = st.sidebar.radio(
    "Choose input mode:",
    ("Manual Gene Expression", "Search by Gene Symbol", "Search by Disease")
)

prediction_result = None

if option == "Manual Gene Expression":
    st.sidebar.write("Enter gene expression values (0 to 1 scale):")
    user_input = {}
    for gene in genes:
        user_input[gene] = st.sidebar.slider(gene, 0.0, 1.0, 0.5)

    if st.sidebar.button("Predict Disease Risk"):
        input_df = pd.DataFrame([user_input])
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]
        prediction_label = le.inverse_transform([prediction])[0]

        # Display prediction
        st.subheader("Prediction Result:")
        st.write(f"Predicted Class: **{prediction_label}**")

        # Plot probabilities
        proba_df = pd.DataFrame({
            "Disease": le.classes_,
            "Probability": proba
        })
        fig = px.bar(proba_df, x="Disease", y="Probability",
                     color="Probability", color_continuous_scale='Viridis',
                     title="Prediction Probabilities")
        st.plotly_chart(fig)

elif option == "Search by Gene Symbol":
    gene_symbol = st.sidebar.text_input("Enter gene symbol (e.g. BRCA1)")

    if st.sidebar.button("Fetch Gene Info"):
        if gene_symbol.strip():
            st.subheader(f"NCBI Gene Info for {gene_symbol.upper()}:")
            info = fetch_gene_info(gene_symbol.upper())
            st.write(info)
        else:
            st.warning("Please enter a valid gene symbol.")

elif option == "Search by Disease":
    disease = st.sidebar.selectbox("Select disease to compare:", le.classes_)

    if st.sidebar.button("Show Gene Expression Profiles"):
        st.subheader(f"Gene Expression Profiles for {disease}")
        subset = df[df["label"] == disease]
        st.write(f"Showing expression values of {len(subset)} samples.")

        # Show mean gene expression for this disease
        mean_expression = subset[genes].mean().sort_values(ascending=False)
        fig = px.bar(x=mean_expression.index, y=mean_expression.values,
                     labels={"x": "Gene", "y": "Mean Expression"},
                     title=f"Average Gene Expression in {disease}")
        st.plotly_chart(fig)

st.markdown("---")
st.write("© 2025 Gene Expression Explorer Hackathon Demo")
