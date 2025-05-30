import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import plotly.express as px
from Bio import Entrez

# --- Set page config ---
st.set_page_config(page_title="Gene Expression Explorer", layout="wide")

# --- Load model and label encoder ---
model = load("model.joblib")
le = load("label_encoder.joblib")

# --- Load gene expression dataset for reference ---
df = pd.read_csv("sample_gene_expression.csv")
genes = [col for col in df.columns if col != "label"]

# --- NCBI Entrez email ---
Entrez.email = "your_email@example.com"  # Change to your email!

# --- Synthetic gene info dictionary ---
synthetic_gene_info = {f"GENE{i}": f"Synthetic description for gene GENE{i}." for i in range(1, 21)}

# --- Helper to fetch gene info ---
def fetch_gene_info(gene_symbol):
    if gene_symbol in synthetic_gene_info:
        return synthetic_gene_info[gene_symbol]
    try:
        handle = Entrez.esearch(db="gene", term=gene_symbol + "[sym]", retmax=1)
        record = Entrez.read(handle)
        handle.close()
        if record["IdList"]:
            gene_id = record["IdList"][0]
            handle = Entrez.efetch(db="gene", id=gene_id, retmode="xml")
            records = Entrez.read(handle)
            handle.close()
            summary = records[0].get("Entrezgene_summary", "")
            if summary:
                return summary
            else:
                return "No summary available for this gene."
        else:
            return "No gene information found."
    except Exception as e:
        return f"Error fetching gene info: {e}"

# --- Title ---
st.title("🧬 Gene Expression Explorer")
st.markdown("""
Enter gene expression values or a gene symbol to check your disease risk prediction 
based on our trained classifier on synthetic data.
""")



# --- Sidebar ---
st.sidebar.header("Input Options")

option = st.sidebar.radio(
    "Choose input mode:",
    ("Manual Gene Expression", "Search by Gene Symbol", "Search by Disease")
)

# --------------------------- MANUAL INPUT --------------------------------
if option == "Manual Gene Expression":
    st.sidebar.subheader("Input Mode")
    input_mode = st.sidebar.selectbox("Select input method:",
                                      ["Full Gene Input", "Selective Gene Input", "Upload CSV File"])

    if input_mode == "Full Gene Input":
        st.sidebar.write("Enter gene expression values (0 to 1 scale):")
        user_input = {}
        for gene in genes:
            user_input[gene] = st.sidebar.slider(gene, 0.0, 1.0, 0.5)

        if st.sidebar.button("Predict Disease Risk"):
            input_df = pd.DataFrame([user_input])
            prediction = model.predict(input_df)[0]
            proba = model.predict_proba(input_df)[0]
            prediction_label = le.inverse_transform([prediction])[0]

            st.subheader("Prediction Result:")
            st.write(f"Predicted Class: **{prediction_label}**")

            proba_df = pd.DataFrame({
                "Disease": le.classes_,
                "Probability": proba
            })
            fig = px.bar(proba_df, x="Disease", y="Probability",
                         color="Probability", color_continuous_scale='Viridis',
                         title="Prediction Probabilities")
            st.plotly_chart(fig)

            # Comparison with healthy samples
            if "Healthy" in le.classes_:
                healthy_samples = df[df["label"] == "Healthy"]
                if not healthy_samples.empty:
                    mean_healthy_expr = healthy_samples[genes].mean()
                    user_expr_series = pd.Series(user_input)

                    comp_df = pd.DataFrame({
                        "Gene": genes,
                        "Your Expression": user_expr_series.values,
                        "Healthy Average": mean_healthy_expr.values
                    }).set_index("Gene")

                    comp_df_plot = comp_df.reset_index().melt(id_vars="Gene",
                                                              var_name="Expression Type",
                                                              value_name="Expression Value")
                    fig2 = px.bar(comp_df_plot, x="Gene", y="Expression Value",
                                  color="Expression Type", barmode="group",
                                  title="Your Gene Expression vs Healthy Average")
                    st.plotly_chart(fig2)
                else:
                    st.warning("No healthy samples found in the dataset for comparison.")
            else:
                st.info("Healthy class not found in label classes for comparison.")

    elif input_mode == "Selective Gene Input":
        selected_genes = st.sidebar.multiselect("Select genes to input expression for:", genes)
        if selected_genes:
            user_input = {}
            for gene in selected_genes:
                user_input[gene] = st.sidebar.slider(gene, 0.0, 1.0, 0.5)

            fill_genes = list(set(genes) - set(selected_genes))
            mean_expr = df[genes].mean()

            for gene in fill_genes:
                user_input[gene] = mean_expr[gene]

            if st.sidebar.button("Predict Disease Risk"):
                input_df = pd.DataFrame([user_input])
                prediction = model.predict(input_df)[0]
                proba = model.predict_proba(input_df)[0]
                prediction_label = le.inverse_transform([prediction])[0]

                st.subheader("Prediction Result:")
                st.write(f"Predicted Class: **{prediction_label}**")

                proba_df = pd.DataFrame({
                    "Disease": le.classes_,
                    "Probability": proba
                })
                fig = px.bar(proba_df, x="Disease", y="Probability",
                             color="Probability", color_continuous_scale='Viridis',
                             title="Prediction Probabilities")
                st.plotly_chart(fig)

                if "Healthy" in le.classes_:
                    healthy_samples = df[df["label"] == "Healthy"]
                    if not healthy_samples.empty:
                        mean_healthy_expr = healthy_samples[genes].mean()
                        user_expr_series = pd.Series(user_input)

                        comp_df = pd.DataFrame({
                            "Gene": genes,
                            "Your Expression": user_expr_series.values,
                            "Healthy Average": mean_healthy_expr.values
                        }).set_index("Gene")

                        comp_df_plot = comp_df.reset_index().melt(id_vars="Gene",
                                                                  var_name="Expression Type",
                                                                  value_name="Expression Value")
                        fig2 = px.bar(comp_df_plot, x="Gene", y="Expression Value",
                                      color="Expression Type", barmode="group",
                                      title="Your Gene Expression vs Healthy Average")
                        st.plotly_chart(fig2)
                    else:
                        st.warning("No healthy samples found in the dataset for comparison.")
                else:
                    st.info("Healthy class not found in label classes for comparison.")
        else:
            st.sidebar.info("Please select at least one gene to input values.")

    elif input_mode == "Upload CSV File":
        uploaded_file = st.sidebar.file_uploader("Upload CSV with gene expression data", type=["csv"])
        if uploaded_file is not None:
            try:
                uploaded_df = pd.read_csv(uploaded_file)
                st.write("Preview of uploaded data:")
                st.dataframe(uploaded_df.head())

                missing_genes = [g for g in genes if g not in uploaded_df.columns]
                if missing_genes:
                    st.error(f"Uploaded file is missing these gene columns: {missing_genes}")
                else:
                    if st.sidebar.button("Predict Disease Risk for Uploaded Data"):
                        predictions = model.predict(uploaded_df[genes])
                        probabilities = model.predict_proba(uploaded_df[genes])

                        pred_labels = le.inverse_transform(predictions)
                        results_df = uploaded_df.copy()
                        results_df["Predicted Class"] = pred_labels

                        st.subheader("Batch Prediction Results:")
                        st.dataframe(results_df[genes + ["Predicted Class"]])

                        st.write("Prediction class counts:")
                        st.bar_chart(pd.Series(pred_labels).value_counts())

                        # Show comparison plots for first few samples
                        if "Healthy" in le.classes_:
                            healthy_samples = df[df["label"] == "Healthy"]
                            if not healthy_samples.empty:
                                mean_healthy_expr = healthy_samples[genes].mean()
                                st.subheader("Graphical Comparison with Healthy Average (First 3 Samples)")
                                for idx, row in uploaded_df.iterrows():
                                    if idx >= 3:
                                        break
                                    user_expr_series = row[genes]

                                    comp_df = pd.DataFrame({
                                        "Gene": genes,
                                        "Sample Expression": user_expr_series.values,
                                        "Healthy Average": mean_healthy_expr.values
                                    }).set_index("Gene")

                                    comp_df_plot = comp_df.reset_index().melt(id_vars="Gene",
                                                                              var_name="Expression Type",
                                                                              value_name="Expression Value")

                                    fig2 = px.bar(comp_df_plot, x="Gene", y="Expression Value",
                                                  color="Expression Type", barmode="group",
                                                  title=f"Sample {idx+1} vs Healthy Average")
                                    st.plotly_chart(fig2)
                            else:
                                st.warning("No healthy samples found for comparison.")
                        else:
                            st.info("Healthy class not found in label classes for comparison.")
            except Exception as e:
                st.error(f"Error reading uploaded file: {e}")

# --------------------------- GENE SYMBOL SEARCH ---------------------------
elif option == "Search by Gene Symbol":
    gene_options = [""] + sorted(genes)
    gene_symbol = st.sidebar.selectbox("Enter gene symbol (autocomplete):", gene_options)

    if st.sidebar.button("Fetch Gene Info"):
        if gene_symbol:
            st.subheader(f"NCBI Gene Info for {gene_symbol.upper()}:")
            info = fetch_gene_info(gene_symbol.upper())
            st.write(info)
        else:
            st.warning("Please select or enter a valid gene symbol.")

# --------------------------- DISEASE SEARCH ---------------------------
elif option == "Search by Disease":
    disease = st.sidebar.selectbox("Select disease to compare:", le.classes_)

    if st.sidebar.button("Show Gene Expression Profiles"):
        st.subheader(f"Gene Expression Profiles for {disease}")
        subset = df[df["label"] == disease]
        st.write(f"Showing expression values of {len(subset)} samples.")

        mean_expression = subset[genes].mean().sort_values(ascending=False)
        fig = px.bar(x=mean_expression.index, y=mean_expression.values,
                     labels={"x": "Gene", "y": "Mean Expression"},
                     title=f"Average Gene Expression in {disease}")
        st.plotly_chart(fig)

st.markdown("---")
st.write("© 2025 Gene Expression Explorer Hackathon Demo")
