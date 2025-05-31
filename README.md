🧬 Gene Expression Explorer
🔍 Overview
Gene Expression Explorer is a Streamlit-based interactive web application designed to:

🔬 Predict disease risk using synthetic gene expression data and a trained machine learning model.

🧠 Visualize gene expression compared to healthy individuals.

🧬 Fetch gene information directly from NCBI using gene symbols.

📊 Explore disease-specific gene expression profiles.

This app is built as a hackathon project to demonstrate practical applications of bioinformatics and machine learning in genomics.

🚀 Features
🧪 Disease Risk Prediction
Input gene expression values:

Full gene set (20 synthetic genes)

Selective gene input

Upload CSV files for batch prediction

Predict risk using a trained classifier

Visual comparison of input gene expression with healthy average

Probability distribution over predicted disease classes

🔍 Gene Symbol Lookup
Search a gene symbol and retrieve its description from NCBI Entrez

Includes fallback to synthetic descriptions for demo genes

🧬 Disease-wise Gene Profile
Select a disease to:

View mean expression of all genes for that disease

Compare with healthy gene expression profile (NEWLY ADDED)

📈 Interactive Visualizations
Plotly-based charts for:

Class probabilities

Expression comparison

Disease-specific gene profiles

🧰 Tech Stack
Frontend/UI: Streamlit

Machine Learning: scikit-learn (trained classifier, label encoder)

Data Processing: pandas, numpy

Visualization: Plotly

Bioinformatics API: Biopython (Entrez for NCBI access)

Model Format: joblib

🛠️ Setup Instructions
1. Clone the repository

git clone https://github.com/your-username/gene-expression-explorer.git
cd gene-expression-explorer
2. Install dependencies

pip install -r requirements.txt
3. Add model and data files
Ensure the following files are placed in the root directory:

model.joblib – Trained classifier

label_encoder.joblib – Label encoder for disease classes

sample_gene_expression.csv – Gene expression dataset with labels

4. Set your email in the code (for NCBI Entrez)
Edit the following line in the code with your email:

Entrez.email = "your_email@example.com"
5. Run the app

streamlit run app.py