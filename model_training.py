import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from joblib import dump

# --- Generate synthetic gene expression data ---
np.random.seed(42)

num_samples = 100
num_genes = 20

# Gene names like GENE1, GENE2, ..., GENE20
genes = [f"GENE{i}" for i in range(1, num_genes + 1)]

# Simulate gene expression values (random floats)
X = np.random.rand(num_samples, num_genes)

# Simulate labels: 3 classes: Healthy, DiseaseA, DiseaseB
labels = np.random.choice(["Healthy", "DiseaseA", "DiseaseB"], size=num_samples)

# Create DataFrame
df = pd.DataFrame(X, columns=genes)
df['label'] = labels

# Save dataset
df.to_csv("sample_gene_expression.csv", index=False)
print("Sample dataset saved to sample_gene_expression.csv")

# --- Train model ---
X = df[genes]
y = df['label']

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train/test split (not mandatory for demo)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and label encoder
dump(model, "model.joblib")
dump(le, "label_encoder.joblib")
print("Model and label encoder saved!")
