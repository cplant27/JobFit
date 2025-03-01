import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sentence_transformers import InputExample, SentenceTransformer, losses, util
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load dataset
splits = {"train": "train.csv", "test": "test.csv"}
df_train = pd.read_csv(
    "hf://datasets/cnamuangtoun/resume-job-description-fit/" + splits["train"]
)
df_test = pd.read_csv(
    "hf://datasets/cnamuangtoun/resume-job-description-fit/" + splits["test"]
)

# Convert labels to numerical values
label_mapping = {"No Fit": 0, "Potential Fit": 1, "Good Fit": 2}
df_train["label"] = df_train["label"].map(label_mapping)
df_test["label"] = df_test["label"].map(label_mapping)

# Load model on GPU
model = SentenceTransformer("all-MiniLM-L6-v2").to(device)

# Prepare training data
train_examples = [
    InputExample(
        texts=[row["resume_text"], row["job_description_text"]],
        label=float(row["label"]),
    )
    for _, row in tqdm(
        df_train.iterrows(), total=len(df_train), desc="Preparing training data"
    )
]
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.CosineSimilarityLoss(model)

# Fine-tune model
print("Fine-tuning model...")
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=1,
    warmup_steps=100,
    show_progress_bar=True,
)


# Compute similarity
def compute_similarity(row):
    embedding1 = model.encode(row["resume_text"], convert_to_tensor=True, device=device)
    embedding2 = model.encode(
        row["job_description_text"], convert_to_tensor=True, device=device
    )
    similarity = util.cos_sim(embedding1, embedding2).item()
    return similarity


df_test["similarity"] = [
    compute_similarity(row)
    for _, row in tqdm(
        df_test.iterrows(), total=len(df_test), desc="Computing similarities"
    )
]


# Define threshold for classification
def classify_fit(similarity):
    if similarity > 0.7:
        return 2  # Good Fit
    elif similarity > 0.4:
        return 1  # Potential Fit
    else:
        return 0  # No Fit


df_test["predicted_label"] = [
    classify_fit(sim) for sim in tqdm(df_test["similarity"], desc="Classifying job fit")
]

# Evaluation Metrics
actual = df_test["label"].values
predicted = df_test["predicted_label"].values

# Compute confusion matrix
conf_matrix = confusion_matrix(actual, predicted)

# Display confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Fit", "Potential Fit", "Good Fit"],
    yticklabels=["No Fit", "Potential Fit", "Good Fit"],
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Classification report
report = classification_report(
    actual,
    predicted,
    target_names=["No Fit", "Potential Fit", "Good Fit"],
    output_dict=True,
)
report_df = pd.DataFrame(report).transpose()
print(report_df)

# Save results
df_test.to_csv("output_with_similarity.csv", index=False)
