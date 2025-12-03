# train_critic.py (UPGRADED VERSION)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer
import joblib
import os

print("--- Critic Model Training Script (Upgraded with Embeddings) ---")

# --- 1. Load and Prepare Data ---
THUMBS_UP_FILE = "thumbs_up_log.xlsx"
THUMBS_DOWN_FILE = "thumbs_down_log.xlsx"

if not os.path.exists(THUMBS_UP_FILE) or not os.path.exists(THUMBS_DOWN_FILE):
    print("Error: Log files not found. Please run the bot and collect feedback first.")
    exit()

up_df = pd.read_excel(THUMBS_UP_FILE)
up_df = up_df.rename(columns={"successful_answer": "answer"})
up_df['label'] = 1

down_df = pd.read_excel(THUMBS_DOWN_FILE)
down_df = down_df.rename(columns={"failed_answer": "answer"})
down_df['label'] = 0

full_df = pd.concat([up_df[['question', 'answer', 'label']], down_df[['question', 'answer', 'label']]])
full_df.dropna(inplace=True)
full_df['text_input'] = full_df['question'] + " [SEP] " + full_df['answer']

if len(full_df) < 20:
    print(f"Warning: Low data ({len(full_df)} examples). Model may not be very accurate.")

# --- 2. Create Text Embeddings (The "Smart" Upgrade) ---
print("\nLoading sentence transformer model to create embeddings...")
# This model is small, fast, and effective for this task.
embedding_model = SentenceTransformer('all-MiniLM-L6-v2') 

print("Converting text to numerical embeddings... (This may take a moment)")
X_embeddings = embedding_model.encode(full_df['text_input'].tolist(), show_progress_bar=True)
y_labels = full_df['label'].values

print(f"Created {X_embeddings.shape[0]} embeddings of size {X_embeddings.shape[1]}.")

# --- 3. Train the Classifier on the Embeddings ---
X_train, X_test, y_train, y_test = train_test_split(
    X_embeddings, y_labels, test_size=0.25, random_state=42, stratify=y_labels
)

# We no longer need a Pipeline. We just train a classifier on the embeddings.
classifier = LogisticRegression(class_weight='balanced', max_iter=1000)

print("\nTraining Critic model on embeddings...")
classifier.fit(X_train, y_train)

# --- 4. Evaluate and Save ---
accuracy = classifier.score(X_test, y_test)
print(f"✅ Critic model accuracy on test data: {accuracy:.2%}")

# We save the embedding model AND the classifier together for convenience
final_model_package = {
    'embedding_model': embedding_model,
    'classifier': classifier
}

joblib.dump(final_model_package, 'critic_model.pkl')
print("✅ Critic model package saved successfully to critic_model.pkl")