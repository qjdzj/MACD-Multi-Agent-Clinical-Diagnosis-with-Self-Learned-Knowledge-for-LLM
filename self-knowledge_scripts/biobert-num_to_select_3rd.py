import os 
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch as th
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load BioBERT model and tokenizer
def load_biobert():
    tokenizer = AutoTokenizer.from_pretrained(
        ''
        )
    model = AutoModel.from_pretrained('')
    return tokenizer, model

# Convert text to vectors
def encode_texts(texts, tokenizer, model, max_length=128):
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    with th.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()

def compute_similarity_matrix(embeddings):
    return cosine_similarity(embeddings)

def select_low_similarity_subset(texts, embeddings, num_to_select):
    similarity_matrix = compute_similarity_matrix(embeddings)
    selected_indices = []
    remaining_indices = set(range(len(texts)))

    for _ in range(num_to_select):
        min_sim = float("inf")
        best_idx = None
        for idx in remaining_indices:
            if not selected_indices:
                best_idx = idx
                break
            max_sim_to_selected = max(similarity_matrix[idx][i] for i in selected_indices)
            if max_sim_to_selected < min_sim:
                min_sim = max_sim_to_selected
                best_idx = idx

        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)

    selected_texts = [texts[i] for i in selected_indices]
    return selected_texts

def main():
    texts = [

    ]

    num_to_select = 7  

    # Load model and vectorize
    tokenizer, model = load_biobert()
    embeddings = encode_texts(texts, tokenizer, model)

    # Filter texts with lowest similarity
    selected_texts = select_low_similarity_subset(texts, embeddings, num_to_select)
    print("Selected diagnostic criteria:")
    for text in selected_texts:
        print(text)

if __name__ == "__main__":
    main()
