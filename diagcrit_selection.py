import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from apricot import FacilityLocationSelection

# Text data
texts = [
    "This is a sample text about machine learning.",
    "Another text discussing deep learning concepts.",
    "This is related to artificial intelligence.",
    "A sample discussing neural networks in AI."
    "Example on transformers and attention mechanisms."
]

# 1. Text vectorization
vectorizer = TfidfVectorizer()
text_features = vectorizer.fit_transform(texts).toarray()
print(f"Text Features:\n{text_features}")

# 2. Generate concept features through clustering
num_concepts = 2
kmeans = KMeans(n_clusters=num_concepts)
concept_features = kmeans.fit_transform(text_features)
print(f"Concept Features:\n{concept_features}")

# 3. Calculate similarity between concepts
concept_sim_matrix = cosine_similarity(concept_features)
print(f"Concept Similarity Matrix:\n{concept_sim_matrix}")

# 4. Submodular selection
def submodular_select_text(features, num_concepts):
    selector = FacilityLocationSelection(num_concepts, metric='cosine')
    selected_idx = selector.fit_transform(features)
    return selected_idx

# 5. Select optimal concepts
num_selected_concepts = 2
selected_idx = submodular_select_text(text_features, num_selected_concepts)
print(f"selected_idx: {selected_idx}")
selected_idx = np.array(selected_idx).ravel().tolist()
print(f"selected_idx: {selected_idx}")
selected_idx = [int(i) for i in selected_idx]
print(f"selected_idx: {selected_idx}")

# Output selection results
selected_texts = [texts[i] for i in selected_idx]
print("Selected Texts:", selected_texts)
