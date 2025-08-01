import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

class EmbeddingRecommender:
    def __init__(self, processed_dir, vector_size=64):
        self.processed_dir = processed_dir
        self.vector_size = vector_size
        self._load_artifacts()
        self._build_embeddings_matrix()

    def _load_artifacts(self):
        print("🔄 Loading saved artifacts...")
        self.track_df = pd.read_pickle(os.path.join(self.processed_dir, 'tracks.pkl'))
        with open(os.path.join(self.processed_dir, 'mappings.pkl'), 'rb') as f:
            mappings = pickle.load(f)
        self.track2id = mappings['track2id']
        self.id2track = mappings['id2track']
        with open(os.path.join(self.processed_dir, 'word2vec.pkl'), 'rb') as f:
            self.w2v_model = pickle.load(f)

    def _build_embeddings_matrix(self):
        print("⚙️ Building embedding matrix...")
        n_tracks = len(self.track2id)
        self.embeddings = np.zeros((n_tracks, self.vector_size), dtype=np.float32)

        for uri, idx in tqdm(self.track2id.items(), desc="Embedding tracks"):
            row = self.track_df[self.track_df['track_uri'] == uri]
            if row.empty:
                continue
            text = f"{row.iloc[0]['track_name']} {row.iloc[0]['artist_name']}"
            tokens = text.lower().split()
            vectors = [self.w2v_model.wv[w] for w in tokens if w in self.w2v_model.wv]
            if vectors:
                self.embeddings[idx] = np.mean(vectors, axis=0)

        # Normalize for cosine similarity
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings = np.divide(self.embeddings, norms, out=np.zeros_like(self.embeddings), where=norms != 0)

    def recommend_similar_tracks(self, track_uri, top_n=10):
        if track_uri not in self.track2id:
            raise ValueError(f"❌ Track URI '{track_uri}' not found.")
        
        idx = self.track2id[track_uri]
        query_vec = self.embeddings[idx].reshape(1, -1)

        if not np.any(query_vec):
            print(f"⚠️ No embedding found for {track_uri}")
            return []

        scores = self.embeddings @ query_vec.T
        scores[idx] = -1  # avoid self-match
        top_indices = np.argpartition(scores[:, 0], -top_n)[-top_n:]
        top_indices = top_indices[np.argsort(scores[top_indices, 0])[::-1]]

        return [(self.id2track[i], float(scores[i])) for i in top_indices]

# 🧪 Example usage
if __name__ == "__main__":
    BASE_DIR = "/Users/vishnupb/Desktop/spotify_recommender/v2/processed"
    recommender = EmbeddingRecommender(processed_dir=BASE_DIR)

    test_uri = "spotify:track:0ESJlaM8CE1jRWaNtwSNj8"
    print(f"\n🎯 Top recommendations for: {test_uri}\n")
    recommendations = recommender.recommend_similar_tracks(test_uri, top_n=10)
    for uri, score in recommendations:
        print(f"{uri:<45}  Similarity: {score:.4f}")
