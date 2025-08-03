import os
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp

# 📂 Paths
ROOT = "/Users/vishnupb/Desktop/spotify_recommender"
PROCESSED_DIR = os.path.join(ROOT, "v2", "processed")
MODEL_DIR = os.path.join(ROOT, "v2", "models")

# 📥 Load preprocessed data
with open(os.path.join(PROCESSED_DIR, "tfidf_matrix.pkl"), "rb") as f:
    tfidf_matrix = pickle.load(f)

with open(os.path.join(PROCESSED_DIR, "tracks.pkl"), "rb") as f:
    tracks_df = pickle.load(f)

with open(os.path.join(PROCESSED_DIR, "mappings.pkl"), "rb") as f:
    mappings = pickle.load(f)

track2id = mappings["track2id"]
id2track = mappings["id2track"]

print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

# 🎯 Recommend similar tracks (no full matrix)
def recommend_similar_tracks(input_track_uri, top_n=10, batch_size=50000):
    if input_track_uri not in track2id:
        raise ValueError(f"Track URI {input_track_uri} not found in track2id mapping.")

    track_idx = track2id[input_track_uri]
    input_vec = tfidf_matrix[track_idx]

    num_tracks = tfidf_matrix.shape[0]
    top_scores = []
    top_indices = []

    for start in range(0, num_tracks, batch_size):
        end = min(start + batch_size, num_tracks)
        batch = tfidf_matrix[start:end]
        sim_scores = cosine_similarity(input_vec, batch).flatten()

        for i, score in enumerate(sim_scores):
            global_idx = start + i
            if global_idx == track_idx or score >= 0.9999:
                continue
            top_scores.append((global_idx, score))

    # Sort and select top_n
    top_scores.sort(key=lambda x: x[1], reverse=True)
    recommendations = []
    count = 0

    for idx, score in top_scores:
        if idx in id2track:
            recommendations.append((id2track[idx], score))
            count += 1
        if count == top_n:
            break

    return recommendations

# 🧪 Batch-wise precomputation
def batch_precompute_recommendations(batch_size=100, top_n=10, max_tracks=1000):
    print(f"\n💾 Precomputing recommendations in batches (batch_size = {batch_size})...")
    all_uris = list(track2id.keys())[:max_tracks]
    all_recommendations = {}

    for i, uri in enumerate(all_uris):
        try:
            recs = recommend_similar_tracks(uri, top_n=top_n)
            all_recommendations[uri] = recs
        except Exception as e:
            print(f"⚠️ Skipping {uri}: {e}")
            continue

        if (i + 1) % 10 == 0:
            print(f"✅ Processed {i+1}/{len(all_uris)}")

    return all_recommendations

# 🚀 Main
if __name__ == "__main__":
    sample_uri = list(track2id.keys())[0]
    print(f"\n🎵 Recommendations for track: {sample_uri}")

    recommendations = recommend_similar_tracks(sample_uri)
    for uri, score in recommendations:
        print(f"{uri}: {score:.4f}")

    # 💾 Save precomputed recommendations
    precomputed = batch_precompute_recommendations(batch_size=100, top_n=10, max_tracks=1000)
    with open(os.path.join(MODEL_DIR, "content_top_recs.pkl"), "wb") as f:
        pickle.dump(precomputed, f)
    print("\n✅ Saved precomputed recommendations.")
