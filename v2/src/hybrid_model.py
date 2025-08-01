import os
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity

# === CONFIG ===
BASE_DIR = '/Users/vishnupb/Desktop/spotify_recommender'
PROCESSED_DIR = os.path.join(BASE_DIR, 'v2', 'processed')

# === LOAD ARTIFACTS ===
print("Loading artifacts...")

playlist_df = pd.read_pickle(os.path.join(PROCESSED_DIR, 'playlists.pkl'))
track_df = pd.read_pickle(os.path.join(PROCESSED_DIR, 'tracks.pkl'))

with open(os.path.join(PROCESSED_DIR, 'mappings.pkl'), 'rb') as f:
    mappings = pickle.load(f)
track2id = mappings['track2id']
id2track = mappings['id2track']

with open(os.path.join(PROCESSED_DIR, 'word2vec.pkl'), 'rb') as f:
    w2v_model = pickle.load(f)

word2vec_matrix = np.load(os.path.join(PROCESSED_DIR, 'track_embeddings.npy'))

interaction_matrix = load_npz(os.path.join(PROCESSED_DIR, 'interaction_matrix.npz'))

print(f"Playlists: {len(playlist_df)}, Tracks: {len(track_df)}")
print(f"Interaction matrix shape: {interaction_matrix.shape}")
print(f"Word2Vec embeddings shape: {word2vec_matrix.shape}")

# === HELPER: Build playlist index mapping ===
pid_to_idx = {pid: idx for idx, pid in enumerate(playlist_df['pid'])}
idx_to_pid = {v: k for k, v in pid_to_idx.items()}

# === 1) Collaborative Filtering Recommendation Logic ===
# Here, since you have interaction_matrix (playlists x tracks),
# we'll compute for each playlist the most interacted tracks and then recommend popular ones not in playlist.
# A simple CF heuristic based on popularity weighted by playlist similarity.

# Compute track popularity (number of playlists containing track)
track_popularity = np.array(interaction_matrix.sum(axis=0)).flatten()  # shape = (num_tracks,)

def get_cf_recommendations_for_playlist(pid, top_k=50):
    """Recommend tracks for playlist pid using CF heuristic:
       - Find playlists similar to pid (by Jaccard or cosine similarity)
       - Aggregate track popularity from similar playlists weighted by similarity
       - Exclude tracks already in the playlist
    """
    p_idx = pid_to_idx[pid]
    
    # Binary vector of tracks in playlist
    p_vec = interaction_matrix.getrow(p_idx).toarray().flatten()
    
    # Compute cosine similarity between this playlist and all playlists
    # Cosine similarity between binary vectors
    similarities = interaction_matrix.dot(p_vec)
    norm_p = np.linalg.norm(p_vec)
    norms = np.sqrt(interaction_matrix.multiply(interaction_matrix).sum(axis=1)).A1  # playlist norms
    denom = norms * norm_p + 1e-8
    cosine_sim = similarities / denom  # shape = (num_playlists,)
    
    # Aggregate track scores from other playlists weighted by similarity
    weighted_sum = cosine_sim @ interaction_matrix  # shape = (num_tracks,)
    
    # Remove tracks already in playlist
    weighted_sum[p_vec > 0] = 0
    
    # Get top_k tracks by score
    top_indices = np.argpartition(-weighted_sum, top_k)[:top_k]
    top_scores = weighted_sum[top_indices]
    
    # Sort top_k tracks by score descending
    sorted_idx = top_indices[np.argsort(-top_scores)]
    
    recs = [(id2track[i], weighted_sum[i]) for i in sorted_idx if weighted_sum[i] > 0]
    
    return recs[:top_k]

# Build CF recommendations dictionary for all playlists
print("Building CF recommendations dict...")
cf_recs_dict = {}
for pid in playlist_df['pid']:
    cf_recs_dict[pid] = get_cf_recommendations_for_playlist(pid, top_k=50)
print("CF recommendations built.")

# === 2) Content-Based Recommendations Using Word2Vec ===
# For each track, find top similar tracks by cosine similarity in embedding space

print("Computing content similarity matrix (Word2Vec)...")
content_sim_matrix = cosine_similarity(word2vec_matrix)  # tracks x tracks

def get_content_recs_for_track(track_uri, top_k=20):
    if track_uri not in track2id:
        return []
    t_idx = track2id[track_uri]
    sim_scores = content_sim_matrix[t_idx]
    # exclude self similarity
    sim_scores[t_idx] = -1
    top_indices = np.argpartition(-sim_scores, top_k)[:top_k]
    top_scores = sim_scores[top_indices]
    sorted_idx = top_indices[np.argsort(-top_scores)]
    recs = [(id2track[i], sim_scores[i]) for i in sorted_idx if sim_scores[i] > 0]
    return recs[:top_k]

# Build content recommendations dictionary for all tracks in track_df
print("Building content recommendations dict...")
content_recs = {}
unique_track_uris = track_df['track_uri'].unique()
for track_uri in unique_track_uris:
    content_recs[track_uri] = get_content_recs_for_track(track_uri, top_k=20)
print("Content recommendations built.")

# === 3) Hybrid Recommendation Function ===

def hybrid_recommend_for_playlist(pid, top_k=50, alpha=0.6):
    """
    Hybrid recommendation for a playlist.
    - alpha: weight for CF scores
    - (1 - alpha): weight for content similarity scores
    For each CF recommended track, look at content-based neighbors and combine scores.
    """
    # Get CF recommendations for playlist (track_uri, score)
    cf_recs = cf_recs_dict.get(pid, [])
    if not cf_recs:
        return []
    
    # Tracks already in playlist (to exclude)
    p_idx = pid_to_idx[pid]
    p_tracks = set(track_df[track_df['pid'] == pid]['track_uri'].unique())
    
    # Dict to accumulate final scores {track_uri: score}
    final_scores = {}
    
    for track_uri, cf_score in cf_recs:
        # Add CF score directly if not in playlist
        if track_uri not in p_tracks:
            final_scores[track_uri] = final_scores.get(track_uri, 0) + alpha * cf_score
        
        # Get content similar tracks to this track
        content_neighbors = content_recs.get(track_uri, [])
        for c_track_uri, c_score in content_neighbors:
            if c_track_uri not in p_tracks:
                # Add weighted content score scaled by CF score to favor CF rec context
                final_scores[c_track_uri] = final_scores.get(c_track_uri, 0) + (1 - alpha) * cf_score * c_score
    
    # Sort final scores descending
    ranked_recs = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    
    return ranked_recs[:top_k]

# === Example usage ===
sample_pid = playlist_df['pid'].iloc[0]
hybrid_recs = hybrid_recommend_for_playlist(sample_pid, top_k=20, alpha=0.7)

print(f"\nHybrid recommendations for playlist {sample_pid}:")
for track_uri, score in hybrid_recs:
    print(f"{track_uri} - score: {score:.4f}")
