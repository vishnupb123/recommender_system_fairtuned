import os
import pickle
import numpy as np
from scipy.sparse import load_npz
from implicit.als import AlternatingLeastSquares

# Paths (adjust if needed)
BASE_DIR = '/Users/vishnupb/Desktop/spotify_recommender'
PROCESSED_DIR = os.path.join(BASE_DIR, 'v2', 'processed')
MODEL_DIR = os.path.join(BASE_DIR, 'v2', 'models')

# Load artifacts
print("Loading interaction matrix...")
interaction_matrix = load_npz(os.path.join(PROCESSED_DIR, 'interaction_matrix.npz'))
print(f"Interaction matrix shape: {interaction_matrix.shape}")

print("Loading mappings...")
with open(os.path.join(PROCESSED_DIR, 'mappings.pkl'), 'rb') as f:
    mappings = pickle.load(f)
track2id = mappings['track2id']
id2track = mappings['id2track']

# ALS model training
print("Training ALS model...")
als_model = AlternatingLeastSquares(
    factors=64,
    regularization=0.1,
    iterations=15,
    use_gpu=False
)

# Implicit expects item-user matrix, so transpose the interaction matrix
print("Fitting ALS model...")
als_model.fit(interaction_matrix.T)

# Save the model
os.makedirs(MODEL_DIR, exist_ok=True)
model_path = os.path.join(MODEL_DIR, 'cf_model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(als_model, f)
print(f"CF model saved to {model_path}")


def recommend_tracks(model, user_id, track2id, id2track, user_items, N=10):
    recommended = model.recommend(userid=user_id, user_items=user_items[user_id], N=N, filter_already_liked_items=True)
    # recommended is a tuple: (array_of_item_ids, array_of_scores)

    item_ids = recommended[0]
    scores = recommended[1]

    results = []
    for track_id, score in zip(item_ids, scores):
        results.append((id2track[track_id], score))
    return results



if __name__ == '__main__':
    # Example usage: recommend for user 0
    print("Generating recommendations for playlist index 0...")
    recs = recommend_tracks(als_model, user_id=0, track2id=track2id, id2track=id2track, user_items=interaction_matrix, N=10)
    for track_uri, score in recs:
        print(f"{track_uri}: {score:.4f}")
