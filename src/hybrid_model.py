# src/hybrid_model.py

import pandas as pd
from collections import Counter


class HybridRecommender:
    def __init__(self, cf_model, content_model, embedding_model, weights=(0.5, 0.2, 0.3)):
        """
        Initialize HybridRecommender with component models and their respective weights.
        """
        self.cf_model = cf_model
        self.content_model = content_model
        self.embedding_model = embedding_model
        self.weights = weights  # (cf_weight, content_weight, embedding_weight)

    def recommend_tracks(self, playlist_id, playlist_tracks, top_n=10):
        from collections import defaultdict
    
        score_counter = defaultdict(float)

    # CF recommendations
        if self.cf_model:
          cf_recs = self.cf_model.recommend_tracks(playlist_id, N=top_n * 2)
          for rank, track_uri in enumerate(cf_recs):
            score_counter[track_uri] += 0.4 * (top_n - rank)  # example weighting

    # Content-based recommendations
        if self.content_model:
          content_recs = self.content_model.recommend_tracks(playlist_tracks, top_n=top_n * 2)
          for rank, track_uri in enumerate(content_recs):
            score_counter[track_uri] += 0.3 * (top_n - rank)

    # Embedding-based recommendations
        if self.embedding_model:
          embedding_recs = self.embedding_model.recommend_for_playlist(playlist_tracks, top_n=top_n * 2)
          for rank, track_uri in enumerate(embedding_recs):
            score_counter[track_uri] += 0.3 * (top_n - rank)

    # Remove tracks already in playlist
        for track in playlist_tracks:
          if track in score_counter:
            del score_counter[track]

    # Sort by score descending
        sorted_scores = sorted(score_counter.items(), key=lambda x: x[1], reverse=True)
    
    # Return top_n list of (track_uri, score)
        return sorted_scores[:top_n]

