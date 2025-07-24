# src/cf_model.py

import pandas as pd
from scipy.sparse import coo_matrix
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import bm25_weight
import numpy as np

class CFRecommender:
    def __init__(self, factors=128, regularization=0.01, iterations=20):
        self.model = AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            iterations=iterations,
            use_gpu=False
        )
        self.playlist_map = {}
        self.track_map = {}
        self.rev_playlist_map = {}
        self.rev_track_map = {}
        self.user_item_matrix = None

    def prepare_matrix(self, df: pd.DataFrame):
        print("🔄 Preparing user-item matrix...")
        playlists = df['playlist_id'].unique()
        tracks = df['track_uri'].unique()

        self.playlist_map = {pid: idx for idx, pid in enumerate(playlists)}
        self.track_map = {tid: idx for idx, tid in enumerate(tracks)}
        self.rev_playlist_map = {idx: pid for pid, idx in self.playlist_map.items()}
        self.rev_track_map = {idx: tid for tid, idx in self.track_map.items()}

        rows = df['playlist_id'].map(self.playlist_map)
        cols = df['track_uri'].map(self.track_map)
        data = [1] * len(df)

        matrix = coo_matrix((data, (rows, cols)), shape=(len(playlists), len(tracks)))
        matrix = bm25_weight(matrix).tocsr()

        self.user_item_matrix = matrix
        return matrix

    def train(self):
        print("Training collaborative filtering model...")
        self.model.fit(self.user_item_matrix)

    import numpy as np

    def recommend_tracks(self, playlist_id: int, N=10):
      if playlist_id not in self.playlist_map:
        print("❌ Playlist not found in training set.")
        return []

      playlist_idx = self.playlist_map[playlist_id]
      user_items_row = self.user_item_matrix[playlist_idx]

      recommendations = self.model.recommend(
        userid=playlist_idx,
        user_items=user_items_row,
        N=N
    )

    # ✅ Handle output format + ensure idx is an integer
      track_ids = []
      for r in recommendations:
        if isinstance(r, tuple):
            idx = r[0]
        else:
            idx = r
        if isinstance(idx, (list, tuple, np.ndarray)):
            idx = int(idx[0])
        else:
            idx = int(idx)
        track_ids.append(idx)

      return [self.rev_track_map[idx] for idx in track_ids]

