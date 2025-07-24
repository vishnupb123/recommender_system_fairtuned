import numpy as np
import pandas as pd

class Reranker:
    def __init__(self,
                 track_popularity: pd.Series,
                 popularity_weight=0.7,
                 diversity_weight=0.3,
                 underrep_weight=0.2,
                 user_fairness_weight=0.0,
                 underrepresented_artists: set = None,
                 user_profile: dict = None):
        """
        Enhanced reranker with fairness support.

        track_popularity: pd.Series with track_uri as index
        popularity_weight: penalize popular tracks
        diversity_weight: penalize repeated artists
        underrep_weight: boost underrepresented artists
        user_fairness_weight: apply user-level preference balancing
        underrepresented_artists: set of artist_names to promote
        user_profile: dict of user's fairness profile (e.g. {'low_popularity_ratio': 0.4})
        """
        self.track_popularity = track_popularity
        self.popularity_weight = popularity_weight
        self.diversity_weight = diversity_weight
        self.underrep_weight = underrep_weight
        self.user_fairness_weight = user_fairness_weight
        self.underrepresented_artists = underrepresented_artists or set()
        self.user_profile = user_profile or {}
        self.track_artist_map = {}

    def set_artist_map(self, track_artist_map):
        self.track_artist_map = track_artist_map

    def rerank(self, recommended_tracks: list, original_scores: list):
        pop_scores = self.track_popularity.reindex(recommended_tracks).fillna(0)
        pop_norm = (pop_scores - pop_scores.min()) / (pop_scores.max() - pop_scores.min() + 1e-9)

        # --- Artist Diversity Penalty ---
        artist_counts = {}
        diversity_penalty = []
        underrep_boost = []
        for track in recommended_tracks:
            artist = self.track_artist_map.get(track, "unknown")
            
            # diversity
            count = artist_counts.get(artist, 0)
            diversity_penalty.append(count)
            artist_counts[artist] = count + 1

            # underrepresented boost
            boost = 1 if artist in self.underrepresented_artists else 0
            underrep_boost.append(boost)

        diversity_penalty = np.array(diversity_penalty)
        diversity_norm = (diversity_penalty - diversity_penalty.min()) / (diversity_penalty.max() - diversity_penalty.min() + 1e-9)

        underrep_boost = np.array(underrep_boost)

        # --- Combine Everything ---
        combined_score = (np.array(original_scores)
                          - self.popularity_weight * pop_norm
                          - self.diversity_weight * diversity_norm
                          + self.underrep_weight * underrep_boost)

        # --- OPTIONAL: User fairness adjustment ---
        # For now just a placeholder logic — can be expanded
        if self.user_fairness_weight > 0 and 'low_popularity_ratio' in self.user_profile:
            user_low_pop_ratio = self.user_profile['low_popularity_ratio']
            current_ratio = np.mean(pop_norm < 0.3)
            deviation = abs(current_ratio - user_low_pop_ratio)
            combined_score -= self.user_fairness_weight * deviation  # apply soft penalty if deviated

        reranked_indices = np.argsort(-combined_score)
        reranked_tracks = [recommended_tracks[i] for i in reranked_indices]
        return reranked_tracks
