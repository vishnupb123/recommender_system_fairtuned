import os
import json
import glob
import pickle
import gc
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from gensim.models import Word2Vec
from scipy.sparse import csr_matrix, save_npz, vstack
from tqdm import tqdm
import logging
import hashlib
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OptimizedSpotifyPreprocessor:
    """
    Memory-efficient preprocessing pipeline for hybrid music recommendation system
    """
    
    def __init__(self, base_dir: str, output_dir: str = None, max_playlists: int = 80000):
        self.base_dir = base_dir
        self.data_dir = os.path.join(base_dir, 'data')
        self.output_dir = output_dir or os.path.join(base_dir, 'v3', 'processed')
        self.max_playlists = max_playlists
        
        # Simplified configuration for better performance
        self.config = {
            'tfidf_max_features': 3000,  # Reduced for faster processing
            'word2vec_vector_size': 64,   # Reduced from 128 to 64
            'word2vec_window': 3,         # Reduced window size
            'word2vec_min_count': 3,      # Increased min_count for efficiency
            'min_playlist_length': 5,
            'min_track_frequency': 3,     # Increased for better filtering
            'popularity_bins': [0, 5, 20, 50, np.inf],  # Simplified bins
            'duration_bins': [0, 120, 240, 360, np.inf],  # Simplified bins
            'batch_size': 5000            # Smaller batches for memory efficiency
        }
        
        # Data containers
        self.track_metadata = {}
        self.artist_metadata = {}
        self.album_metadata = {}
        
        logger.info(f"Initialized preprocessor with max_playlists={max_playlists}")
    
    def load_playlists_efficiently(self) -> List[Dict]:
        """Load playlists with memory-efficient streaming"""
        logger.info("🔄 Loading playlists with streaming approach...")
        
        playlists = []
        track_counter = Counter()
        files = glob.glob(os.path.join(self.data_dir, "*.json"))
        
        logger.info(f"Found {len(files)} JSON files")
        
        for file_path in tqdm(files, desc="Processing files"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    file_playlists = data.get('playlists', [])
                    
                    # Filter and collect playlists
                    for playlist in file_playlists:
                        if len(playlist.get('tracks', [])) >= self.config['min_playlist_length']:
                            # Count track frequencies for filtering
                            for track in playlist['tracks']:
                                track_counter[track['track_uri']] += 1
                            
                            playlists.append(playlist)
                            
                            if len(playlists) >= self.max_playlists:
                                logger.info(f"Reached max playlist limit: {self.max_playlists}")
                                return playlists, track_counter
                                
            except Exception as e:
                logger.warning(f"Error processing file {file_path}: {e}")
                continue
        
        logger.info(f"Loaded {len(playlists)} playlists with {len(track_counter)} unique tracks")
        return playlists, track_counter
    
    def enhanced_stratified_sampling(self, playlists: List[Dict], track_counter: Counter) -> List[Dict]:
        """Enhanced stratified sampling with multiple criteria"""
        logger.info("🎯 Performing enhanced stratified sampling...")
        
        # Categorize playlists by multiple dimensions
        playlist_categories = defaultdict(list)
        
        for playlist in tqdm(playlists, desc="Categorizing playlists"):
            # Calculate playlist characteristics
            track_counts = [track_counter[track['track_uri']] for track in playlist['tracks']]
            avg_popularity = np.mean(track_counts) if track_counts else 0
            playlist_length = len(playlist['tracks'])
            num_followers = playlist.get('num_followers', 0)
            
            # Multi-dimensional categorization
            pop_bin = self._get_popularity_bin(avg_popularity)
            length_bin = self._get_length_bin(playlist_length)
            follower_bin = self._get_follower_bin(num_followers)
            
            # Create composite category key
            category_key = f"{pop_bin}_{length_bin}_{follower_bin}"
            playlist_categories[category_key].append(playlist)
        
        # Sample from each category
        sampled_playlists = []
        total_categories = len(playlist_categories)
        
        if total_categories == 0:
            logger.warning("No playlist categories found!")
            return playlists[:self.max_playlists]
        
        per_category_limit = max(1, self.max_playlists // total_categories)
        
        for category, category_playlists in playlist_categories.items():
            sample_size = min(per_category_limit, len(category_playlists))
            # Shuffle for randomness
            np.random.shuffle(category_playlists)
            sampled_playlists.extend(category_playlists[:sample_size])
            
            logger.info(f"Category {category}: sampled {sample_size}/{len(category_playlists)} playlists")
        
        # Final shuffle and trim
        np.random.shuffle(sampled_playlists)
        final_sample = sampled_playlists[:self.max_playlists]
        
        logger.info(f"Final stratified sample: {len(final_sample)} playlists")
        return final_sample
    
    def _get_popularity_bin(self, avg_popularity: float) -> str:
        """Categorize by popularity"""
        bins = self.config['popularity_bins']
        for i in range(len(bins) - 1):
            if bins[i] <= avg_popularity < bins[i + 1]:
                return f"pop_{i}"
        return f"pop_{len(bins)-2}"
    
    def _get_length_bin(self, length: int) -> str:
        """Categorize by playlist length"""
        if length < 10:
            return "short"
        elif length < 30:
            return "medium"
        elif length < 100:
            return "long"
        else:
            return "very_long"
    
    def _get_follower_bin(self, followers: int) -> str:
        """Categorize by follower count"""
        if followers == 0:
            return "no_followers"
        elif followers < 10:
            return "few_followers"
        elif followers < 100:
            return "moderate_followers"
        else:
            return "popular"
    
    def extract_and_normalize_data(self, playlists: List[Dict], track_counter: Counter) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Extract and normalize playlist and track data with enhanced features"""
        logger.info("🔧 Extracting and normalizing data...")
        
        playlist_records = []
        track_records = []
        
        # Filter tracks by minimum frequency
        frequent_tracks = {uri for uri, count in track_counter.items() 
                          if count >= self.config['min_track_frequency']}
        
        logger.info(f"Filtering to {len(frequent_tracks)} frequent tracks")
        
        for playlist in tqdm(playlists, desc="Processing playlists"):
            pid = playlist['pid']
            
            # Filter tracks to only frequent ones
            filtered_tracks = [track for track in playlist['tracks'] 
                             if track['track_uri'] in frequent_tracks]
            
            if len(filtered_tracks) < self.config['min_playlist_length']:
                continue
            
            # Enhanced playlist features
            playlist_record = {
                'pid': pid,
                'name': self._clean_text(playlist['name']),
                'num_tracks': len(filtered_tracks),
                'num_followers': playlist.get('num_followers', 0),
                'modified_at': playlist.get('modified_at', 0),
                'collaborative': playlist.get('collaborative', False),
                # Additional features
                'avg_duration': np.mean([track['duration_ms'] for track in filtered_tracks]),
                'std_duration': np.std([track['duration_ms'] for track in filtered_tracks]),
                'num_unique_artists': len(set(track['artist_uri'] for track in filtered_tracks)),
                'num_unique_albums': len(set(track['album_uri'] for track in filtered_tracks)),
                'popularity_score': np.mean([track_counter[track['track_uri']] for track in filtered_tracks])
            }
            playlist_records.append(playlist_record)
            
            # Process tracks with enhanced features
            for pos, track in enumerate(filtered_tracks):
                track_record = {
                    'pid': pid,
                    'track_uri': track['track_uri'],
                    'track_name': self._clean_text(track['track_name']),
                    'artist_uri': track['artist_uri'],
                    'artist_name': self._clean_text(track['artist_name']),
                    'album_uri': track['album_uri'],
                    'album_name': self._clean_text(track['album_name']),
                    'duration_ms': track['duration_ms'],
                    'position': pos,
                    'frequency': track_counter[track['track_uri']],
                    # Additional features
                    'duration_category': self._categorize_duration(track['duration_ms']),
                    'popularity_score': np.log1p(track_counter[track['track_uri']]),
                    'track_hash': hashlib.md5(track['track_name'].encode()).hexdigest()[:8]
                }
                track_records.append(track_record)
                
                # Store metadata for later use
                self._update_metadata(track)
        
        # Create DataFrames
        playlist_df = pd.DataFrame(playlist_records)
        track_df = pd.DataFrame(track_records)
        
        # Additional feature engineering
        playlist_df = self._engineer_playlist_features(playlist_df)
        track_df = self._engineer_track_features(track_df)
        
        logger.info(f"Created DataFrames: {len(playlist_df)} playlists, {len(track_df)} track entries")
        return playlist_df, track_df
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not isinstance(text, str):
            return ""
        return text.strip().lower().replace('\n', ' ').replace('\t', ' ')
    
    def _categorize_duration(self, duration_ms: int) -> str:
        """Categorize track duration"""
        duration_sec = duration_ms / 1000
        bins = self.config['duration_bins']
        for i in range(len(bins) - 1):
            if bins[i] <= duration_sec < bins[i + 1]:
                return f"duration_{i}"
        return f"duration_{len(bins)-2}"
    
    def _update_metadata(self, track: Dict):
        """Update metadata dictionaries"""
        # Track metadata
        if track['track_uri'] not in self.track_metadata:
            self.track_metadata[track['track_uri']] = {
                'name': track['track_name'],
                'duration_ms': track['duration_ms']
            }
        
        # Artist metadata
        if track['artist_uri'] not in self.artist_metadata:
            self.artist_metadata[track['artist_uri']] = {
                'name': track['artist_name']
            }
        
        # Album metadata
        if track['album_uri'] not in self.album_metadata:
            self.album_metadata[track['album_uri']] = {
                'name': track['album_name']
            }
    
    def _engineer_playlist_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer simplified playlist features"""
        logger.info("Engineering playlist features...")
        
        # Basic numerical features only
        df['diversity_score'] = df['num_unique_artists'] / df['num_tracks']
        
        # Simplified follower categories
        df['follower_category'] = pd.cut(df['num_followers'], 
                                       bins=[0, 1, 50, np.inf],
                                       labels=['none', 'some', 'many'])
        
        return df
    
    def _engineer_track_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer additional track features"""
        logger.info("Engineering track features...")
        
        # Position-based features
        df['position_normalized'] = df.groupby('pid')['position'].transform(
            lambda x: x / (x.max() + 1) if x.max() > 0 else 0
        )
        
        # Frequency-based features
        df['frequency_percentile'] = df['frequency'].rank(pct=True)
        df['is_popular'] = df['frequency_percentile'] > 0.8
        df['is_niche'] = df['frequency_percentile'] < 0.2
        
        return df
    
    def build_enhanced_mappings(self, track_df: pd.DataFrame) -> Dict:
        """Build comprehensive mapping dictionaries"""
        logger.info("🗺️ Building enhanced mappings...")
        
        # Core mappings
        unique_tracks = track_df['track_uri'].unique()
        unique_artists = track_df['artist_uri'].unique()
        unique_albums = track_df['album_uri'].unique()
        unique_playlists = track_df['pid'].unique()
        
        mappings = {
            # Track mappings
            'track2id': {uri: idx for idx, uri in enumerate(unique_tracks)},
            'id2track': {idx: uri for idx, uri in enumerate(unique_tracks)},
            
            # Artist mappings
            'artist2id': {uri: idx for idx, uri in enumerate(unique_artists)},
            'id2artist': {idx: uri for idx, uri in enumerate(unique_artists)},
            
            # Album mappings
            'album2id': {uri: idx for idx, uri in enumerate(unique_albums)},
            'id2album': {idx: uri for idx, uri in enumerate(unique_albums)},
            
            # Playlist mappings
            'playlist2id': {pid: idx for idx, pid in enumerate(unique_playlists)},
            'id2playlist': {idx: pid for idx, pid in enumerate(unique_playlists)},
            
            # Metadata
            'track_metadata': self.track_metadata,
            'artist_metadata': self.artist_metadata,
            'album_metadata': self.album_metadata,
            
            # Statistics
            'stats': {
                'num_tracks': len(unique_tracks),
                'num_artists': len(unique_artists),
                'num_albums': len(unique_albums),
                'num_playlists': len(unique_playlists),
                'total_interactions': len(track_df)
            }
        }
        
        logger.info(f"Built mappings: {mappings['stats']}")
        return mappings
    
    def build_optimized_matrices(self, track_df: pd.DataFrame, mappings: Dict) -> Tuple[csr_matrix, csr_matrix]:
        """Build interaction and feature matrices efficiently"""
        logger.info("🔨 Building optimized sparse matrices...")
        
        # Build interaction matrix
        playlist_ids = [mappings['playlist2id'][pid] for pid in track_df['pid']]
        track_ids = [mappings['track2id'][uri] for uri in track_df['track_uri']]
        
        # Use frequency as interaction strength (implicit feedback)
        weights = track_df['frequency'].values
        
        interaction_matrix = csr_matrix(
            (weights, (playlist_ids, track_ids)),
            shape=(len(mappings['playlist2id']), len(mappings['track2id']))
        )
        
        # Build binary interaction matrix for collaborative filtering
        binary_weights = np.ones(len(track_df))
        binary_interaction_matrix = csr_matrix(
            (binary_weights, (playlist_ids, track_ids)),
            shape=(len(mappings['playlist2id']), len(mappings['track2id']))
        )
        
        logger.info(f"Interaction matrix shape: {interaction_matrix.shape}")
        logger.info(f"Sparsity: {1 - interaction_matrix.nnz / (interaction_matrix.shape[0] * interaction_matrix.shape[1]):.4f}")
        
        return interaction_matrix, binary_interaction_matrix
    
    def extract_enhanced_tfidf(self, track_df: pd.DataFrame) -> Tuple[csr_matrix, TfidfVectorizer]:
        """Extract simplified TF-IDF features for better performance"""
        logger.info("📝 Extracting TF-IDF features...")
        
        # Simplified text combination - just track and artist names
        texts = []
        for _, row in track_df.iterrows():
            # Simple combination: track name + artist name
            combined_text = f"{row['track_name']} {row['artist_name']}"
            texts.append(combined_text)
        
        # Simplified TF-IDF configuration
        vectorizer = TfidfVectorizer(
            max_features=self.config['tfidf_max_features'],
            ngram_range=(1, 1),  # Only unigrams for speed
            lowercase=True,
            strip_accents='unicode',
            min_df=3,            # Higher min_df for efficiency
            max_df=0.8,          # Lower max_df to remove very common words
            token_pattern=r'\b\w{2,}\b'  # Only words with 2+ characters
        )
        
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        logger.info(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
        logger.info(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
        
        return tfidf_matrix, vectorizer
    
    def extract_enhanced_word2vec(self, track_df: pd.DataFrame) -> Tuple[np.ndarray, Word2Vec]:
        """Extract simplified Word2Vec embeddings for better performance"""
        logger.info("🧠 Training simplified Word2Vec model...")
        
        # Simplified corpus preparation
        corpus = []
        for _, row in track_df.iterrows():
            # Simple tokenization: track name + artist name
            words = []
            
            # Clean and tokenize track name
            track_words = [w.lower() for w in row['track_name'].split() if len(w) > 1]
            words.extend(track_words)
            
            # Clean and tokenize artist name
            artist_words = [w.lower() for w in row['artist_name'].split() if len(w) > 1]
            words.extend(artist_words)
            
            # Add duration category as a single token
            words.append(row['duration_category'])
            
            if len(words) >= 2:  # Ensure minimum sentence length
                corpus.append(words)
        
        logger.info(f"Prepared corpus with {len(corpus)} sentences")
        
        # Train simplified Word2Vec model
        model = Word2Vec(
            sentences=corpus,
            vector_size=self.config['word2vec_vector_size'],
            window=self.config['word2vec_window'],
            min_count=self.config['word2vec_min_count'],
            workers=min(4, cpu_count()),  # Limit workers for stability
            sg=0,           # Use CBOW (faster than skip-gram)
            epochs=5,       # Reduced epochs for speed
            negative=5,     # Reduced negative sampling
            sample=1e-3     # Subsampling threshold
        )
        
        # Generate embeddings efficiently
        embeddings = []
        zero_vector = np.zeros(self.config['word2vec_vector_size'])
        
        for words in tqdm(corpus, desc="Generating embeddings"):
            # Get valid word vectors
            word_vectors = [model.wv[word] for word in words if word in model.wv]
            
            if word_vectors:
                # Simple average of word vectors
                avg_vector = np.mean(word_vectors, axis=0)
            else:
                # Use zero vector as fallback
                avg_vector = zero_vector.copy()
            
            embeddings.append(avg_vector)
        
        embeddings_matrix = np.array(embeddings, dtype=np.float32)  # Use float32 to save memory
        
        logger.info(f"Word2Vec embeddings shape: {embeddings_matrix.shape}")
        logger.info(f"Vocabulary size: {len(model.wv.key_to_index)}")
        
        return embeddings_matrix, model
    
    def compute_enhanced_popularity(self, track_df: pd.DataFrame) -> Dict:
        """Compute simplified popularity metrics"""
        logger.info("📊 Computing popularity metrics...")
        
        popularity_metrics = {}
        
        # Basic frequency-based popularity
        track_counts = track_df.groupby('track_uri').size()
        max_count = track_counts.max()
        
        for track_uri, count in track_counts.items():
            # Simplified popularity metrics
            popularity_metrics[track_uri] = {
                'raw_frequency': count,
                'log_popularity': np.log1p(count),
                'normalized_popularity': count / max_count,
                'is_mainstream': count > track_counts.quantile(0.8),
                'is_niche': count < track_counts.quantile(0.2)
            }
        
        logger.info(f"Computed popularity metrics for {len(popularity_metrics)} tracks")
        return popularity_metrics
    
    def save_enhanced_artifacts(self, playlist_df: pd.DataFrame, track_df: pd.DataFrame,
                              mappings: Dict, interaction_matrix: csr_matrix,
                              binary_interaction_matrix: csr_matrix,
                              tfidf_matrix: csr_matrix, tfidf_vectorizer: TfidfVectorizer,
                              embeddings: np.ndarray, word2vec_model: Word2Vec,
                              popularity_metrics: Dict):
        """Save all processed artifacts"""
        logger.info(f"💾 Saving enhanced artifacts to {self.output_dir}...")
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save DataFrames
        playlist_df.to_pickle(os.path.join(self.output_dir, 'playlists.pkl'))
        track_df.to_pickle(os.path.join(self.output_dir, 'tracks.pkl'))
        
        # Save mappings and metadata
        with open(os.path.join(self.output_dir, 'mappings.pkl'), 'wb') as f:
            pickle.dump(mappings, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Save matrices
        save_npz(os.path.join(self.output_dir, 'interaction_matrix.npz'), interaction_matrix)
        save_npz(os.path.join(self.output_dir, 'binary_interaction_matrix.npz'), binary_interaction_matrix)
        save_npz(os.path.join(self.output_dir, 'tfidf_matrix.npz'), tfidf_matrix)
        
        # Save models
        with open(os.path.join(self.output_dir, 'tfidf_vectorizer.pkl'), 'wb') as f:
            pickle.dump(tfidf_vectorizer, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Save embeddings and Word2Vec model
        np.save(os.path.join(self.output_dir, 'track_embeddings.npy'), embeddings)
        word2vec_model.save(os.path.join(self.output_dir, 'word2vec_model.bin'))
        
        # Save popularity metrics
        with open(os.path.join(self.output_dir, 'popularity_metrics.pkl'), 'wb') as f:
            pickle.dump(popularity_metrics, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Save preprocessing configuration
        with open(os.path.join(self.output_dir, 'preprocessing_config.pkl'), 'wb') as f:
            pickle.dump(self.config, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Create summary report
        self._create_summary_report(playlist_df, track_df, mappings)
        
        logger.info("✅ All artifacts saved successfully!")
    
    def _create_summary_report(self, playlist_df: pd.DataFrame, track_df: pd.DataFrame, mappings: Dict):
        """Create a summary report of the preprocessing"""
        summary = {
            'preprocessing_stats': {
                'total_playlists': len(playlist_df),
                'total_tracks': len(track_df),
                'unique_tracks': mappings['stats']['num_tracks'],
                'unique_artists': mappings['stats']['num_artists'],
                'unique_albums': mappings['stats']['num_albums'],
                'avg_playlist_length': playlist_df['num_tracks'].mean(),
                'median_playlist_length': playlist_df['num_tracks'].median(),
                'avg_track_frequency': track_df['frequency'].mean(),
                'sparsity': 1 - len(track_df) / (len(playlist_df) * mappings['stats']['num_tracks'])
            },
            'feature_stats': {
                'tfidf_features': self.config['tfidf_max_features'],
                'word2vec_dimensions': self.config['word2vec_vector_size'],
                'duration_categories': len(set(track_df['duration_category'])),
                'popularity_range': [track_df['frequency'].min(), track_df['frequency'].max()]
            },
            'configuration': self.config
        }
        
        with open(os.path.join(self.output_dir, 'summary_report.json'), 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info("📋 Summary report created")
    
    def run_complete_pipeline(self):
        """Run the complete preprocessing pipeline"""
        logger.info("🚀 Starting complete preprocessing pipeline...")
        
        try:
            # Step 1: Load data
            playlists, track_counter = self.load_playlists_efficiently()
            
            # Step 2: Stratified sampling
            sampled_playlists = self.enhanced_stratified_sampling(playlists, track_counter)
            
            # Step 3: Extract and normalize data
            playlist_df, track_df = self.extract_and_normalize_data(sampled_playlists, track_counter)
            
            # Memory cleanup
            del playlists, sampled_playlists
            gc.collect()
            
            # Step 4: Build mappings
            mappings = self.build_enhanced_mappings(track_df)
            
            # Step 5: Build matrices
            interaction_matrix, binary_interaction_matrix = self.build_optimized_matrices(track_df, mappings)
            
            # Step 6: Extract features
            tfidf_matrix, tfidf_vectorizer = self.extract_enhanced_tfidf(track_df)
            embeddings, word2vec_model = self.extract_enhanced_word2vec(track_df)
            
            # Step 7: Compute popularity metrics
            popularity_metrics = self.compute_enhanced_popularity(track_df)
            
            # Step 8: Save everything
            self.save_enhanced_artifacts(
                playlist_df, track_df, mappings,
                interaction_matrix, binary_interaction_matrix,
                tfidf_matrix, tfidf_vectorizer,
                embeddings, word2vec_model,
                popularity_metrics
            )
            
            logger.info("🎉 Complete preprocessing pipeline finished successfully!")
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            raise


# Main execution
if __name__ == '__main__':
    # Configuration
    BASE_DIR = '/Users/vishnupb/Desktop/spotify_recommender'
    OUTPUT_DIR = os.path.join(BASE_DIR, 'v3', 'processed')
    MAX_PLAYLISTS = 80000  # Reduced to 80,000
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Initialize and run preprocessor
    preprocessor = OptimizedSpotifyPreprocessor(
        base_dir=BASE_DIR,
        output_dir=OUTPUT_DIR,
        max_playlists=MAX_PLAYLISTS
    )
    
    # Run complete pipeline
    preprocessor.run_complete_pipeline()
    
    print("✅ Preprocessing complete! Ready for hybrid recommendation system.")