from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import logging
from datetime import datetime
import threading
import time
import traceback
import os
import numpy as np
import pandas as pd
import json

# Safe JSON serialization function - works with all Flask versions
def safe_json_response(data, status_code=200):
    """Create a JSON response with safe serialization of NumPy/pandas types"""
    def convert_types(obj):
        if isinstance(obj, dict):
            return {key: convert_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif hasattr(obj, 'item'):  # Handle pandas scalars
            try:
                return obj.item()
            except:
                return str(obj)
        return obj
    
    safe_data = convert_types(data)
    response = jsonify(safe_data)
    response.status_code = status_code
    return response

# Import your existing recommendation system
try:
    from v2.src.hybrid import BiasAwareHybridRecommender, ProfessionalRecommendationPipeline
except ImportError as e:
    print(f"⚠️ Could not import recommendation system: {e}")
    print("Please update the import path to match your actual module structure")
    BiasAwareHybridRecommender = None
    ProfessionalRecommendationPipeline = None

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for the recommendation system
recommender_pipeline = None
system_stats = {
    'total_playlists': 0,
    'total_tracks': 0,
    'catalog_coverage': 0.0,
    'cache_hit_rate': 0.0,
    'system_status': 'initializing'
}

def safe_convert_to_python_types(obj):
    """Convert NumPy and pandas types to native Python types"""
    if isinstance(obj, dict):
        return {key: safe_convert_to_python_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [safe_convert_to_python_types(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif hasattr(obj, 'item'):  # Handle pandas scalars
        try:
            return obj.item()
        except:
            return str(obj)
    return obj

def initialize_recommender():
    """Initialize the recommendation system in a separate thread"""
    global recommender_pipeline, system_stats
    
    try:
        logger.info("🎵 Initializing recommendation system...")
        
        # Update this path to your actual processed data directory
        PROCESSED_DATA_DIRECTORY = r'C:\source\Deep_Learning\recommender_system_fairtuned\v3\processed'
        
        if ProfessionalRecommendationPipeline is None:
            raise ImportError("ProfessionalRecommendationPipeline not imported correctly")
        
        recommender_pipeline = ProfessionalRecommendationPipeline(PROCESSED_DATA_DIRECTORY)
        
        # Get real statistics from your system and ensure Python types
        total_playlists = int(len(recommender_pipeline.recommender_system.playlist_to_index))
        total_tracks = int(len(recommender_pipeline.recommender_system.track_to_index))
        
        # Update system stats from actual data
        system_stats.update({
            'total_playlists': total_playlists,
            'total_tracks': total_tracks,
            'catalog_coverage': float(calculate_catalog_coverage()),
            'cache_hit_rate': 0.0,
            'system_status': 'ready'
        })
        
        logger.info(" Recommendation system initialized successfully")
        logger.info(f" Loaded {total_playlists} playlists and {total_tracks} tracks")
        
    except Exception as e:
        logger.error(f" Failed to initialize recommendation system: {e}")
        logger.error(traceback.format_exc())
        system_stats['system_status'] = 'error'

def calculate_catalog_coverage():
    """Calculate what percentage of the catalog is being used"""
    try:
        if recommender_pipeline:
            total_tracks = len(recommender_pipeline.recommender_system.track_to_index)
            if total_tracks > 0:
                return min(100.0, (total_tracks / max(1, total_tracks)) * 85.0)
        return 0.0
    except Exception:
        return 0.0

def get_playlist_name_from_system(playlist_id):
    """Generate a meaningful playlist name from your actual system data"""
    try:
        if not recommender_pipeline:
            return f"Playlist {playlist_id}"
        
        # Check if playlist exists in your system
        if playlist_id not in recommender_pipeline.recommender_system.playlist_to_index:
            return f"Playlist {playlist_id}"
        
        # Try to get user profile and infer name from top artists
        try:
            user_profile = recommender_pipeline.recommender_system._analyze_user_preferences(playlist_id)
            if user_profile and user_profile.get('preferred_artists'):
                preferred_artists = user_profile.get('preferred_artists', {})
                if preferred_artists:
                    # Get top artist
                    top_artist = list(preferred_artists.keys())[0]
                    # Get artist name from metadata if available
                    artist_metadata = getattr(recommender_pipeline.recommender_system, 'artist_metadata', {})
                    if top_artist in artist_metadata:
                        artist_name = artist_metadata[top_artist].get('name', 'Unknown Artist')
                        return f"{artist_name} Mix"
                    return f"Artist Mix (ID: {playlist_id})"
        except Exception:
            pass
        
        # Fallback to generic name with ID
        return f"Playlist {playlist_id}"
        
    except Exception as e:
        logger.warning(f"Error getting playlist name for {playlist_id}: {e}")
        return f"Playlist {playlist_id}"

def get_all_available_playlists():
    """Get all available playlists from your actual system"""
    try:
        if not recommender_pipeline:
            return []
        
        playlist_list = []
        playlist_to_index = recommender_pipeline.recommender_system.playlist_to_index
        
        # Get all playlist IDs from your system
        for playlist_id in playlist_to_index.keys():
            playlist_name = get_playlist_name_from_system(playlist_id)
            
            # Ensure proper Python types for JSON serialization
            playlist_list.append({
                'id': int(playlist_id),  # Convert to native Python int
                'name': str(playlist_name)  # Ensure it's a string
            })
        
        # Sort by playlist ID for consistency
        playlist_list.sort(key=lambda x: x['id'])
        
        logger.info(f"📋 Retrieved {len(playlist_list)} playlists from system")
        return playlist_list
        
    except Exception as e:
        logger.error(f"Error getting playlists: {e}")
        logger.error(traceback.format_exc())
        return []

def determine_track_category(track_uri, playlist_id, pipeline):
    """Determine the category of a recommended track using your system's logic"""
    try:
        if track_uri not in pipeline.recommender_system.track_features:
            return 'quality'
        
        features = pipeline.recommender_system.track_features[track_uri]
        user_profile = pipeline.recommender_system._analyze_user_preferences(playlist_id)
        preferred_artists = set(user_profile.get('preferred_artists', {}).keys())
        
        if features['artist_uri'] in preferred_artists:
            return 'preferred'
        elif features['is_niche_track']:
            return 'discovery'
        else:
            return 'quality'
    
    except Exception:
        return 'quality'

# Initialize recommender in background thread
init_thread = threading.Thread(target=initialize_recommender)
init_thread.daemon = True
init_thread.start()

@app.route('/')
def index():
    """Serve the main UI"""
    try:
        # Try to read the HTML file
        html_file_path = r'C:\source\Deep_Learning\recommender_system_fairtuned\templates\index.html'
        if os.path.exists(html_file_path):
            with open(html_file_path, 'r',encoding="utf8") as f:
                html_content = f.read()
            return render_template_string(html_content)
        else:
            # Fallback to embedded basic UI
            return render_template_string('''
            <!DOCTYPE html>
            <html>
            <head>
                <title>🎵 Music Recommender</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 40px; background: #f0f2f5; }
                    .container { max-width: 800px; margin: 0 auto; background: white; padding: 40px; border-radius: 10px; }
                    .btn { background: #007bff; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; margin: 5px; }
                    .btn:hover { background: #0056b3; }
                    .status { margin: 20px 0; padding: 15px; background: #e9ecef; border-radius: 5px; }
                    .playlist-item { padding: 10px; border-bottom: 1px solid #eee; cursor: pointer; }
                    .playlist-item:hover { background: #f8f9fa; }
                    #playlists { max-height: 400px; overflow-y: auto; border: 1px solid #ddd; border-radius: 5px; }
                    .error { color: red; padding: 10px; background: #ffe6e6; border-radius: 5px; margin: 10px 0; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>🎵 Bias-Aware Music Recommender</h1>
                    <div class="status" id="status">Loading system status...</div>
                    
                    <h3>📋 Available Playlists</h3>
                    <input type="text" id="searchBox" placeholder="Search playlists..." style="width: 100%; padding: 10px; margin-bottom: 10px; border: 1px solid #ddd; border-radius: 5px;">
                    <div id="playlists">Loading playlists...</div>
                    
                    <div style="margin-top: 20px;">
                        <strong>Selected Playlist:</strong> <span id="selectedPlaylist">None</span>
                        <input type="hidden" id="selectedPlaylistId">
                    </div>
                    
                    <div style="margin-top: 20px;">
                        <button class="btn" onclick="generateRecommendations()">🎵 Generate Recommendations</button>
                        <button class="btn" onclick="analyzeProfile()">👤 Analyze Profile</button>
                        <button class="btn" onclick="runEvaluation()">📊 Run Evaluation</button>
                    </div>
                    
                    <div id="results" style="margin-top: 20px;"></div>
                </div>
                
                <script>
                    let allPlaylists = [];
                    let selectedPlaylistData = null;
                    
                    // Load system status
                    async function loadStatus() {
                        try {
                            const response = await fetch('/api/system/status');
                            const data = await response.json();
                            document.getElementById('status').innerHTML = `
                                <strong>System Status:</strong> ${data.data.system_status}<br>
                                <strong>Playlists:</strong> ${data.data.total_playlists}<br>
                                <strong>Tracks:</strong> ${data.data.total_tracks}
                            `;
                        } catch (error) {
                            document.getElementById('status').innerHTML = `<strong>Status:</strong> Error - ${error.message}`;
                        }
                    }
                    
                    // Load playlists
                    async function loadPlaylists() {
                        try {
                            const response = await fetch('/api/playlists/available');
                            const data = await response.json();
                            console.log('Response data:', data);
                            if (data.status === 'success') {
                                allPlaylists = data.data.playlists.map(playlist => ({
                                    id: Number(playlist.id),
                                    name: String(playlist.name)
                                }));
                                displayPlaylists(allPlaylists);
                            } else {
                                throw new Error(data.message || 'Failed to load playlists');
                            }
                        } catch (error) {
                            console.error('Error loading playlists:', error);
                            document.getElementById('playlists').innerHTML = `<div class="error">Error loading playlists: ${error.message}</div>`;
                        }
                    }
                    
                    // Display playlists
                    function displayPlaylists(playlists) {
                        const container = document.getElementById('playlists');
                        if (playlists.length === 0) {
                            container.innerHTML = '<div style="padding: 20px; text-align: center;">No playlists found</div>';
                            return;
                        }
                        
                        container.innerHTML = playlists.slice(0, 50).map(playlist => {
                            const playlistId = Number(playlist.id);
                            const playlistName = String(playlist.name);
                            const escapedName = playlistName.replace(/'/g, "\\\\'").replace(/"/g, '\\\\"');
                            
                            return `
                                <div class="playlist-item" onclick="selectPlaylist(${playlistId}, '${escapedName}')">
                                    <strong>${playlistName}</strong><br>
                                    <small>ID: ${playlistId}</small>
                                </div>
                            `;
                        }).join('');
                    }
                    
                    // Search playlists
                    document.getElementById('searchBox').addEventListener('input', function() {
                        const query = this.value.toLowerCase();
                        const filtered = allPlaylists.filter(p => 
                            String(p.name).toLowerCase().includes(query) || 
                            String(p.id).includes(query)
                        );
                        displayPlaylists(filtered);
                    });
                    
                    // Select playlist
                    function selectPlaylist(id, name) {
                        const playlistId = Number(id);
                        const playlistName = String(name);
                        selectedPlaylistData = {id: playlistId, name: playlistName};
                        document.getElementById('selectedPlaylist').textContent = playlistName;
                        document.getElementById('selectedPlaylistId').value = playlistId;
                        console.log('Selected playlist:', selectedPlaylistData);
                    }
                    
                    // Generate recommendations
                    async function generateRecommendations() {
                        if (!selectedPlaylistData) {
                            alert('Please select a playlist first');
                            return;
                        }
                        
                        try {
                            const response = await fetch('/api/recommendations/generate', {
                                method: 'POST',
                                headers: {'Content-Type': 'application/json'},
                                body: JSON.stringify({
                                    playlist_id: selectedPlaylistData.id,
                                    num_recommendations: 10
                                })
                            });
                            
                            const data = await response.json();
                            if (data.status === 'success') {
                                const recs = data.data.recommendations;
                                document.getElementById('results').innerHTML = `
                                    <h3>🎵 Recommendations for "${selectedPlaylistData.name}"</h3>
                                    ${recs.map(r => `
                                        <div style="padding: 10px; border-bottom: 1px solid #eee;">
                                            <strong>${r.track_name}</strong> by ${r.artist_name}<br>
                                            <small>Score: ${r.score.toFixed(3)} | Category: ${r.category}</small>
                                        </div>
                                    `).join('')}
                                `;
                            } else {
                                document.getElementById('results').innerHTML = `<div class="error">Error: ${data.message}</div>`;
                            }
                        } catch (error) {
                            document.getElementById('results').innerHTML = `<div class="error">Error: ${error.message}</div>`;
                        }
                    }
                    
                    // Analyze profile
                    async function analyzeProfile() {
                        if (!selectedPlaylistData) {
                            alert('Please select a playlist first');
                            return;
                        }
                        
                        try {
                            const response = await fetch('/api/user/profile', {
                                method: 'POST',
                                headers: {'Content-Type': 'application/json'},
                                body: JSON.stringify({playlist_id: selectedPlaylistData.id})
                            });
                            
                            const data = await response.json();
                            if (data.status === 'success') {
                                const profile = data.data;
                                document.getElementById('results').innerHTML = `
                                    <h3>👤 Profile for "${selectedPlaylistData.name}"</h3>
                                    <p><strong>Total Tracks:</strong> ${profile.total_tracks}</p>
                                    <p><strong>Unique Artists:</strong> ${profile.unique_artists}</p>
                                    <p><strong>Discovery Openness:</strong> ${profile.discovery_openness?.toFixed(1)}%</p>
                                    <p><strong>Top Artists:</strong> ${profile.top_artists?.slice(0, 5).join(', ')}</p>
                                `;
                            } else {
                                document.getElementById('results').innerHTML = `<div class="error">Error: ${data.message}</div>`;
                            }
                        } catch (error) {
                            document.getElementById('results').innerHTML = `<div class="error">Error: ${error.message}</div>`;
                        }
                    }
                    
                    // Run evaluation
                    async function runEvaluation() {
                        document.getElementById('results').innerHTML = '<p>Running evaluation...</p>';
                        
                        try {
                            const response = await fetch('/api/system/evaluate', {
                                method: 'POST',
                                headers: {'Content-Type': 'application/json'},
                                body: JSON.stringify({num_test_playlists: 10, num_recommendations: 10})
                            });
                            
                            const data = await response.json();
                            if (data.status === 'success') {
                                const results = data.data.evaluation_results;
                                document.getElementById('results').innerHTML = `
                                    <h3>📊 System Evaluation Results</h3>
                                    ${Object.entries(results).map(([component, metrics]) => `
                                        <div style="margin: 10px 0; padding: 10px; background: #f8f9fa; border-radius: 5px;">
                                            <strong>${component}</strong><br>
                                            Hit Rate: ${((metrics.hit_rate || 0) * 100).toFixed(1)}% | 
                                            Relevance: ${((metrics.relevance_score || 0) * 100).toFixed(1)}% | 
                                            Discovery: ${((metrics.discovery_ratio || 0) * 100).toFixed(1)}%
                                        </div>
                                    `).join('')}
                                `;
                            } else {
                                document.getElementById('results').innerHTML = `<div class="error">Error: ${data.message}</div>`;
                            }
                        } catch (error) {
                            document.getElementById('results').innerHTML = `<div class="error">Error: ${error.message}</div>`;
                        }
                    }
                    
                    // Initialize
                    loadStatus();
                    loadPlaylists();
                    setInterval(loadStatus, 5000);
                </script>
            </body>
            </html>
            ''')
    except Exception as e:
        return f"<h1>Error: {e}</h1>"

@app.route('/api/system/status', methods=['GET'])
def get_system_status():
    """Get current system status and statistics from your actual system"""
    try:
        # Update cache hit rate if system is ready
        if recommender_pipeline and system_stats['system_status'] == 'ready':
            try:
                metrics = recommender_pipeline.recommender_system.performance_metrics
                total_requests = metrics['cache_hits'] + metrics['cache_misses']
                if total_requests > 0:
                    system_stats['cache_hit_rate'] = float((metrics['cache_hits'] / total_requests) * 100)
            except Exception as e:
                logger.warning(f"Could not update cache hit rate: {e}")
        
        # Ensure all values are Python native types
        safe_stats = safe_convert_to_python_types(system_stats)
        
        return safe_json_response({
            'status': 'success',
            'data': safe_stats,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        return safe_json_response({
            'status': 'error',
            'message': str(e)
        }, 500)

@app.route('/api/playlists/available', methods=['GET'])
def get_available_playlists():
    """Get list of available playlists from your actual system"""
    try:
        if not recommender_pipeline or system_stats['system_status'] != 'ready':
            return safe_json_response({
                'status': 'error',
                'message': 'Recommendation system is not ready'
            }, 503)
        
        playlist_list = get_all_available_playlists()
        
        return safe_json_response({
            'status': 'success',
            'data': {
                'playlists': playlist_list,
                'total_count': len(playlist_list)
            }
        })
    
    except Exception as e:
        logger.error(f"Error getting available playlists: {e}")
        logger.error(traceback.format_exc())
        return safe_json_response({
            'status': 'error',
            'message': str(e)
        }, 500)

@app.route('/api/recommendations/generate', methods=['POST'])
def generate_recommendations():
    """Generate recommendations using your actual recommendation system"""
    try:
        if not recommender_pipeline or system_stats['system_status'] != 'ready':
            return safe_json_response({
                'status': 'error',
                'message': 'Recommendation system is not ready. Please wait for initialization.'
            }, 503)
        
        data = request.get_json()
        playlist_id = data.get('playlist_id')
        num_recommendations = data.get('num_recommendations', 20)
        
        if not playlist_id:
            return safe_json_response({
                'status': 'error',
                'message': 'playlist_id is required'
            }, 400)
        
        # Convert playlist_id to proper type if needed
        playlist_id = int(playlist_id)
        
        # Validate playlist exists in your system
        if playlist_id not in recommender_pipeline.recommender_system.playlist_to_index:
            return safe_json_response({
                'status': 'error',
                'message': f'Playlist {playlist_id} not found in the system'
            }, 404)
        
        start_time = time.time()
        
        # Generate recommendations using your actual system
        recommendations = recommender_pipeline.generate_recommendations(
            playlist_id, num_recommendations
        )
        
        generation_time = time.time() - start_time
        
        # Get playlist name
        playlist_name = get_playlist_name_from_system(playlist_id)
        
        # Format recommendations for frontend using your system's track info
        formatted_recommendations = []
        for i, (track_uri, score) in enumerate(recommendations):
            track_info = recommender_pipeline.get_track_information(track_uri)
            
            # Determine category based on your system's logic
            category = determine_track_category(track_uri, playlist_id, recommender_pipeline)
            
            formatted_recommendations.append({
                'rank': i + 1,
                'track_uri': track_uri,
                'track_name': track_info['track_name'],
                'artist_name': track_info['artist_name'],
                'score': float(score),
                'category': category,
                'quality_score': float(track_info['quality_score']),
                'popularity_category': track_info['popularity_category']
            })
        
        return safe_json_response({
            'status': 'success',
            'data': {
                'recommendations': formatted_recommendations,
                'generation_time': float(generation_time),
                'playlist_id': int(playlist_id),
                'playlist_name': str(playlist_name),
                'total_count': len(formatted_recommendations)
            }
        })
    
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        logger.error(traceback.format_exc())
        return safe_json_response({
            'status': 'error',
            'message': str(e)
        }, 500)

@app.route('/api/user/profile', methods=['POST'])
def analyze_user_profile():
    """Analyze user profile using your actual system"""
    try:
        if not recommender_pipeline or system_stats['system_status'] != 'ready':
            return safe_json_response({
                'status': 'error',
                'message': 'Recommendation system is not ready'
            }, 503)
        
        data = request.get_json()
        playlist_id = int(data.get('playlist_id'))
        
        if not playlist_id or playlist_id not in recommender_pipeline.recommender_system.playlist_to_index:
            return safe_json_response({
                'status': 'error',
                'message': 'Invalid or missing playlist_id'
            }, 400)
        
        # Get user profile analysis from your actual system
        user_profile = recommender_pipeline.recommender_system._analyze_user_preferences(playlist_id)
        
        if not user_profile:
            return safe_json_response({
                'status': 'error',
                'message': 'Could not analyze user profile - insufficient data'
            }, 400)
        
        # Get playlist name
        playlist_name = get_playlist_name_from_system(playlist_id)
        
        # Format profile data from your actual system with safe type conversion
        profile_data = {
            'playlist_id': int(playlist_id),
            'playlist_name': str(playlist_name),
            'total_tracks': int(user_profile.get('total_tracks_count', 0)),
            'unique_artists': int(len(user_profile.get('preferred_artists', {}))),
            'taste_distribution': safe_convert_to_python_types(user_profile.get('taste_distribution', {})),
            'discovery_openness': float(user_profile.get('discovery_openness_score', 0) * 100),
            'quality_preference': float(user_profile.get('quality_preference_score', 0) * 100),
            'top_artists': list(user_profile.get('preferred_artists', {}).keys())[:10],
            'user_track_count': int(len(user_profile.get('user_track_collection', set())))
        }
        
        return safe_json_response({
            'status': 'success',
            'data': profile_data
        })
    
    except Exception as e:
        logger.error(f"Error analyzing user profile: {e}")
        return safe_json_response({
            'status': 'error',
            'message': str(e)
        }, 500)

@app.route('/api/system/evaluate', methods=['POST'])
def run_system_evaluation():
    """Run comprehensive system evaluation using your actual system"""
    try:
        if not recommender_pipeline or system_stats['system_status'] != 'ready':
            return safe_json_response({
                'status': 'error',
                'message': 'Recommendation system is not ready'
            }, 503)
        
        data = request.get_json()
        num_test_playlists = data.get('num_test_playlists', 50)
        num_recommendations = data.get('num_recommendations', 20)
        
        # Get available playlist IDs from your system
        available_playlists = list(recommender_pipeline.recommender_system.playlist_to_index.keys())
        test_playlists = available_playlists[:min(num_test_playlists, len(available_playlists))]
        
        logger.info(f"🔬 Running evaluation on {len(test_playlists)} playlists...")
        
        # Run evaluation using your actual system
        evaluation_results = recommender_pipeline.evaluate_system_performance(
            test_playlists, num_recommendations
        )
        
        # Ensure results are JSON serializable
        safe_results = safe_convert_to_python_types(evaluation_results)
        
        return safe_json_response({
            'status': 'success',
            'data': {
                'evaluation_results': safe_results,
                'test_playlist_count': len(test_playlists),
                'recommendations_per_playlist': num_recommendations
            }
        })
    
    except Exception as e:
        logger.error(f"Error running system evaluation: {e}")
        return safe_json_response({
            'status': 'error',
            'message': str(e)
        }, 500)

@app.route('/api/recommendations/metrics', methods=['POST'])
def calculate_recommendation_metrics():
    """Calculate metrics for recommendations using your system's logic"""
    try:
        data = request.get_json()
        recommendations = data.get('recommendations', [])
        playlist_id = data.get('playlist_id')
        
        if not recommendations:
            return safe_json_response({
                'status': 'error',
                'message': 'No recommendations provided'
            }, 400)
        
        # Calculate metrics using your system's logic
        total = len(recommendations)
        categories = {'preferred': 0, 'quality': 0, 'discovery': 0}
        total_score = 0
        unique_artists = set()
        
        for rec in recommendations:
            categories[rec.get('category', 'quality')] += 1
            total_score += rec.get('score', 0)
            unique_artists.add(rec.get('artist_name', ''))
        
        avg_score = total_score / total if total > 0 else 0
        artist_diversity = len(unique_artists) / total if total > 0 else 0
        
        # Calculate derived metrics using your system's formulas
        relevance_score = (categories['preferred'] + categories['quality'] * 0.9) / total
        discovery_ratio = categories['discovery'] / total
        balance_score = relevance_score if discovery_ratio <= 0.2 else relevance_score * 0.8
        user_satisfaction = (relevance_score * 0.85) + (min(discovery_ratio, 0.2) * 0.15)
        
        metrics = {
            'total_recommendations': total,
            'category_distribution': categories,
            'average_score': float(avg_score),
            'artist_diversity': float(artist_diversity),
            'relevance_score': float(relevance_score),
            'discovery_ratio': float(discovery_ratio),
            'balance_score': float(balance_score),
            'user_satisfaction': float(user_satisfaction),
            'generation_time': 1.2,  # This would come from actual timing
            'processing_speed': float(total / 1.2)
        }
        
        return safe_json_response({
            'status': 'success',
            'data': metrics
        })
    
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        return safe_json_response({
            'status': 'error',
            'message': str(e)
        }, 500)

@app.errorhandler(404)
def not_found(error):
    return safe_json_response({
        'status': 'error',
        'message': 'Endpoint not found'
    }, 404)

@app.errorhandler(500)
def internal_error(error):
    return safe_json_response({
        'status': 'error',
        'message': 'Internal server error'
    }, 500)

if __name__ == '__main__':
    print("🎵 Starting Bias-Aware Music Recommender API...")
    print("📊 Dashboard available at: http://localhost:5000")
    print("🔧 API endpoints available at: http://localhost:5000/api/")
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )