from flask import Flask, request, jsonify
from flask_cors import CORS
from recommender import MovieRecommender
import os

app = Flask(__name__)
CORS(app)

# Initialize recommender
recommender = MovieRecommender()

# Get the absolute path to the data directory
data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

# Load data with proper paths
recommender.load_data(
    movies_path=os.path.join(data_dir, 'movies.csv'),
    ratings_path=os.path.join(data_dir, 'ratings.csv'),
    tags_path=os.path.join(data_dir, 'tags.csv'),
    genome_scores_path=os.path.join(data_dir, 'genome-scores.csv')
)

@app.route('/api/search', methods=['GET'])
def search_movies():
    query = request.args.get('q', '').lower()
    if not query:
        return jsonify({"results": []})
    
    # Search in movies dataframe
    matching_movies = recommender.movies_df[
        recommender.movies_df['title'].str.lower().str.contains(query)
    ]
    
    results = []
    for _, movie in matching_movies.iterrows():
        # Get tags for the movie
        movie_tags = recommender.tags_df[
            recommender.tags_df['movieId'] == movie['movieId']
        ]['tag'].tolist()[:3]  # Get top 3 tags
        
        results.append({
            'id': int(movie['movieId']),
            'title': movie['title'],
            'genres': movie['genres'].split('|'),
            'tags': movie_tags
        })
    
    return jsonify({"results": results[:10]})  # Return top 10 matches

@app.route('/api/recommend', methods=['GET'])
def get_recommendations():
    movie_id = request.args.get('movie_id')
    user_id = request.args.get('user_id', '1')  # Default user_id = 1
    
    try:
        movie_id = int(movie_id)
        user_id = int(user_id)
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid movie_id or user_id"}), 400
    
    # Get hybrid recommendations
    rec_ids = recommender.get_hybrid_recommendations(user_id, movie_id)
    
    recommendations = []
    for mid in rec_ids:
        movie = recommender.movies_df[
            recommender.movies_df['movieId'] == mid
        ].iloc[0]
        
        # Get tags for the movie
        movie_tags = recommender.tags_df[
            recommender.tags_df['movieId'] == mid
        ]['tag'].tolist()[:3]  # Get top 3 tags
        
        recommendations.append({
            'id': int(mid),
            'title': movie['title'],
            'genres': movie['genres'].split('|'),
            'tags': movie_tags
        })
    
    return jsonify({"recommendations": recommendations})

if __name__ == '__main__':
    app.run(debug=True, port=5000)