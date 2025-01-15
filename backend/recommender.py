import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix, hstack
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class MovieRecommender:
    def __init__(self):
        """Initialize the recommendation system"""
        self.movies_df = None
        self.ratings_df = None
        self.users_df = None
        self.movie_features = None
        
    def load_data(self, movies_path, ratings_path, tags_path=None, genome_scores_path=None):
        """
        Load the MovieLens dataset
        
        Parameters:
        movies_path (str): Path to movies CSV file
        ratings_path (str): Path to ratings CSV file
        tags_path (str): Path to tags CSV file (optional)
        genome_scores_path (str): Path to genome scores CSV file (optional)
        """
        # Load movies and ratings data
        self.movies_df = pd.read_csv(movies_path)
        self.ratings_df = pd.read_csv(ratings_path)
        
        # Load tags if provided
        if tags_path:
            self.tags_df = pd.read_csv(tags_path)
            print(f"Number of tags: {len(self.tags_df)}")
        else:
            self.tags_df = pd.DataFrame(columns=['movieId', 'tag'])
            
        # Load genome scores if provided
        if genome_scores_path:
            self.genome_scores_df = pd.read_csv(genome_scores_path)
            print(f"Number of genome scores: {len(self.genome_scores_df)}")
            # Pivot genome scores to create a movie-feature matrix
            self.genome_matrix = self.genome_scores_df.pivot(
                index='movieId', 
                columns='tagId', 
                values='relevance'
            ).fillna(0)
        else:
            self.genome_scores_df = None
            self.genome_matrix = None
            
        print("Data loaded successfully!")
        print(f"Number of movies: {len(self.movies_df)}")
        print(f"Number of ratings: {len(self.ratings_df)}")
        
    def build_content_based_features(self):
        """Build content-based features using TF-IDF and genome scores"""
        # Preprocess tags first
        self.preprocess_tags()
        
        # Merge movies with their tags
        movies_with_tags = pd.merge(
            self.movies_df,
            self.movie_tags,
            on='movieId',
            how='left'
        )
        
        # Fill NaN tags with empty string
        movies_with_tags['tag'] = movies_with_tags['tag'].fillna('')
        
        # Create text-based features
        content_features = (
            movies_with_tags['title'] + ' ' + 
            movies_with_tags['genres'] + ' ' + 
            movies_with_tags['tag']
        )
        
        # Create TF-IDF vectors
        tfidf = TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 2)
        )
        text_features = tfidf.fit_transform(content_features)
        
        # If genome scores are available, combine them with text features
        if self.genome_matrix is not None:
            # Get genome features for each movie
            movie_ids = movies_with_tags['movieId'].values
            genome_features = []
            
            for movie_id in movie_ids:
                if movie_id in self.genome_matrix.index:
                    genome_features.append(self.genome_matrix.loc[movie_id].values)
                else:
                    genome_features.append(np.zeros(self.genome_matrix.shape[1]))
                    
            genome_features = np.vstack(genome_features)
            
            # Convert genome features to sparse matrix
            genome_features_sparse = csr_matrix(genome_features)
            
            # Combine text and genome features
            self.movie_features = hstack([text_features, genome_features_sparse])
        else:
            self.movie_features = text_features
            
        print("Content-based features built successfully!")
        
    def preprocess_text(self, text):
        """
        Clean and preprocess text data
        
        Parameters:
        text (str): Text to preprocess
        
        Returns:
        str: Preprocessed text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and non-alphabetic tokens
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
        
        return ' '.join(tokens)

    def build_content_based_features(self):
        """Build content-based features using TF-IDF on movie descriptions/titles"""
        # Create TF-IDF vectors from movie titles and genres
        tfidf = TfidfVectorizer(stop_words='english')
        self.movie_features = tfidf.fit_transform(self.movies_df['title'] + ' ' + self.movies_df['genres'])
        
        print("Content-based features built successfully!")
        
    def get_content_based_recommendations(self, movie_id, n_recommendations=5):
        """
        Get content-based recommendations for a movie
        
        Parameters:
        movie_id (int): ID of the movie to get recommendations for
        n_recommendations (int): Number of recommendations to return
        
        Returns:
        list: List of recommended movie IDs
        """
        # Calculate similarity between movies
        movie_idx = self.movies_df[self.movies_df['movieId'] == movie_id].index[0]
        movie_similarities = cosine_similarity(self.movie_features[movie_idx], self.movie_features)
        
        # Get top similar movies
        similar_indices = movie_similarities.argsort()[0][-n_recommendations-1:-1][::-1]
        
        return self.movies_df.iloc[similar_indices]['movieId'].tolist()

    def build_collaborative_filtering(self, n_factors=50):
        """
        Build collaborative filtering model using SVD with memory optimization
        """
        # Create sparse matrix instead of dense pivot table
        from scipy.sparse import csr_matrix
        
        # Get unique users and movies
        unique_users = self.ratings_df['userId'].unique()
        unique_movies = self.ratings_df['movieId'].unique()
        
        # Create mappings for users and movies to matrix indices
        user_to_index = {user: i for i, user in enumerate(unique_users)}
        movie_to_index = {movie: i for i, movie in enumerate(unique_movies)}
        
        # Convert ratings to matrix format
        row_indices = [user_to_index[user] for user in self.ratings_df['userId']]
        col_indices = [movie_to_index[movie] for movie in self.ratings_df['movieId']]
        ratings = self.ratings_df['rating'].values
        
        # Create sparse matrix
        user_movie_matrix = csr_matrix((ratings, (row_indices, col_indices)), 
                                     shape=(len(unique_users), len(unique_movies)))
        
        # Store mappings for later use
        self.user_to_index = user_to_index
        self.movie_to_index = movie_to_index
        self.index_to_movie = {v: k for k, v in movie_to_index.items()}
        
        # Perform SVD on sparse matrix
        U, sigma, Vt = svds(user_movie_matrix, k=n_factors)
        
        # Convert to diagonal matrix
        sigma = np.diag(sigma)
        
        # Store the matrices for later use
        self.user_features = U
        self.movie_features_cf = Vt.T
        self.sigma = sigma
        
        print("Collaborative filtering model built successfully!")
        
    def get_collaborative_recommendations(self, user_id, n_recommendations=5):
        """
        Get collaborative filtering recommendations for a user
        
        Parameters:
        user_id (int): ID of the user to get recommendations for
        n_recommendations (int): Number of recommendations to return
        
        Returns:
        list: List of recommended movie IDs
        """
        # Get user index
        if user_id not in self.user_to_index:
            return []  # Return empty list if user not in training data
            
        user_idx = self.user_to_index[user_id]
        
        # Calculate predicted ratings for all movies
        predicted_ratings = np.dot(np.dot(self.user_features[user_idx], self.sigma), 
                                 self.movie_features_cf.T)
        
        # Get movies the user hasn't rated yet
        user_rated_movies = set(self.ratings_df[self.ratings_df['userId'] == user_id]['movieId'])
        
        # Create movie scores using the mapping
        movie_scores = []
        for movie_id in self.movie_to_index.keys():
            if movie_id not in user_rated_movies:
                movie_idx = self.movie_to_index[movie_id]
                movie_scores.append((movie_id, predicted_ratings[movie_idx]))
        
        # Sort and get top recommendations
        movie_scores = sorted(movie_scores, key=lambda x: x[1], reverse=True)
        
        return [movie_id for movie_id, _ in movie_scores[:n_recommendations]]
        
        # Get movies the user hasn't rated yet
        user_rated_movies = set(self.ratings_df[self.ratings_df['userId'] == user_id]['movieId'])
        all_movies = set(self.movies_df['movieId'])
        candidate_movies = list(all_movies - user_rated_movies)
        
        # Get top recommendations
        movie_scores = [(movie_id, predicted_ratings[self.movies_df[self.movies_df['movieId'] == movie_id].index[0]])
                       for movie_id in candidate_movies]
        movie_scores = sorted(movie_scores, key=lambda x: x[1], reverse=True)
        
        return [movie_id for movie_id, _ in movie_scores[:n_recommendations]]

    def get_hybrid_recommendations(self, user_id, movie_id, n_recommendations=5, 
                                 content_weight=0.5):
        """
        Get hybrid recommendations combining content-based and collaborative filtering
        
        Parameters:
        user_id (int): User ID
        movie_id (int): Movie ID
        n_recommendations (int): Number of recommendations to return
        content_weight (float): Weight for content-based recommendations (0 to 1)
        
        Returns:
        list: List of recommended movie IDs
        """
        try:
            # Get recommendations from both methods
            content_recs = set(self.get_content_based_recommendations(movie_id, n_recommendations * 2))
            collab_recs = set(self.get_collaborative_recommendations(user_id, n_recommendations * 2))
            
            # Handle case where one method fails
            if not content_recs:
                return list(collab_recs)[:n_recommendations]
            if not collab_recs:
                return list(content_recs)[:n_recommendations]
            
            # Combine recommendations
            # First, get movies that appear in both lists
            common_recs = list(content_recs & collab_recs)
            
            # Then, fill remaining spots with highest weighted recommendations
            remaining_spots = n_recommendations - len(common_recs)
            if remaining_spots > 0:
                content_unique = list(content_recs - set(common_recs))
                collab_unique = list(collab_recs - set(common_recs))
                
                # Calculate how many to take from each based on weights
                n_content = int(remaining_spots * content_weight)
                n_collab = remaining_spots - n_content
                
                hybrid_recs = (common_recs + 
                             content_unique[:n_content] + 
                             collab_unique[:n_collab])
            else:
                hybrid_recs = common_recs[:n_recommendations]
            
            return hybrid_recs[:n_recommendations]
        except Exception as e:
            print(f"Error in hybrid recommendations: {str(e)}")
            # Return content-based recommendations as fallback
            return self.get_content_based_recommendations(movie_id, n_recommendations)

    def evaluate_recommendations(self, test_size=0.2):
        """
        Evaluate the recommendation system using precision, recall, and F1 score
        
        Parameters:
        test_size (float): Proportion of data to use for testing
        
        Returns:
        dict: Dictionary containing evaluation metrics
        """
        # Split data into train and test sets
        train_data, test_data = train_test_split(
            self.ratings_df, 
            test_size=test_size, 
            random_state=42
        )
        
        # Train models on training data
        self.ratings_df = train_data
        self.build_collaborative_filtering()
        
        # Evaluate on test data
        predictions = []
        actuals = []
        
        for user_id in test_data['userId'].unique():
            # Get actual movies rated by user in test set
            actual_movies = set(test_data[test_data['userId'] == user_id]['movieId'])
            
            # Get recommended movies
            if len(actual_movies) > 0:
                sample_movie = list(actual_movies)[0]
                recommended_movies = set(self.get_hybrid_recommendations(
                    user_id, 
                    sample_movie, 
                    n_recommendations=10
                ))
                
                # Calculate binary predictions and actuals for metrics
                all_movies = set(self.movies_df['movieId'])
                for movie in all_movies:
                    predictions.append(1 if movie in recommended_movies else 0)
                    actuals.append(1 if movie in actual_movies else 0)
        
        # Calculate metrics
        metrics = {
            'precision': precision_score(actuals, predictions),
            'recall': recall_score(actuals, predictions),
            'f1': f1_score(actuals, predictions)
        }
        
        return metrics

if __name__ == "__main__":
    recommender = MovieRecommender()
    
    try:
        recommender.load_data(
            movies_path='movie.csv',
            ratings_path='rating.csv',
            tags_path='tag.csv',
            genome_scores_path='genome_scores.csv'
        )
        
        # Build both recommendation models
        recommender.build_content_based_features()
        recommender.build_collaborative_filtering()
        
        movie_id = 1
        user_id = 1
        
        def get_movie_info(movie_ids):
            """Helper function to get movie titles and tags"""
            movie_info = []
            for mid in movie_ids:
                movie_data = recommender.movies_df[recommender.movies_df['movieId'] == mid].iloc[0]
                if not recommender.tags_df.empty:
                    tags = recommender.tags_df[recommender.tags_df['movieId'] == mid]['tag'].tolist()
                    tags_str = ', '.join(tags[:3]) if tags else 'No tags'
                else:
                    tags_str = 'No tags available'
                movie_info.append((mid, movie_data['title'], movie_data['genres'], tags_str))
            return movie_info
        
        print("\nContent-based recommendations:")
        print(f"Movie {movie_id} ({recommender.movies_df[recommender.movies_df['movieId'] == movie_id]['title'].iloc[0]})")
        content_recs = recommender.get_content_based_recommendations(movie_id)
        print("Recommendations:")
        for mid, title, genres, tags in get_movie_info(content_recs):
            print(f"- {title}")
            print(f"  Genres: {genres}")
            print(f"  Tags: {tags}")
        
        print("\nCollaborative filtering recommendations:")
        collab_recs = recommender.get_collaborative_recommendations(user_id)
        print(f"Recommendations for user {user_id}:")
        for mid, title, genres, tags in get_movie_info(collab_recs):
            print(f"- {title}")
            print(f"  Genres: {genres}")
            print(f"  Tags: {tags}")
        
        print("\nHybrid recommendations:")
        hybrid_recs = recommender.get_hybrid_recommendations(user_id, movie_id)
        print(f"Hybrid recommendations for user {user_id}:")
        for mid, title, genres, tags in get_movie_info(hybrid_recs):
            print(f"- {title}")
            print(f"  Genres: {genres}")
            print(f"  Tags: {tags}")
        
        # Evaluate the system
        print("\nEvaluating recommendation system...")
        metrics = recommender.evaluate_recommendations(test_size=0.2)
        print("Evaluation metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.3f}")
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")