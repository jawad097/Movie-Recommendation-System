# Movie Recommendation System

This project implements a hybrid movie recommendation system using collaborative filtering, content-based filtering, and tag genome scores. The system provides personalized movie recommendations based on user preferences, movie content, and user-generated tags.

## Features

- **Content-Based Filtering**: Recommends movies based on title, genre, and tag similarity
- **Collaborative Filtering**: Suggests movies based on user rating patterns
- **Hybrid Recommendations**: Combines both approaches for better accuracy
- **Tag-Based Analysis**: Utilizes user-generated tags and genome scores
- **Evaluation Metrics**: Includes precision, recall, and F1-score measurements

## Dataset

The system uses the MovieLens dataset, which includes:
- 27,278 movies
- 20,000,263 ratings
- 465,564 tags
- 11,709,768 genome scores

## Requirements

```bash
numpy==1.21.0
pandas==1.3.0
scikit-learn==0.24.2
scipy==1.7.0
nltk==3.6.2
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/movie-recommender.git
cd movie-recommender
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Download the MovieLens dataset and place it in the `data/` directory.

## Usage

Basic usage example:

```python
from src.recommender import MovieRecommender

# Initialize recommender
recommender = MovieRecommender()

echo "## Data Setup
The MovieLens dataset files are not included due to size limitations.
Download the dataset from: https://grouplens.org/datasets/movielens/
Place the following files in the data/ directory:
- movies.csv
- ratings.csv
- tags.csv

# Build models
recommender.build_content_based_features()
recommender.build_collaborative_filtering()

# Get recommendations
movie_id = 1  # Example: Toy Story
user_id = 1   # Example user

# Get different types of recommendations
content_recs = recommender.get_content_based_recommendations(movie_id)
collab_recs = recommender.get_collaborative_recommendations(user_id)
hybrid_recs = recommender.get_hybrid_recommendations(user_id, movie_id)
```

## Sample Output

Here's an example of hybrid recommendations:

```
Hybrid recommendations for user 1:
1. Toy Story 3 (2010)
   Genres: Adventure|Animation|Children|Comedy|Fantasy|IMAX
   Tags: tense, Alive toys, adventure

2. Fargo (1996)
   Genres: Comedy|Crime|Drama|Thriller
   Tags: crime gone awry, black comedy

3. Indiana Jones and the Temple of Doom (1984)
   Genres: Action|Adventure|Fantasy
   Tags: archaeology, Indiana Jones
```

## Project Structure

```
movie-recommender/
├── data/                 # Dataset files
├── src/                  # Source code
├── notebooks/            # Jupyter notebooks
├── requirements.txt      # Dependencies
└── README.md            # Documentation
```

## Model Components

1. **Content-Based Filtering**
   - Uses TF-IDF vectorization for movie features
   - Incorporates movie titles, genres, and tags
   - Calculates similarity using cosine similarity

2. **Collaborative Filtering**
   - Implements matrix factorization using SVD
   - Handles sparse user-movie interaction matrix
   - Predicts ratings for unwatched movies

3. **Hybrid System**
   - Combines recommendations from both approaches
   - Weights can be adjusted between methods
   - Provides more robust recommendations

## Future Improvements

- Add a web interface
- Implement real-time recommendation updates
- Add support for new movies/users
- Improve recommendation diversity
- Add more evaluation metrics

## Acknowledgments

- MovieLens dataset from GroupLens Research
