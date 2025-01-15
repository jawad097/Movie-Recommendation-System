import React, { useState, useEffect } from 'react';
import { Search } from 'lucide-react';

const API_URL = 'http://localhost:5000/api';

const MovieRecommender = () => {
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [selectedMovie, setSelectedMovie] = useState(null);
  const [recommendations, setRecommendations] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  // Debounced search
  useEffect(() => {
    if (!searchQuery) {
      setSearchResults([]);
      return;
    }

    const timeoutId = setTimeout(async () => {
      try {
        setIsLoading(true);
        const response = await fetch(`${API_URL}/search?q=${searchQuery}`);
        if (!response.ok) throw new Error('Search failed');
        const data = await response.json();
        setSearchResults(data.results);
        setError(null);
      } catch (err) {
        setError('Error searching movies');
        setSearchResults([]);
      } finally {
        setIsLoading(false);
      }
    }, 300);

    return () => clearTimeout(timeoutId);
  }, [searchQuery]);

  // Get recommendations when a movie is selected
  useEffect(() => {
    if (!selectedMovie) return;

    const fetchRecommendations = async () => {
      try {
        setIsLoading(true);
        const response = await fetch(`${API_URL}/recommend?movie_id=${selectedMovie.id}`);
        if (!response.ok) throw new Error('Failed to get recommendations');
        const data = await response.json();
        setRecommendations(data.recommendations);
        setError(null);
      } catch (err) {
        setError('Error fetching recommendations');
        setRecommendations([]);
      } finally {
        setIsLoading(false);
      }
    };

    fetchRecommendations();
  }, [selectedMovie]);

  const MovieCard = ({ movie, isSelected, onClick }) => (
    <div 
      className={`p-4 rounded-lg ${
        isSelected ? 'bg-blue-100' : 'bg-white'
      } border shadow-sm hover:shadow-md transition-all cursor-pointer`}
      onClick={() => onClick(movie)}
    >
      <div className="flex items-start justify-between">
        <div>
          <h3 className="font-semibold text-lg">{movie.title}</h3>
        </div>
      </div>
      <div className="mt-2">
        <div className="flex flex-wrap gap-1">
          {movie.genres.map((genre, idx) => (
            <span 
              key={idx}
              className="px-2 py-1 text-xs rounded-full bg-gray-100"
            >
              {genre}
            </span>
          ))}
        </div>
      </div>
      <div className="mt-2">
        <div className="flex flex-wrap gap-1">
          {movie.tags.map((tag, idx) => (
            <span 
              key={idx}
              className="px-2 py-1 text-xs rounded-full bg-blue-50"
            >
              #{tag}
            </span>
          ))}
        </div>
      </div>
    </div>
  );

  return (
    <div className="max-w-4xl mx-auto p-4">
      <div className="mb-8">
        <h1 className="text-3xl font-bold mb-4">Movie Recommender</h1>
        
        {/* Search Bar */}
        <div className="relative">
          <input
            type="text"
            className="w-full p-4 pl-12 text-lg border rounded-lg"
            placeholder="Search for a movie..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
          <Search className="absolute left-4 top-4 text-gray-400" />
        </div>

        {/* Search Results */}
        {searchResults.length > 0 && (
          <div className="mt-4 border rounded-lg">
            {searchResults.map((movie) => (
              <div
                key={movie.id}
                className="p-2 hover:bg-gray-50"
                onClick={() => setSelectedMovie(movie)}
              >
                <MovieCard movie={movie} onClick={() => setSelectedMovie(movie)} />
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Selected Movie */}
      {selectedMovie && (
        <div className="mb-8">
          <h2 className="text-xl font-semibold mb-4">Selected Movie</h2>
          <MovieCard 
            movie={selectedMovie} 
            isSelected={true}
            onClick={() => {}}
          />
        </div>
      )}

      {/* Recommendations */}
      {recommendations.length > 0 && (
        <div className="mb-8">
          <h2 className="text-xl font-semibold mb-4">Recommendations</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {recommendations.map((movie) => (
              <MovieCard 
                key={movie.id}
                movie={movie}
                onClick={() => setSelectedMovie(movie)}
              />
            ))}
          </div>
        </div>
      )}

      {/* Loading State */}
      {isLoading && (
        <div className="text-center py-4">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-gray-900 mx-auto"></div>
        </div>
      )}

      {/* Error State */}
      {error && (
        <div className="text-center py-4 text-red-500">
          {error}
        </div>
      )}
    </div>
  );
};

export default MovieRecommender;