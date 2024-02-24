from flask import Flask, request, jsonify, render_template, redirect, url_for
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load 'movies_df' from the data folder
movies_df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'preprocessed_data.csv'))

# Chemins vers les fichiers des modèles pré-entraînés
cosine_sim_path = os.path.join(os.path.dirname(__file__), 'models', 'movie_cosine_similarity_model.pkl')

# Importer les modèles directement avec joblib.load
cosine_sim = joblib.load(cosine_sim_path)

def get_recommendations(movie_title):
    # Check if the movie title exists in the DataFrame
    if movie_title not in movies_df['original_title'].values:
        return []

    # Utilize the models loaded for the recommendation
    idx = movies_df.index.get_loc(movies_df.index[movies_df['original_title'] == movie_title].tolist()[0])

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Exclude the movie itself (the most similar would be the movie itself)
    movie_indices = [i[0] for i in sim_scores]
    return movies_df['original_title'].iloc[movie_indices].tolist()
    

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        movie_title = request.form.get('movie_title', '')
        print(f"Movie title (from form): {movie_title}")
        print(f"Form data: {request.form}")

        recommendations = get_recommendations(movie_title)
        print(f"Recommendations: {recommendations}")

        # Redirect to the recommendations page with the movie titles
        return redirect(url_for('recommendations', movie_title=movie_title))

    return render_template('index.html')


@app.route('/recommendations/<movie_title>', methods=['GET'])
def recommendations(movie_title):
    # Retrieve recommendations for the given movie_title
    recommendations = get_recommendations(movie_title)
    
    # Render the recommendations template
    return render_template('recommendations.html', movie_title=movie_title, recommendations=recommendations)


# New route for autocomplete suggestions
@app.route('/autocomplete', methods=['GET'])
def autocomplete():
    partial_title = request.args.get('partial_title')
    
    # Convert everything to lowercase for case-insensitive matching
    partial_title_lower = partial_title.lower()
    
    # Filter matching titles
    matching_titles = movies_df[movies_df['original_title'].str.lower().str.startswith(partial_title_lower)]['original_title'].tolist()
    
    # Return a JSON response even if there are no matching titles
    return jsonify(matching_titles)


if __name__ == '__main__':
    app.run(debug=True)


