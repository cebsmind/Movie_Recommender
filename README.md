# PROJECT : Movie Recommendation System
The goal of this project is to create a Movie Recommendation System using content-based filtering & build an app with **Flask**

## How it works ? 
Content-based filtering uses item features to recommend other items similar to what the user likes, based on their previous actions or explicit feedback
![alt text](https://github.com/cebsmind/Movie_Recommender/blob/main/images/ContentBasedFiltering.png?raw=true)

# App preview
![alt text](https://github.com/cebsmind/Movie_Recommender/blob/main/images/MainPage.png?raw=true)

# Data set Information :
### [Data Link from Kaggle](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)
About the data set : 
- **movies_metadata.csv**: The main Movies Metadata file. Contains information on 45,000 movies featured in the Full MovieLens dataset. Features include posters, backdrops, budget, revenue, release dates, languages, production countries and companies.
- **keywords.csv**: Contains the movie plot keywords for our MovieLens movies. Available in the form of a stringified JSON Object.
- **credits.csv:** Consists of Cast and Crew Information for all our movies. Available in the form of a stringified JSON Object.
# Code Walkthrough & Explanation
## 1. Variables 
In this project, I opted for a straightforward approach, focusing on fundamental movie details:

# Deploy App
To use my app, you need to
- Download the Kaggle Data set
- Run my Notebooks to get the 

- **id** : Unique ID of the movie
- **original_title**: Title
- **overview**: Synopsis of the movie
- **genres**: The genres types of the movie (can be multiple)
- **keywords**: Contains the movie plot keywords
- **cast**: Refers to the list of actors and actresses who play roles in the movie.
- **crew**: Refers to the team of individuals involved in the production of the movie, excluding the cast. This includes various roles such as the director, producer, cinematographer, editor, and other behind-the-scenes contributors. The crew is responsible for the technical and creative aspects of filmmaking.
- **vote_average**: Numerical variable (from 1 to 5)
- **popularity**: Popularity of the movie
### Utilization in Model Building
All these details are integral for constructing our model. However, before proceeding with model development, it's essential to preprocess and treat these variables appropriately.
## 2. Data Pre-processing
Effective model building relies on thorough pre-processing to enhance accuracy. In our context, this involves:
In our case, it implies :
- **Transforming Variables:** Adjusting variables to better suit model requirements.
- **Cleaning Text Variables:** Ensuring text data is free from noise and inconsistencies.
- **Imputing Missing Values:** Addressing any gaps in the dataset.
- **Utilizing NLP Techniques:** Employing Natural Language Processing techniques to prepare text data.

 ### Combining Text Variables
To streamline our analysis, I've chosen to merge all cleaned text variables into a unified one called "tags." This consolidation is performed as follows:

 ```python
# Create a "tags" variable that combine every text variable we need
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
```
Handling numerical variables will be addressed subsequently in our process.

## 3. Word Embedding
### What is word embedding ? 
- Word Embeddings in NLP is a technique where individual words are represented as real-valued vectors in a lower-dimensional space and captures inter-word semantics. Each word is represented by a real-valued vector with tens or hundreds of dimensions
![image](https://github.com/cebsmind/Movie_Recommender/assets/154905924/d9bf3625-149d-4139-81d1-e817bd568042)

For our case, we decided to use a pre-trained GloVe (Global Vectors for Word Representation) model. GloVe is an unsupervised learning algorithm that generates word embeddings based on the co-occurrence statistics of words in a large corpus. The model we loaded has 200-dimensional word vectors.

## Let's break it down step by step:
### 1. Tokenization and Preprocessing:
Our text in the "tags" variable is tokenized, and common preprocessing steps like lowercasing and lemmatization are applied to each token.
### 2. Word Vector Lookup:
For each token, we checks if it is present in the pre-trained GloVe model. If the word is in the model, its corresponding word vector is retrieved, and this vector is added to a list..
### 3. Aggregation - Mean Calculation:
After collecting all the word vectors for the tokens present in the GloVe model, the code calculates the mean (average) of these vectors. This mean vector serves as a representation of the entire text in the "tags" variable.

Upon completion, we obtain a numpy array with dimensions (46628, 200):

- **46628:** Represents the number of movies, assuming each movie has a corresponding "tags" entry.
- **200:** Signifies the dimensionality of the word vectors in our pre-trained GloVe model.
### Incorporating Numeric Variables
After vectorizing our "tags" variable, we proceed to augment our feature set by introducing numeric variables (**'vote_average'** and **'popularity'**). However, an essential preprocessing step precedes this:

### Normalization
The features 'vote_average' and 'popularity' undergo normalization using Min-Max scaling. This transformation adjusts the values of a feature to a standardized range between 0 and 1, determined by the minimum and maximum values within the dataset. The MinMaxScaler is employed for this purpose.

Upon completion of this process, our feature matrix attains a final shape of (46628, 202). This combined feature set incorporates both the vectorized "tags" variable and the normalized numeric variables, creating a comprehensive representation for each movie in the dataset.


## 2.B Basic example to understand Word Embedding using Glove Pre-trained model : 
Let's go through a simple example to illustrate how the GloVe model works. Suppose we have the following preprocessed and tokenized text:

```python
tokens = ["movie", "action", "thrilling", "plot", "characters"]
```

Now, let's assume we have a simplified GloVe model where word vectors are 3-dimensional (for simplicity):

```python
glove_model = {
    "movie": [0.1, 0.2, 0.3],
    "action": [0.4, 0.5, 0.6],
    "thrilling": [0.7, 0.8, 0.9],
    "plot": [0.2, 0.3, 0.4],
    "characters": [0.5, 0.6, 0.7]
}
```
In a real-world scenario, the vectors would be much larger (in our case we have 200 dimensions), but we are simplifying for this example.

Now, let's apply the GloVe model to our tokens list:
```python
word_vectors = [glove_model[word] for word in tokens if word in glove_model]
```
For our example, this would result in:

```python
word_vectors = [
    [0.1, 0.2, 0.3],  # "movie"
    [0.4, 0.5, 0.6],  # "action"
    [0.7, 0.8, 0.9],  # "thrilling"
    [0.2, 0.3, 0.4],  # "plot"
    [0.5, 0.6, 0.7]   # "characters"
]
```
Now, let's calculate the mean of these vectors:
```python
doc_vector = np.mean(word_vectors, axis=0)
```
In our simplified example, the mean vector would be:

```python
doc_vector = [0.38, 0.48, 0.58]
```

## 4. Recommend Movies
After successfully vectorizing our data, the next exciting step is movie recommendations, facilitated by **cosine similarity**.

###  Understanding Cosine Similarity
Cosine similarity quantifies the cosine of the angle between two vectors. In the realm of word embeddings, these vectors encapsulate the semantic meaning of words in a multi-dimensional space. The metric ranges from -1 to 1, with 1 denoting identical vectors, 0 implying no similarity, and -1 indicating opposite vectors.
![image](https://github.com/cebsmind/Movie_Recommender/assets/154905924/14fcec43-a4ef-4364-b235-bbaf1a32880a)

Our objective is to compute the cosine similarity matrix for each movie.

### Visualizing the Cosine Similarity Matrix
![image](https://github.com/cebsmind/Movie_Recommender/assets/154905924/69a7eda3-0f5f-4048-87ae-937607e1dd66)

For every movie, we calculate the cosine similarity with every other movie. A score closer to 1 indicates higher similarity. The movie itself will have a score of 1, necessitating the exclusion of the movie when recommending similar ones using this matrix.

To streamline the process, a function is defined with two parameters:
- The 'Title' of the movie
- The number of movies to recommend.

Example usage
```python
# Example: Get top 10 movie recommendations for a movie titled 'Spider-Man'
recommend_top_movies('Spider-Man', top_n=10)
```

We get :
```python
['Spider-Man 3',
 'Doctor Strange',
 'Spider-Man 2',
 'X-Men: The Last Stand',
 'Superman/Batman: Apocalypse',
 'Batman Returns',
 'Look, Up in the Sky: The Amazing Story of Superman',
 'The Amazing Spider-Man',
 'Iron Man',
 'Guardians of the Galaxy Vol. 2']
```
This function efficiently calculates the cosine similarity of a specified movie and suggests the top N similar movies based on the cosine matrix.

# Set up APP

### To run the app you need 
- Download Kaggle Data Set
- Run my Notebooks to get the `movie_cosine_similarity_model.pkl`
- Set up folders : Below is the suggested folder structure for organizing your Flask app

```plaintext
flask-app/
│
├── data/
│   └── preprocessed_data.csv
│
├── models/
│   └── movie_cosine_similarity_model.pkl
│
├── static/
│   ├── css/
│   │   ├── rain.css
│   │   ├── recommendations.css
│   │   └── style.css
│   │
│   └── js/
│       ├── AutoComplete.js
│       └── rain.js
│
├── templates/
│      ├── index.html
│      └── recommendations.html
│
├── app.py
├── requirements.txt
├── .gitignore.txt
├── .gcloudignore
└── README.md
