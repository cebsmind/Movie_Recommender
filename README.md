# PROJECT : Movie Recommendation System
The goal of this project is to create a Movie Recommendation System using content-based filtering & build an app with **Flask**

# App preview
![alt text](https://github.com/cebsmind/Movie_Recommender/blob/main/images/MainPage.png?raw=true)

## How it works ? 
Content-based filtering uses item features to recommend other items similar to what the user likes, based on their previous actions or explicit feedback
![alt text](https://github.com/cebsmind/Movie_Recommender/blob/main/images/ContentBasedFiltering.png?raw=true)
# Data set Information :
### [Data Link from Kaggle](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)
About the data set : 
- **movies_metadata.csv**: The main Movies Metadata file. Contains information on 45,000 movies featured in the Full MovieLens dataset. Features include posters, backdrops, budget, revenue, release dates, languages, production countries and companies.
- **keywords.csv**: Contains the movie plot keywords for our MovieLens movies. Available in the form of a stringified JSON Object.
- **credits.csv:** Consists of Cast and Crew Information for all our movies. Available in the form of a stringified JSON Object.
# Code Walkthrough & Explanation
## 1. Variables 
In this project, I opted for a straightforward approach, focusing on fundamental movie details:

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

In the end, we get a numpy array of dimension (46628, 200) 
- **46628** is the number of movies (assuming that each movie has a corresponding "tags").
- **200** is the dimensionality of the word vectors in your pre-trained GloVe model.

### Add numeric variable
After we finally have vectorized our "tags" variable, we can add our numerical variables ('vote_average' and 'popularity'). But before we need to Normalize it : 
The features 'vote_average' and 'popularity' are being normalized using Min-Max scaling. Min-Max scaling transforms the values of a feature to a range between 0 and 1, based on the minimum and maximum values of that feature in the dataset. The MinMaxScaler is used for this purpose.
#### The final shape will be (46628, 202)

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
Once we finally have our vectorized data, we can finally recommend movies using **cosine similarity**
###  What is Cosine Similarity ?
Cosine similarity measures the cosine of the angle between two vectors. For word embeddings, the vectors represent the semantic meaning of words in a high-dimensional space. Cosine similarity ranges from -1 to 1, with 1 indicating identical vectors, 0 indicating no similarity, and -1 indicating opposite vectors.
![image](https://github.com/cebsmind/Movie_Recommender/assets/154905924/14fcec43-a4ef-4364-b235-bbaf1a32880a)

Our goal is to calculate the cosine similarity matrix of each movie.
### An example of how it will looks like 
![image](https://github.com/cebsmind/Movie_Recommender/assets/154905924/69a7eda3-0f5f-4048-87ae-937607e1dd66)

For each movie, we obtain the cosine similarity for every movie, and the more the score is close to 1, the more is similar. The movie itself will have a score of 1 so we need to recomment the top movies excluding himself using this matrix.

We can easily define a function that take 2 parameters : 
- the 'Title' of the movie
- number of movie to recommend

Where we calculate the cosine similarity of this particular movie, and find the top n movie similar using the cosine matrix.

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
