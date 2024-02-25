# PROJECT : Movie Recommendation System
The goal of this project is to create a Movie Recommendation System using content-based filtering
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
For this project, I decied to go with a simple approach with basic informations from the movie :
- **id** : Unique ID of the movie
- **original_title**: Title
- **overview**: Synopsis of the movie
- **genres**: The genres types of the movie (can be multiple)
- **keywords**: Contains the movie plot keywords
- **cast**: Refers to the list of actors and actresses who play roles in the movie.
- **crew**: Refers to the team of individuals involved in the production of the movie, excluding the cast. This includes various roles such as the director, producer, cinematographer, editor, and other behind-the-scenes contributors. The crew is responsible for the technical and creative aspects of filmmaking.
- **vote_average**: Numerical variable (from 1 to 5)
- **popularity**: Popularity of the movie
### All theses informations will be used to build our model. But before that we need to treat all theses variables and pre-process it
## 2. Pre-processing
Pre processing is an important step to build an accurate model. 
In our case, it implies :
- Transform variables
- Clean text variables
- Impute missing values
- Use NLP techniques to prepare our text variables

 #### I decided to merge all ouf our clean texts variables into one, named "tag" 
 ```python
# Create a "tags" variable that combine every text variable we need
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
```
We will deal with both numerical variable after
## 3. Word Embedding
### What is word embedding ? 
- Word Embeddings in NLP is a technique where individual words are represented as real-valued vectors in a lower-dimensional space and captures inter-word semantics. Each word is represented by a real-valued vector with tens or hundreds of dimensions
![image](https://github.com/cebsmind/Movie_Recommender/assets/154905924/d9bf3625-149d-4139-81d1-e817bd568042)
For our case, we decided to use a pre-trained GloVe (Global Vectors for Word Representation) model. GloVe is an unsupervised learning algorithm that generates word embeddings based on the co-occurrence statistics of words in a large corpus. The model we loaded has 200-dimensional word vectors.

## What we did ? 
Let's break it down step by step:
### 1. Tokenization and Preprocessing:
Our text in the "tags" variable is tokenized, and common preprocessing steps like lowercasing and lemmatization are applied to each token.
### 2. Word Vector Lookup:
For each token, we checks if it is present in the pre-trained GloVe model. If the word is in the model, its corresponding word vector is retrieved, and this vector is added to a list..
### 3. Aggregation - Mean Calculation:
After collecting all the word vectors for the tokens present in the GloVe model, the code calculates the mean (average) of these vectors. This mean vector serves as a representation of the entire text in the "tags" variable.

In our specific case, we're going through each token (word) in the variable "tags." For each word, it looks up the corresponding word vector in the pre-trained GloVe model. These word vectors are then collected into a list. Finally, the mean (average) of all these word vectors is calculated, creating a single vector representation (embedding of 200 dimensions) that captures the semantic content of the entire text in the "tags" variable.
