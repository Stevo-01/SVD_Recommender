#Importing Libraries
import numpy as np
import pandas as pd
import streamlit as st
from contextlib import contextmanager, redirect_stdout
from io import StringIO

st.write("""
# Group 1 Working Demo GUI For Movie Recomendation System

""")

#Reading dataset (MovieLens 1M movie ratings dataset: downloaded from https://grouplens.org/datasets/movielens/1m/)
# This for the ratin the data
data = pd.io.parsers.read_csv('data/ratings.dat',
    names=['user_id', 'movie_id', 'rating', 'time'],
    engine='python', delimiter='::')
movie_data = pd.io.parsers.read_csv('data/movies.dat',
    names=['movie_id', 'title', 'genre'],
    engine='python', delimiter='::')
# creating dataframe for the input features
def user_input():
    mo= int(st.number_input('Movie ID', min_value =0,value=10))
    rec = st.slider('Recomendation', 1, 100, 10)
    data = {'Movie ID': mo,
            'Recommendation Number': rec}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input()
#To show the dataset
st.write('movie table')
st.write(movie_data)


#Creating the rating matrix (rows as movies, columns as users)
ratings_mat = np.ndarray(
    shape=(np.max(data.movie_id.values), np.max(data.user_id.values)),
    dtype=np.uint8)
ratings_mat[data.movie_id.values-1, data.user_id.values-1] = data.rating.values

#Normalizing the matrix(subtract mean off)
normalised_mat = ratings_mat - np.asarray([(np.mean(ratings_mat, 1))]).T

#Computing the Singular Value Decomposition (SVD)
A = normalised_mat.T / np.sqrt(ratings_mat.shape[0] - 1)
U, S, V = np.linalg.svd(A)

#Function to calculate the cosine similarity (sorting by most similar and returning the top N)
def top_cosine_similarity(data, movie_id, top_n=10):
    index = movie_id - 1 # Movie id starts from 1 in the dataset
    movie_row = data[index, :]
    magnitude = np.sqrt(np.einsum('ij, ij -> i', data, data))
    similarity = np.dot(movie_row, data.T) / (magnitude[index] * magnitude)
    sort_indexes = np.argsort(-similarity)
    return sort_indexes[:top_n]

# Function to print top N similar movies
def print_similar_movies(movie_data, movie_id, top_indexes):
    print('Recommendations for {0}: \n'.format(
    movie_data[movie_data.movie_id == movie_id].title.values[0]))
    for id in top_indexes + 1:
        print(movie_data[movie_data.movie_id == id].title.values[0])

#k-principal components to represent movies, movie_id to find recommendations, top_n print n results
k = 50
movie_id =df['Movie ID'].iloc[0] # (getting an id from movies.dat)
top_n = df['Recommendation Number'].iloc[0]
sliced = V.T[:, :k] # representative data
indexes = top_cosine_similarity(sliced, movie_id, top_n)



#Printing the top N similar movies
@contextmanager
def st_capture(output_func):
    with StringIO() as stdout, redirect_stdout(stdout):
        old_write = stdout.write

        def new_write(string):
            ret = old_write(string)
            output_func(stdout.getvalue())
            return ret

        stdout.write = new_write
        yield

output = st.empty()
with st_capture(output.code):
    print_similar_movies(movie_data, movie_id, indexes)
