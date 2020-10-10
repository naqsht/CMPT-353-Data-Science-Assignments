import sys
import numpy as np
import pandas as pd
import difflib

# Command line arguments
movie_list = sys.argv[1]
movie_ratings = sys.argv[2]
output_file = sys.argv[3]

# Function to find similar strings
def title_match(word):
    return difflib.get_close_matches(word, movies, n=500)

# To read input in movie_list and converting into a list
movies = open(movie_list).readlines()
movies = list(map(lambda x: x.strip(), movies))

# Creating a DataFrame for movies
movies_df = pd.DataFrame(movies, columns=['title'])

# Reading movie_ratings into a variable
ratings = pd.read_csv(movie_ratings)

# To obtain title matches and further filtering of movie titles
ratings['title'] = ratings['title'].apply(lambda title: title_match(title))

# To omit brackets in movie title
ratings['title'] = ratings.title.apply(''.join)

# To omit empty strings from movie title
ratings = ratings[ratings.title != '']

# To reset indexes
ratings = ratings.reset_index(drop=True)
ratings = ratings.groupby('title', 0).mean().reset_index()

# Building the output file
output = movies_df.merge(ratings, on='title')

# Rounding ratings to two decimal places
output['rating'] = output['rating'].round(2)


# Output CSV file
output.to_csv(output_file)

