#################################
# Content Based Recommendation
#################################

#################################
# 1. Creating the TF-IDF Matrix
#################################

# import necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# set pandas display options
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)
# pd.set_option('display.expand_frame_repr', False)
# pd.set_option('display.float_format', lambda x: '%.5f' % x)

# read the data into a pandas dataframe
df = pd.read_csv("datasets/the_movies_dataset/movies_metadata.csv", low_memory=False)

# define function to check the dataframe
def check_df(dataframe, head=5):
    print('##################### Shape #####################')
    print(dataframe.shape)
    print('##################### Types #####################')
    print(dataframe.dtypes)
    print('##################### Head #####################')
    print(dataframe.head(head))
    print('##################### Tail #####################')
    print(dataframe.tail(head))
    print('##################### NA #####################')
    print(dataframe.isnull().sum())
    print('##################### Quantiles #####################')
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

# check the dataframe using the check_df function
check_df(df)

# select the 'overview' column from the dataframe and fill any missing values with an empty string
df['overview'].head()
df['overview'].isnull().sum()
df['overview'] = df['overview'].fillna('')

# create a TfidfVectorizer object and fit it to the 'overview' column
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['overview'])

# print the shape of the dataframe and the shape of the tfidf_matrix
df.shape # (45466, 24)
tfidf_matrix.shape # (45466, 75827)

# print the feature names and stop words used by the TfidfVectorizer
tfidf.get_feature_names_out()
tfidf.get_stop_words()

#################################
# 2. Creating the Cosine Similarity Matrix
#################################

# create the cosine similarity matrix from the tfidf_matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# print the shape of the cosine similarity matrix
cosine_sim.shape # (45466, 45466)

#################################
# 3. Making Recommendations Based on Similarities
#################################

# Define the indices using the df index and title columns
indices = pd.Series(df.index, index=df['title'])

# Count the value counts of the index
indices.index.value_counts()

# Remove any duplicated indices
indices = indices[~indices.index.duplicated(keep='first')]

# Define the movie index based on the Sherlock Holmes movie
movie_index = indices['Sherlock Holmes']

# Calculate similarity scores using cosine similarity
similarity_score = pd.DataFrame(cosine_sim[movie_index], columns=['score'])

# Sort the top 10 movie scores based on similarity scores
top_10_movie_score = similarity_score.sort_values('score', ascending=False)[1:11]

# Get the movie indices based on the top 10 movie scores
movie_indices = similarity_score.sort_values('score', ascending=False)[1:11].index

# Get the movie recommendations based on the movie indices
recommend_movies = df['title'].iloc[movie_indices]

# Create a data frame for the movie recommendations
recommend_movies = pd.DataFrame(recommend_movies)
top_10_movie_score = pd.DataFrame(top_10_movie_score)

# Reset the index of the movie recommendations and top 10 movie scores data frames
recommend_movies.reset_index(inplace=True)
top_10_movie_score.reset_index(inplace=True)

# Merge the movie recommendations and top 10 movie scores data frames
rec_df = pd.merge(recommend_movies, top_10_movie_score, how='left', on='index')

# Rename the columns of the merged data frame
rec_df.columns =['movie_id', 'movie_name', 'score']

#    movie_id                                         movie_name     score
# 0      3166                               They Might Be Giants  0.383196
# 1      4434                                     Without a Clue  0.334898
# 2      2301                              Young Sherlock Holmes  0.296491
# 3     24665   The Sign of Four: Sherlock Holmes' Greatest Case  0.296446
# 4     39647               Sherlock Holmes and the Leading Lady  0.284516
# 5     34750  The Adventures of Sherlock Holmes and Doctor W...  0.279187
# 6     17830                                    Sherlock Holmes  0.277366
# 7     26251                      The Hound of the Baskervilles  0.263890
# 8      9743                        The Seven-Per-Cent Solution  0.260478
# 9     14821                                  The Royal Scandal  0.252820

#################################
# 4. Preparing the Working Script
#################################

# Define a function for calculating cosine similarity scores based on the overview column
def calculate_cosine_sim(dataframe):
    dataframe['overview'] = dataframe['overview'].fillna('')
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(dataframe['overview'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

# Calculate cosine similarity scores based on the data frame df
cosine_sim = calculate_cosine_sim(df)

# Define a function for generating content-based recommendations
def content_based_recommender(title, cosine_sim, dataframe):
    indices = pd.Series(dataframe.index, index=dataframe['title'])
    indices = indices[~indices.index.duplicated(keep='first')]
    movie_index = indices[title]
    similarity_scores = pd.DataFrame(cosine_sim[movie_index], columns=["score"])
    movie_indices = similarity_scores.sort_values("score", ascending=False)[1:11].index
    return dataframe['title'].iloc[movie_indices]

# Generate content-based recommendations for the Sherlock Holmes movie
content_based_recommender("Sherlock Holmes", cosine_sim, df)

# 3166                                  They Might Be Giants
# 4434                                        Without a Clue
# 2301                                 Young Sherlock Holmes
# 24665     The Sign of Four: Sherlock Holmes' Greatest Case
# 39647                 Sherlock Holmes and the Leading Lady
# 34750    The Adventures of Sherlock Holmes and Doctor W...
# 17830                                      Sherlock Holmes
# 26251                        The Hound of the Baskervilles
# 9743                           The Seven-Per-Cent Solution
# 14821                                    The Royal Scandal

# Generate content-based recommendations for The Matrix movie
content_based_recommender("The Matrix", cosine_sim, df)

# 44161                        A Detective Story
# 44167                              Kid's Story
# 44163                             World Record
# 33854                                Algorithm
# 167                                    Hackers
# 20707    Underground: The Julian Assange Story
# 6515                                  Commando
# 24202                                 Who Am I
# 22085                           Berlin Express
# 9159                                  Takedown
