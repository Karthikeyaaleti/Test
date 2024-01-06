#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


data=pd.read_csv('movies.csv')


# In[3]:


print(data)


# In[4]:


data.shape


# In[5]:


data1=pd.read_csv('ratings.csv')
print(data1)


# In[6]:


data1.shape


# In[7]:


pd.unique(userId)


# In[8]:


DataFrame.unique(userId)


# In[9]:


data1.unique(userId)


# In[11]:


print(print(data1.groupby('userId').agg({"userId": "nunique"})))


# In[12]:


df = pd.read_csv('ratings.csv')


# In[13]:


print(df)


# In[14]:


print("Number of unique users:", df.userId.nunique())


# In[20]:


# Grouping ratings by movieId and counting the number of ratings
ratings_count = data1.groupby('movieId')['rating'].count()

# Finding the movie with the maximum number of ratings
max_ratings_movie_id = ratings_count.idxmax()

# Retrieving the movie title using the movieId
max_ratings_movie_title = data.loc[data['movieId'] == max_ratings_movie_id, 'title'].values[0]

print(f"The movie with the maximum number of ratings is '{max_ratings_movie_title}' with {ratings_count[max_ratings_movie_id]} rating.")


# In[21]:


# Step 1: Find the movieId for "The Matrix (1999)"
matrix_movie_id = data.loc[data['title'] == 'Matrix, The (1999)', 'movieId'].values[0]

# Step 2: Filter the tags DataFrame based on the movieId
matrix_tags = tags[tags['movieId'] == matrix_movie_id]

# Display the tags for "The Matrix (1999)"
print("Tags for 'The Matrix (1999)':")
print(matrix_tags[['userId', 'tag']])


# In[22]:


data2=pd.read_csv('tags.csv')
print(data2)


# In[23]:


# Step 1: Find the movieId for "The Matrix (1999)"
matrix_movie_id = data.loc[data['title'] == 'Matrix, The (1999)', 'movieId'].values[0]

# Step 2: Filter the tags DataFrame based on the movieId
matrix_tags = data2[data2['movieId'] == matrix_movie_id]

# Display the tags for "The Matrix (1999)"
print("Tags for 'The Matrix (1999)':")
print(matrix_tags[['userId', 'tag']])


# In[24]:


# Step 1: Find the movieId for "Terminator 2: Judgment Day (1991)"
terminator2_movie_id = data.loc[data['title'] == 'Terminator 2: Judgment Day (1991)', 'movieId'].values[0]

# Step 2: Filter the ratings DataFrame based on the movieId
terminator2_ratings = data1[data1['movieId'] == terminator2_movie_id]

# Step 3: Calculate the average rating
average_rating = terminator2_ratings['rating'].mean()

print(f"The average user rating for 'Terminator 2: Judgment Day (1991)' is {average_rating:.2f}.")


# In[25]:


import matplotlib.pyplot as plt

# Step 1: Find the movieId for "Fight Club (1999)"
fight_club_movie_id = data.loc[data['title'] == 'Fight Club (1999)', 'movieId'].values[0]

# Step 2: Filter the ratings DataFrame based on the movieId
fight_club_ratings = data1[data1['movieId'] == fight_club_movie_id]

# Step 3: Plot the histogram
plt.hist(fight_club_ratings['rating'], bins=5, edgecolor='black')
plt.title('User Ratings Distribution for "Fight Club (1999)"')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.show()


# In[26]:


# Calculate the average rating for each movie
average_ratings = data1.groupby('movieId')['rating'].mean()

# Identify the movie with the highest average rating
most_popular_movie_id = average_ratings.idxmax()

# Retrieve the movie title using the movieId
most_popular_movie_title = data.loc[data['movieId'] == most_popular_movie_id, 'title'].values[0]

print(f"The most popular movie based on average user ratings is '{most_popular_movie_title}' with an average rating of {average_ratings[most_popular_movie_id]:.2f}.")


# In[28]:


# Mandatory Operation 1: Group user ratings based on movieId and apply aggregation operations
grouped_ratings = data1.groupby('movieId').agg({'rating': ['count', 'mean']})

# Rename columns for clarity
grouped_ratings.columns = ['rating_count', 'rating_mean']

# Mandatory Operation 2: Apply inner join on dataframe created from movies.csv and the grouped df from step 1
merged_data = pd.merge(data, grouped_ratings, on='movieId', how='inner')

# Mandatory Operation 3: Filter only those movies which have more than 50 user ratings
filtered_data = merged_data[merged_data['rating_count'] > 50]

# Display the resulting DataFrame
print(filtered_data.head())


# In[29]:


# Calculate the average rating for each movie in the filtered data
average_ratings_filtered = filtered_data.groupby('movieId')['rating_mean'].mean()

# Identify the movie with the highest average rating
most_popular_filtered_movie_id = average_ratings_filtered.idxmax()

# Retrieve the movie title using the movieId
most_popular_filtered_movie_title = filtered_data.loc[filtered_data['movieId'] == most_popular_filtered_movie_id, 'title'].values[0]

print(f"The most popular movie based on average user ratings (with more than 50 ratings) is '{most_popular_filtered_movie_title}' with an average rating of {average_ratings_filtered[most_popular_filtered_movie_id]:.2f}.")


# In[30]:


# Select the top 5 popular movies based on number of user ratings
top_popular_movies_by_ratings = filtered_data.nlargest(5, 'rating_count')[['title', 'rating_count']]

# Display the result
print("Top 5 popular movies based on number of user ratings:")
print(top_popular_movies_by_ratings)


# In[31]:


# Filter the Sci-Fi movies from the filtered_data
sci_fi_movies = filtered_data[filtered_data['genres'].str.contains('Sci-Fi')]

# Select the third most popular Sci-Fi movie based on the number of user ratings
third_most_popular_sci_fi_movie = sci_fi_movies.nlargest(3, 'rating_count').iloc[-1]

# Display the result
print("Third most popular Sci-Fi movie based on the number of user ratings:")
print("Title:", third_most_popular_sci_fi_movie['title'])
print("Number of User Ratings:", third_most_popular_sci_fi_movie['rating_count'])



# In[32]:


import pandas as pd

links = pd.read_csv('links.csv')
filtered_data = pd.read_csv('filtered_data.csv')  # Assuming you have the filtered data stored in a CSV file

# Merge filtered_data with links to get IMDb IDs
merged_data = pd.merge(filtered_data, links, on='movieId', how='inner')


# In[33]:


# Assuming 'filtered_data' is your DataFrame
filtered_data.to_csv('filtered_data.csv', index=False)


# In[42]:


links = pd.read_csv('links.csv')
print(links)


# In[34]:


import pandas as pd

links = pd.read_csv('links.csv')
filtered_data = pd.read_csv('filtered_data.csv')  # Assuming you have the filtered data stored in a CSV file

# Merge filtered_data with links to get IMDb IDs
merged_data = pd.merge(filtered_data, links, on='movieId', how='inner')


# In[35]:


print(filtered_data)


# In[36]:


print(merged_data)


# In[37]:


import requests
from bs4 import BeautifulSoup

# Example URL (replace with the actual URL from your DataFrame)
url = 'https://www.imdb.com/title/tt0137523/'

response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# Extract reviews (modify as per IMDb HTML structure)
reviews = soup.find_all('div', class_='text show-more__control')
for review in reviews:
    print(review.get_text())


# In[38]:


# Assuming 'filtered_data' is your DataFrame
highest_imdb_rating_movie_id = filtered_data.loc[filtered_data['imdbRating'].idxmax(), 'movieId']

print(f"The movieId of the movie with the highest IMDb rating is: {highest_imdb_rating_movie_id}")


# In[39]:


base_url = 'https://www.imdb.com/title/'

# Construct IMDb URLs
merged_data['imdb_url'] = base_url + merged_data['imdbId'].astype(str) + '/'


# In[40]:


# Assuming 'filtered_data' is your DataFrame
highest_imdb_rating_movie_id = filtered_data.loc[filtered_data['imdbRating'].idxmax(), 'movieId']

print(f"The movieId of the movie with the highest IMDb rating is: {highest_imdb_rating_movie_id}")


# In[44]:


import requests
from bs4 import BeautifulSoup
import pandas as pd

# Assuming 'filtered_data' DataFrame contains movie information, including IMDb IDs
filtered_data = pd.read_csv('filtered_data.csv')

# Function to scrape reviews for a given IMDb ID
def scrape_reviews(imdb_id):
    base_url = 'https://www.imdb.com/title/'
    url = f'{base_url}{imdb_id}/reviews'

    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        reviews = soup.find_all('div', class_='text show-more__control')
        return [review.get_text() for review in reviews]
    else:
        print(f"Failed to fetch reviews for IMDb ID {imdb_id}")
        return []

# Add a new column to store reviews
filtered_data['reviews'] = filtered_data['imdbId'].apply(scrape_reviews)

# Save the DataFrame with reviews to a CSV file
filtered_data.to_csv('filtered_data_with_reviews.csv', index=False)


# In[45]:





# In[46]:


# Assuming 'filtered_data' is your DataFrame
highest_imdb_rating_movie_id = filtered_data.loc[filtered_data['imdbRating'].idxmax(), 'movieId']

print(f"The movieId of the movie with the highest IMDb rating is: {highest_imdb_rating_movie_id}")


# In[47]:


import requests
from bs4 import BeautifulSoup

# Example IMDb ID
imdb_id = 'tt0137523'

# Construct IMDb URL
url = f'https://www.imdb.com/title/{imdb_id}/reviews'

# Fetch the webpage
response = requests.get(url)
if response.status_code == 200:
    soup = BeautifulSoup(response.text, 'html.parser')
    reviews = soup.find_all('div', class_='text show-more__control')
    
    # Process reviews as needed
    for review in reviews:
        print(review.get_text())
else:
    print(f"Failed to fetch reviews for IMDb ID {imdb_id}")


# In[53]:


# Assuming 'filtered_data' is your DataFrame
highest_imdb_rating_movie_id = filtered_data.loc[filtered_data['rating_mean'].idxmax(), 'movieId']

print(f"The movieId of the movie with the highest IMDb rating is: {highest_imdb_rating_movie_id}")


# In[49]:


# Print column names
print(filtered_data.columns)


# In[50]:


# Check for case sensitivity
print('imdbRating' in filtered_data.columns)


# In[51]:


# Display the first few rows of the DataFrame
print(filtered_data.head())


# In[52]:


import pandas as pd

# Load 'filtered_data' DataFrame
filtered_data = pd.read_csv('filtered_data.csv')

# Load 'links.csv' DataFrame
links = pd.read_csv('links.csv')

# Merge 'filtered_data' with 'links' on 'movieId'
merged_data = pd.merge(filtered_data, links, on='movieId', how='inner')

# Assuming the 'imdbRating' column exists in 'links.csv', create a new 'imdbRating' column in 'filtered_data'
filtered_data['imdbRating'] = merged_data['imdbRating']

# Now, you can find the movie with the highest IMDb rating
highest_imdb_rating_movie_id = filtered_data.loc[filtered_data['imdbRating'].idxmax(), 'movieId']

print(f"The movieId of the movie with the highest IMDb rating is: {highest_imdb_rating_movie_id}")


# In[56]:


import requests
from bs4 import BeautifulSoup

def get_imdb_rating(movie_title):
    # Construct IMDb search URL
    search_url = f'https://www.imdb.com/find?q={movie_title}&s=tt&ttype=ft&ref_=fn_ft'

    # Fetch the IMDb search results page
    response = requests.get(search_url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract the IMDb ID (ttXXXXXXX) from the search results
    imdb_id = soup.find('td', class_='result_text').links['href'].split('/')[2]

    # Construct IMDb movie URL
    movie_url = f'https://www.imdb.com/title/{imdb_id}/'

    # Fetch the IMDb movie page
    response = requests.get(movie_url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract the IMDb rating
    imdb_rating = soup.find('span', itemprop='ratingValue').text

    return imdb_rating

# Example: Get IMDb rating for "The Shawshank Redemption"
movie_title = "The Shawshank Redemption"
imdb_rating = get_imdb_rating(movie_title)
print(f"IMDb rating for '{movie_title}': {imdb_rating}")


# In[57]:


import requests
from bs4 import BeautifulSoup

def get_imdb_rating(movie_title):
    # Construct IMDb search URL
    search_url = f'https://www.imdb.com/find?q={movie_title}&s=tt&ttype=ft&ref_=fn_ft'

    # Fetch the IMDb search results page
    response = requests.get(search_url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract the IMDb ID (ttXXXXXXX) from the search results
    imdb_id = soup.find('td', class_='result_text').a['href'].split('/')[2]

    # Construct IMDb movie URL
    movie_url = f'https://www.imdb.com/title/{imdb_id}/'

    # Fetch the IMDb movie page
    response = requests.get(movie_url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract the IMDb rating
    imdb_rating = soup.find('span', itemprop='ratingValue').text

    return imdb_rating

# Example: Get IMDb rating for "The Shawshank Redemption"
movie_title = "The Shawshank Redemption"
imdb_rating = get_imdb_rating(movie_title)
print(f"IMDb rating for '{movie_title}': {imdb_rating}")


# In[58]:


import requests
from bs4 import BeautifulSoup

def get_imdb_rating(movie_title):
    # Construct IMDb search URL
    search_url = f'https://www.imdb.com/find?q={movie_title}&s=tt&ttype=ft&ref_=fn_ft'

    # Fetch the IMDb search results page
    response = requests.get(search_url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find the first result text with a link
    result_text = soup.find('td', class_='result_text')

    # Check if result_text is not None before accessing its attributes
    if result_text is not None:
        # Extract the IMDb ID (ttXXXXXXX) from the search results
        imdb_id = result_text.a['href'].split('/')[2]

        # Construct IMDb movie URL
        movie_url = f'https://www.imdb.com/title/{imdb_id}/'

        # Fetch the IMDb movie page
        response = requests.get(movie_url)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract the IMDb rating
        imdb_rating_tag = soup.find('span', itemprop='ratingValue')

        # Check if imdb_rating_tag is not None before accessing its text
        if imdb_rating_tag is not None:
            imdb_rating = imdb_rating_tag.text
            return imdb_rating

    # Return None if any step fails
    return None

# Example: Get IMDb rating for "The Shawshank Redemption"
movie_title = "The Shawshank Redemption"
imdb_rating = get_imdb_rating(movie_title)

if imdb_rating is not None:
    print(f"IMDb rating for '{movie_title}': {imdb_rating}")
else:
    print(f"IMDb rating not found for '{movie_title}'.")



# In[74]:


import requests
import numpy as np
from bs4 import BeautifulSoup

def scrapper(imdbId):
    id = str(int(imdbId))
    n_zeroes = 7 - len(id)
    new_id = "0"*n_zeroes + id
    URL = f"https://www.imdb.com/title/tt{new_id}/"
    request_header = {'Content-Type': 'text/html; charset=UTF-8', 
                      'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0', 
                      'Accept-Encoding': 'gzip, deflate, br'}
    response = requests.FILL_IN_THE_BLANK(URL, headers=request_header)
    soup = FILL_IN_THE_BLANK(response.text)
    imdb_rating = soup.find('FILL_IN_THE_BLANK', attrs={'FILL_IN_THE_BLANK' : 'FILL_IN_THE_BLANK'})
    return imdb_rating.text if imdb_rating else np.nan


# In[60]:


pip install FILL_IN_THE_BLANK


# In[61]:


pip install bs4


# In[72]:


import requests
import numpy as np
from bs4 import BeautifulSoup

def scrapper(imdbId):
    id = str(int(imdbId))
    n_zeroes = 7 - len(id)
    new_id = "0"*n_zeroes + id
    URL = f"https://www.imdb.com/title/tt{new_id}/"
    request_header = {'Content-Type': 'text/html; charset=UTF-8', 
                      'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0', 
                      'Accept-Encoding': 'gzip, deflate, br'}
    response = requests.FILL_IN_THE_BLANK(URL, headers=request_header)
    soup = FILL_IN_THE_BLANK(response.text)
    imdb_rating = soup.find('FILL_IN_THE_BLANK', attrs={'FILL_IN_THE_BLANK' : 'FILL_IN_THE_BLANK'})
    return imdb_rating.text if imdb_rating else np.nan


# In[66]:


pip install beautifulsoup4


# In[67]:


pip install requests


# In[70]:


import requests
import numpy as np
from bs4 import BeautifulSoup

def scrapper(imdbId):
    id = str(int(imdbId))
    n_zeroes = 7 - len(id)
    new_id = "0" * n_zeroes + id
    URL = f"https://www.imdb.com/title/tt{new_id}/"
    request_header = {'Content-Type': 'text/html; charset=UTF-8',
                      'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0',
                      'Accept-Encoding': 'gzip, deflate, br'}
    response = requests.get(URL, headers=request_header)
    soup = BeautifulSoup(response.text, 'html.parser')
    imdb_rating = soup.find('span', attrs={'itemprop': 'ratingValue'})
    return imdb_rating.text if imdb_rating else np.nan


# In[71]:


highest_imdb_rating_movie_id = filtered_data.loc[filtered_data['imdbRating'].idxmax(), 'movieId']

print(f"The movieId of the movie with the highest IMDb rating is: {highest_imdb_rating_movie_id}")


# In[73]:


highest_imdb_rating_movie_id = filtered_data.loc[filtered_data['imdbRating'].idxmax(), 'movieId']

print(f"The movieId of the movie with the highest IMDb rating is: {highest_imdb_rating_movie_id}")


# In[75]:


pip install requests numpy beautifulsoup4


# In[76]:


import requests
import numpy as np
from bs4 import BeautifulSoup

def scrapper(imdbId):
    id = str(int(imdbId))
    n_zeroes = 7 - len(id)
    new_id = "0" * n_zeroes + id
    URL = f"https://www.imdb.com/title/tt{new_id}/"
    request_header = {
        'Content-Type': 'text/html; charset=UTF-8',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0',
        'Accept-Encoding': 'gzip, deflate, br'
    }
    response = requests.get(URL, headers=request_header)
    soup = BeautifulSoup(response.text, 'html.parser')
    imdb_rating = soup.find('span', {'itemprop': 'ratingValue'})
    return imdb_rating.text if imdb_rating else np.nan


# In[77]:


# Assuming 'filtered_data' is your DataFrame
highest_imdb_rating_movie_id = filtered_data.loc[filtered_data['imdbRating'].idxmax(), 'movieId']

print(f"The movieId of the movie with the highest IMDb rating is: {highest_imdb_rating_movie_id}")


# In[78]:


# Print column names
print(filtered_data.columns)


# In[79]:


from bs4 import BeautifulSoup
import requests
import re
import pandas as pd


# In[80]:


# Downloading imdb top 250 movie's data
url = 'http://www.imdb.com/chart/top'
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")


# In[81]:


movies = soup.select('td.titleColumn')
crew = [a.attrs.get('title') for a in soup.select('td.titleColumn a')]
ratings = [b.attrs.get('data-value')
		for b in soup.select('td.posterColumn span[name=ir]')]


# In[82]:


# create a empty list for storing
# movie information
list = []

# Iterating over movies to extract
# each movie's details
for index in range(0, len(movies)):
	
	# Separating movie into: 'place',
	# 'title', 'year'
	movie_string = movies[index].get_text()
	movie = (' '.join(movie_string.split()).replace('.', ''))
	movie_title = movie[len(str(index))+1:-7]
	year = re.search('\((.*?)\)', movie_string).group(1)
	place = movie[:len(str(index))-(len(movie))]
	data = {"place": place,
			"movie_title": movie_title,
			"rating": ratings[index],
			"year": year,
			"star_cast": crew[index],
			}
	list.append(data)


# In[83]:


for movie in list:
	print(movie['place'], '-', movie['movie_title'], '('+movie['year'] +
		') -', 'Starring:', movie['star_cast'], movie['rating'])


# In[84]:


from bs4 import BeautifulSoup
import requests
import re
import pandas as pd


# Downloading imdb top 250 movie's data
url = 'http://www.imdb.com/chart/top'
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")
movies = soup.select('td.titleColumn')
crew = [a.attrs.get('title') for a in soup.select('td.titleColumn a')]
ratings = [b.attrs.get('data-value')
		for b in soup.select('td.posterColumn span[name=ir]')]




# create a empty list for storing
# movie information
list = []

# Iterating over movies to extract
# each movie's details
for index in range(0, len(movies)):
	
	# Separating movie into: 'place',
	# 'title', 'year'
	movie_string = movies[index].get_text()
	movie = (' '.join(movie_string.split()).replace('.', ''))
	movie_title = movie[len(str(index))+1:-7]
	year = re.search('\((.*?)\)', movie_string).group(1)
	place = movie[:len(str(index))-(len(movie))]
	data = {"place": place,
			"movie_title": movie_title,
			"rating": ratings[index],
			"year": year,
			"star_cast": crew[index],
			}
	list.append(data)

# printing movie details with its rating.
for movie in list:
	print(movie['place'], '-', movie['movie_title'], '('+movie['year'] +
		') -', 'Starring:', movie['star_cast'], movie['rating'])


##.......##
df = pd.DataFrame(list)
df.to_csv('imdb_top_250_movies.csv',index=False)


# In[85]:


print(df)


# In[86]:


from urllib.request import urlopen as uReq
from bs4 import BeautifulSoup as soup
import pandas as pdp


# In[87]:


my_url = "http://www.imdb.com/search/title?sort=num_votes,desc&start=1&title_type=feature&year=1950,2012"


# In[90]:


filename= "lists.csv"
f= open(filename, "w")

headers= "Name, Year, Runtime \n"
f.write(headers)

for container in containers:
    name= container.img["alt"]
    year_mov= container.findAll("span", {"class": "lister-item-year"})
    year=year_mov[0].text
    runtime_mov= container.findAll("span", {"class": "runtime"})
    runtime=runtime_mov[0].text
    
    print(name + "," + year + "," + runtime +  "\n")
    f.write(name + "," + year + "," + runtime  + "\n")
    
f.close()


# In[91]:


uClient = uReq(my_url)
page_html = uClient.read()
uClient.close()


# In[92]:


import requests
import numpy as np
from bs4 import BeautifulSoup

def scrapper(imdbId):
    id = str(int(imdbId))
    n_zeroes = 7 - len(id)
    new_id = "0"*n_zeroes + id
    URL = f"https://www.imdb.com/title/tt{new_id}/"
    request_header = {'Content-Type': 'text/html; charset=UTF-8', 
                      'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0', 
                      'Accept-Encoding': 'gzip, deflate, br'}
    response = requests.FILL_IN_THE_BLANK(URL, headers=request_header)
    soup = FILL_IN_THE_BLANK(response.text)
    imdb_rating = soup.find('BeautifulSoup', attrs={'FILL_IN_THE_BLAN' : 'FILL_IN_THE_BLANK'})
    return imdb_rating.text if imdb_rating else np.nan


# In[93]:


highest_imdb_rating_movie_id = filtered_data.loc[filtered_data['imdb_rating'].idxmax(), 'movieId']

print(f"The movieId of the movie with the highest IMDb rating is: {highest_imdb_rating_movie_id}")


# In[94]:


import requests
import numpy as np
from bs4 import BeautifulSoup

def scrapper(imdbId):
    id = str(int(imdbId))
    n_zeroes = 7 - len(id)
    new_id = "0" * n_zeroes + id
    URL = f"https://www.imdb.com/title/tt{new_id}/"
    request_header = {
        'Content-Type': 'text/html; charset=UTF-8',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0',
        'Accept-Encoding': 'gzip, deflate, br'
    }
    response = requests.get(URL, headers=request_header)
    soup = BeautifulSoup(response.text, 'html.parser')
    imdb_rating = soup.find('span', {'itemprop': 'ratingValue'})
    return imdb_rating.text if imdb_rating else np.nan


# In[95]:


highest_imdb_rating_movie_id = filtered_data.loc[filtered_data['imdb_rating'].idxmax(), 'movieId']

print(f"The movieId of the movie with the highest IMDb rating is: {highest_imdb_rating_movie_id}")


# In[96]:


import requests
from bs4 import BeautifulSoup

def get_imdb_rating(movie_title):
    # Construct IMDb search URL
    search_url = f'https://www.imdb.com/find?q={movie_title}&s=tt&ttype=ft&ref_=fn_ft'

    # Fetch the IMDb search results page
    response = requests.get(search_url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract the IMDb ID (ttXXXXXXX) from the search results
    imdb_id = soup.find('td', class_='result_text').a['href'].split('/')[2]

    # Construct IMDb movie URL
    movie_url = f'https://www.imdb.com/title/{imdb_id}/'

    # Fetch the IMDb movie page
    response = requests.get(movie_url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract the IMDb rating
    imdb_rating = soup.find('span', itemprop='ratingValue')

    return imdb_rating.text if imdb_rating else None

# Example: Get IMDb rating for "The Shawshank Redemption"
movie_title = "The Shawshank Redemption"
imdb_rating = get_imdb_rating(movie_title)

if imdb_rating is not None:
    print(f"IMDb rating for '{movie_title}': {imdb_rating}")
else:
    print(f"IMDb rating not found for '{movie_title}'.")


# In[98]:


import requests
from bs4 import BeautifulSoup

def get_imdb_rating(movie_title):
    # Construct IMDb search URL
    search_url = f'https://www.imdb.com/find?q={movie_title}&s=tt&ttype=ft&ref_=fn_ft'

    # Fetch the IMDb search results page
    response = requests.get(search_url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find the first result text with a link
    result_text = soup.find('td', class_='result_text')

    # Check if result_text is not None before accessing its attributes
    if result_text is not None:
        # Extract the IMDb ID (ttXXXXXXX) from the search results
        imdb_id = result_text.find('a')['href'].split('/')[2]

        # Construct IMDb movie URL
        movie_url = f'https://www.imdb.com/title/{imdb_id}/'

        # Fetch the IMDb movie page
        response = requests.get(movie_url)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract the IMDb rating
        imdb_rating_tag = soup.find('span', itemprop='ratingValue')

        # Check if imdb_rating_tag is not None before accessing its text
        if imdb_rating_tag is not None:
            imdb_rating = imdb_rating_tag.text
            return imdb_rating

    # Return None if any step fails
    return None




# In[99]:


highest_imdb_rating_movie_id = filtered_data.loc[filtered_data['imdb_rating'].idxmax(), 'movieId']

print(f"The movieId of the movie with the highest IMDb rating is: {highest_imdb_rating_movie_id}")


# In[100]:


# Print column names
print(filtered_data.columns)


# In[101]:


print(data)


# In[103]:


# Check if the IMDb rating column is present
if 'imdb_rating' in filtered_data.columns:
    highest_imdb_rating_movie_id = filtered_data.loc[filtered_data['imdb_rating'].idxmax(), 'movieId']
    print(f"The movieId of the movie with the highest IMDb rating is: {highest_imdb_rating_movie_id}")
else:
    print("IMDb rating column not found in the DataFrame.")


# In[107]:


# Assuming 'filtered_data' is your DataFrame
sci_fi_movies = filtered_data[filtered_data['genres'].str.contains('Sci-Fi', case=False, na=False)]

if not sci_fi_movies.empty:
    highest_imdb_rating_sci_fi_movie_id = sci_fi_movies.loc[sci_fi_movies['rating_mean'].idxmax(), 'movieId']
    print(f"The movieId of the Sci-Fi movie with the highest IMDb rating is: {highest_imdb_rating_sci_fi_movie_id}")
else:
    print("No Sci-Fi movies found in the DataFrame.")


# In[ ]:




