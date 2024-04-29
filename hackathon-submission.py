import pandas as pd
import numpy as np
from numpy import dot, linalg  # Import NumPy for cosine similarity

recipes = pd.DataFrame({
    'RecipeID': [1, 2, 3],
    'Name': ['Spaghetti Aglio Olio', 'Paneer Tikka', 'Caesar Salad'],
    'Ingredients': ['pasta, Onion, Tomato, parmesan cheese', 'Paneer, curry sauce', 'lettuce, croutons, parmesan cheese, Caesar dressing']
})

def preprocess_text(text):
  text = text.lower()  # Convert to lowercase
  text = text.strip()  # Remove leading/trailing whitespaces
  return text.split()  # Split into tokens (words)

recipes['Ingredients'] = recipes['Ingredients'].apply(preprocess_text)

# Create a word-document matrix (manually)
word_counts = {}
for ingredients in recipes['Ingredients']:
  for word in ingredients:
    if word in word_counts:
      word_counts[word] += 1
    else:
      word_counts[word] = 1

ingredient_matrix = pd.DataFrame.from_dict(word_counts, orient='index', columns=['Count'])
ingredient_matrix = ingredient_matrix.transpose().fillna(0)  # Create DataFrame and fill missing values with 0

def cosine_similarity(v1, v2):
  dot_product = dot(v1, v2)
  magnitude1 = linalg.norm(v1)
  magnitude2 = linalg.norm(v2)
  if magnitude1 > 0 and magnitude2 > 0:
    return dot_product / (magnitude1 * magnitude2)
  else:
    return 0 

# Calculate cosine similarity matrix
cosine_sim = np.zeros((len(ingredient_matrix), len(ingredient_matrix)))
for i in range(len(ingredient_matrix)):
  for j in range(i, len(ingredient_matrix)):
    cosine_sim[i, j] = cosine_similarity(ingredient_matrix.iloc[i], ingredient_matrix.iloc[j])
    cosine_sim[j, i] = cosine_sim[i, j]

def get_recommendations(recipe_name, cosine_sim=cosine_sim):
    idx = recipes[recipes['Name'] == recipe_name].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:4]
    recipe_indices = [i[0] for i in sim_scores]
    return recipes['Name'].iloc[recipe_indices]

print('######') 
print(get_recommendations('Spaghetti Aglio Olio'))
print(recipes)
print('&&&&')


