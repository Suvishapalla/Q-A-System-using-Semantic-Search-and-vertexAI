# ## Building a Q&A System using Semantic Search

import pandas as pd
import vertexai
import numpy as np
import pickle
import scann
import time 


from utils import create_index
from utils import authenticate
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances_argmin as distances_argmin
from utils import encode_text_to_embedding_batched
from vertexai.language_models import TextEmbeddingModel
from vertexai.language_models import TextGenerationModel
from IPython.display import Markdown, display

credentials, PROJECT_ID = authenticate()
REGION = 'us-central1'
vertexai.init(project=PROJECT_ID, location=REGION, credentials = credentials)
so_database = pd.read_csv('database_app.csv')
print("Shape: " + str(so_database.shape))
#print(so_database)

#import text embeddings model
embedding_model = TextEmbeddingModel.from_pretrained("textembedding-gecko@001")

#The data from stackoverflow in embdedded in the following way as batches as it cant take
# whole data at a time.
#so_questions = so_database.input_text.tolist()
# question_embeddings = encode_text_to_embedding_batched(sentences = so_questions,
#             api_calls_per_second = 20/60, batch_size = 5)

with open('question_embeddings_app.pkl', 'rb') as file:
    question_embeddings = pickle.load(file)
    print(question_embeddings)

so_database['embeddings'] = question_embeddings.tolist()

#Semantic Search : When someone asks a question, we can quickly include their question and 
#search through all the Stack Overflow questions to find the most similar one.
query = ['How to concat dataframes pandas']
query_embedding = embedding_model.get_embeddings(query)[0].values

#try to find the similar text using cosine similarity
cos_sim_array = cosine_similarity([query_embedding],list(so_database.embeddings.values))
cos_sim_array.shape

#Extracting the highest similarity query
index_doc_cosine = np.argmax(cos_sim_array)
index_doc_distances = distances_argmin([query_embedding], list(so_database.embeddings.values))[0]

so_database.input_text[index_doc_cosine]
so_database.output_text[index_doc_cosine]

# After finding a similar Stack Overflow question, we use a tool to make its answer 
# sound friendlier.
generation_model = TextGenerationModel.from_pretrained("text-bison@001")

context = "Question: " + so_database.input_text[index_doc_cosine] +\
"\n Answer: " + so_database.output_text[index_doc_cosine]

prompt = f"""Here is the context: {context} Using the relevant information from the context, provide an answer to the query: {query}."
             If the context doesn't provide any relevant information, answer with [I couldn't find a good match in the 
             document database for your query] """

# We import Ipython.display to get the graphical representation to compare the similarities
t_value = 0.2
response = generation_model.predict(prompt = prompt,temperature = t_value,
                                    max_output_tokens = 1024)

display(Markdown(response.text))
query = ['How to make the perfect lasagna']
query_embedding = embedding_model.get_embeddings(query)[0].values
cos_sim_array = cosine_similarity([query_embedding], 
                                  list(so_database.embeddings.values))
#cos_sim_array
index_doc = np.argmax(cos_sim_array)
context = so_database.input_text[index_doc] + "\n Answer: " + so_database.output_text[index_doc]

prompt = f"""Here is the context: {context} Using the relevant information from the context, provide an answer to the query: {query}."
             If the context doesn't provide any relevant information, answer with [I couldn't find a good match in the \
             document database for your query]  """

t_value = 0.2
response = generation_model.predict(prompt = prompt,temperature = t_value,
                                    max_output_tokens = 1024)
display(Markdown(response.text))

#Create index using scann
index = create_index(embedded_dataset = question_embeddings, num_leaves = 25, num_leaves_to_search = 10, training_sample_size = 2000)

query = "how to concat dataframes pandas"

# Here we used ScaNN algortithm which is more effetient vector similarity search and 
# compared it with normal search comparision.
start = time.time()
query_embedding = embedding_model.get_embeddings([query])[0].values
neighbors, distances = index.search(query_embedding, final_num_neighbors = 1)
end = time.time()

for id, dist in zip(neighbors, distances):
    print(f"[docid:{id}] [{dist}] -- {so_database.input_text[int(id)][:125]}...")

print("Latency (ms):", 1000 * (end - start))
start = time.time()
query_embedding = embedding_model.get_embeddings([query])[0].values
cos_sim_array = cosine_similarity([query_embedding], list(so_database.embeddings.values))
index_doc = np.argmax(cos_sim_array)
end = time.time()

print(f"[docid:{index_doc}] [{np.max(cos_sim_array)}] -- {so_database.input_text[int(index_doc)][:125]}...")

print("Latency (ms):", 1000 * (end - start))




