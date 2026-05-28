# RAG
# RAG approach 
# RAG (Retrieval-Augmented Generation)

# %%
# 3.3 Loading credentials in Python
from dotenv import load_dotenv
load_dotenv()

# To verify that a variable was loaded correctly:
import os
try:
    QDRANT_URL = os.environ["QDRANT_URL"]
    print("QDRANT_URL loaded successfully")
except KeyError:
    raise ValueError("QDRANT_URL is not set — check your .env file")

# %%
#  Exercise 1: Connections
# Question 0
# Load your secrets stored in the .env file into your environment.
import os
from dotenv import load_dotenv

load_dotenv()
# %%
# Question 1
# Create your llm.lab connection using openai client. Please name this client client_llmlab.
# You will need to use your creds os.environ["LLMLAB_URL"] and os.environ["LLMLAB_API_KEY"] (see the previous chapter).
# More information here about OpenAI API.
# (Go to next question to see the answer).

# %%
# Question 2
# Print all the available models (kindly made available to everyone by the SSPcloud team ❤️)
# You can notice the llm.lab platform provides both generation and embedding models.
from openai import OpenAI

client_llmlab = OpenAI(
    base_url=os.environ["LLMLAB_URL"],
    api_key=os.environ["LLMLAB_API_KEY"],
)

# Print models list
models = client_llmlab.models.list()
for model in models.data:
    print(f"ID: {model.id}")


# %%
# Question 3
# Create a connection to your Qdrant server. Please name it client_qdrant.
# Check all your existing collections (= databases). Probably not a single one for now.
from qdrant_client import QdrantClient

client_qdrant = QdrantClient(
    url=os.environ["QDRANT_URL"],
    api_key=os.environ["QDRANT_API_KEY"],
    port=os.environ["QDRANT_API_PORT"],
    check_compatibility=False
)

collections = client_qdrant.get_collections()
for collection in collections.collections:
    print(collection.name)

# %%
# para saber em que pasta estamos
os.getcwd()

# %%
