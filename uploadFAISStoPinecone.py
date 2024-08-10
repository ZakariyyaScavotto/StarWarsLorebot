from langchain.vectorstores import FAISS
from pinecone import Pinecone, ServerlessSpec
import dotenv, os
from langchain.embeddings import HuggingFaceEmbeddings
import json

# Load your FAISS index
model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "cuda"}
embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
vectorstore = FAISS.load_local("FAISSvectorstore", embeddings)
index = vectorstore.index
print('Loaded FAISS index')

# Extract vectors
vectors = index.reconstruct_n(0, index.ntotal)
print('Extracted vectors')

# Generate unique IDs
ids = [str(i) for i in range(index.ntotal)]

# Prepare data for upserting
vectors = [{"id": id, "values": vector.tolist()} for id, vector in zip(ids, vectors)]
print('Prepared data for upserting')

# Initialize the Pinecone client
dotenv.load_dotenv()
pineconeAPI = os.getenv('PINECONE_API_KEY')
pc = Pinecone(api_key=pineconeAPI)
print('Initialized Pinecone client')

# Create an index if it doesn't exist
index_name = "sw-lorebot-vs"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=len(vectors[0]['values']),
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )
    print('Created index')

# Connect to the index
index = pc.Index(index_name)
print('Connected to index')

# Function to chunk your data within the 4MB limit
def chunks(data, max_bytes):
    current_chunk = []
    current_size = 0
    for record in data:
        record_size = len(json.dumps(record).encode('utf-8'))
        if current_size + record_size > max_bytes:
            yield current_chunk
            current_chunk = []
            current_size = 0
        current_chunk.append(record)
        current_size += record_size
    if current_chunk:
        yield current_chunk

# Upsert vectors in smaller chunks to avoid exceeding 4MB limit
max_batch_size = 4 * 1024 * 1024  # 4MB
for batch in chunks(vectors, max_batch_size):
    try:
        index.upsert(vectors=batch)
        print(f'Upserted batch with {len(batch)} vectors')
    except Exception as e:
        print(f'Error upserting batch: {e}')
        raise e