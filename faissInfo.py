import faiss

# Load the index from a file
index = faiss.read_index("FAISSvectorstore/index.faiss")

# Print the index type
print(f"FAISS Index Type: {type(index).__name__}") # IndexFlatL2
num_vectors = index.ntotal
print(f"Number of vectors in the FAISS index: {num_vectors}") # 952470