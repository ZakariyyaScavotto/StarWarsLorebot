import faiss

# Load the index from a file
index = faiss.read_index("FAISSvectorstoreIVFPQ/index.faiss")

# Print the index type
print(f"FAISS Index Type: {type(index).__name__}") # IndexIVFPQ, IndexFlatL2
num_vectors = index.ntotal
print(f"Number of vectors in the FAISS index: {num_vectors}") # 8108450, 952470
# Check nprobe value
print(f"nprobe value: {index.nprobe}") # 1, <didn't run>