import faiss
import json
import os
import numpy as np

class VectorDatabase:
    def __init__(self, embeddings_path='data/preprocessed/preprocessed_documents.json', index_path='data/preprocessed/database.index'):
        self.embeddings_path = embeddings_path
        self.index_path = index_path

        with open(self.embeddings_path, "r") as f:
            data = json.load(f)

        self.embeddings = np.array(
            [doc["embedding"] for doc in data], dtype="float32"
        )

        self.dim = self.embeddings.shape[1]
        self.index = self.load_index() 

    def load_index(self):
        if os.path.exists(self.index_path):
            return faiss.read_index(self.index_path)

        index = faiss.IndexFlatL2(self.dim)
        index.add(self.embeddings)
        faiss.write_index(index, self.index_path)
        return index

    # search the database for the k nearest neighbors to a particular id
    def search_by_id(self, id, k=3):
        """Search using existing document (given id) as the query"""
        query_embedding = self.embeddings[id].reshape(1, self.dim)
        D, I = self.index.search(query_embedding, k)
        return D, I

    def search_by_vector(self, embedding, k=3):
        """embedding: array of shape (1, 768)"""
        embedding = np.array(embedding, dtype="float32").reshape(1, self.dim)
        D, I = self.index.search(embedding, k)
        return D, I

if __name__ == "__main__":
    # for now, we hard code this
    db = VectorDatabase()
    D, I = db.search_by_id(42, k=3)
    print("Nearest neighbor indices:", I)
    print("Distances:", D)