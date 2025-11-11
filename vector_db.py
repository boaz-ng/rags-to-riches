import faiss
import json

class VectorDatabase:
    def __init__(self, embeddings_path='preprocessed_documents.json'):
        with open(embeddings_path, 'r') as f:
            data = json.load(f)

        self.queries = [doc['embedding'] for doc in data] # do i need to check types here? use numpy?
        self.d = len(self.queries[0])
        self.index = faiss.IndexFlatL2(d)
        index.add(queries)

    # search the database for the k nearest neighbors to a particular id
    def search_by_id(self, id, k=3):
        """Search using existing document (given id) as the query"""
        query_embedding = queries[id].reshape(1, 768) # ensure shape (is this necessary?)
        return index.search(query_embedding, k)
        # return I # for now, return indices of nearest neighbors

    def search_by_vector(self, embedding, k=3):
        """embedding: array of shape (1, 768)"""
        return index.search(embedding, k)

if __name__ == "__main__":
    db = VectorDatabase()
    D, I = db.search_by_id(42, k=3)
    print("Nearest neighbor indices:", I)