import sys
import json

from vector_db import VectorDatabase
from encode import QueryEncoder

class DocumentRetriever:
    def __init__(self, docs_file='data/raw/documents.json'):
        with open(docs_file, 'r') as in_f:
            self.docs = json.load(in_f)
        self.vector_db = VectorDatabase()
        self.query_encoder = QueryEncoder()
        
    def retrieve(self, query_text):
        query_embedding = self.query_encoder.encode(query_text)
        D, I = self.vector_db.search_by_vector(query_embedding, 3)
        return [self.docs[index]['text'] for index in I.flatten()]