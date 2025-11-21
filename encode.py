from FlagEmbedding import Flagmodel
import numpy as np

class QueryEncoder:
    def __init__(self, model_path='BAAI/bge-base-en-v1.5'):
        self.model = FlagModel(model_path=model_path, 
                    query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ",
                    use_fp16=False)
        
    def encode(self, query_text):
        return model.encode(query_text, batch_size=1) # experiment showed that this ran the fastest per document

if __name__ == "__main__":
    qe = QueryEncoder()
    arr = qe.encode('We have been feeding our back yard squirrels for the fall and winter and we noticed that a few of them have missing fur. One has a patch missing down his back and under both arms. Also another has some missing on his whole chest. They are all eating and seem to have a good appetite.')
    print(arr[:5])