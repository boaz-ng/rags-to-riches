from FlagEmbedding import FlagModel
import json
import os

def main():
    model = FlagModel('BAAI/bge-base-en-v1.5', 
                  query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ",
                  use_fp16=False)
    
    with open('documents.json', 'r') as in_f:
        docs = json.load(in_f)

    preprocessed = []
    if os.path.exists('preprocessed_documents.json') and os.path.getsize('preprocessed_documents.json') > 0:
        with open('preprocessed_documents.json', 'r') as out_f:
            preprocessed = json.load(out_f)

    start = len(preprocessed)
    for i in range(start, len(docs), 128):
        batch_docs = docs[i:i+128]
        batch = [doc['text'] for doc in batch_docs]
        embeddings = model.encode(batch, batch_size=1) # experiment to see what runs faster

        batch_data = [
            {
                'id': item['id'], 
                'text': item['text'], 
                'embedding': embedding.tolist()
            } 
            for item, embedding in zip(batch_docs, embeddings)
        ]

        preprocessed.extend(batch_data)
    
        with open('preprocessed_documents.json', 'w') as out_f:
            json.dump(preprocessed, out_f)

        print('Processed ', i + len(batch_docs), ' of ', len(docs), ' documents')


if __name__ == "__main__":
    main()