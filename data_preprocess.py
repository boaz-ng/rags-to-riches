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
    processed = 0
    if os.path.exists('preprocessed_documents.jsonl'):
        with open('preprocessed_documents.jsonl', 'r') as f:
            processed = sum(1 for _ in f) # counts lines
    print(f"Resuming from {processed}/{len(docs)}")

    with open('preprocessed_documents.jsonl', 'a') as out_f:
        for i in range(processed, len(docs), 1024):
            chunk = docs[i:i+1024]
            texts = [doc['text'] for doc in chunk]
            embeddings = model.encode(texts, batch_size=1) # experiment showed that this ran the fastest per document

            for document, embedding in zip(chunk, embeddings):
                json.dump({
                    'id': document['id'],
                    'text': document['text'],
                    'embedding': embedding.tolist()
                }, out_f)
                out_f.write('\n')

            out_f.flush()

            print('Processed ', i + len(chunk), ' of ', len(docs), ' documents')

    # once we finish loading everything, turn it into a .json (rather than jsonl)
    data = [json.loads(line) for line in open('preprocessed_documents.jsonl')]
    json.dump(data, open('preprocessed_documents.json', 'w'))

    # clean up temporary jsonl file
    os.remove('preprocessed_documents.jsonl')

if __name__ == "__main__":
    main()