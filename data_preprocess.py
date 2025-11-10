from FlagEmbedding import FlagModel
import json

def main():
    model = FlagModel('BAAI/bge-base-en-v1.5', 
                  query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ",
                  use_fp16=True)
    
    with open('documents.json', 'r') as in_f, open('preprocessed_documents.json', 'w') as out_f:
        data = json.load(in_f)
        sentences: list[str] = [item['text'] for item in data]
        embeddings = model.encode(sentences, batch_size=128) # experiment to see what runs faster
        data = [{'id': item['id'], 'text': item['text'], 'embedding': embedding.tolist()} for item, embedding in zip(data, embeddings)]
        json.dump(data, out_f)

if __name__ == "__main__":
    main()