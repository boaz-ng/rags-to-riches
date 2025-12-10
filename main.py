from retrieval import DocumentRetriever
from llm_generation import LLMGenerator

def main():
    # here, we use default values for both file locations. Maybe change it to explicitly include them in the future.
    retriever = DocumentRetriever()
    llm = LLMGenerator()
    print('======= Initialized LLM and Document Database =======')

    while True:
        prompt = input("How can I help you?\n")

        print('Getting relevant documents...')
        
        docs = retriever.retrieve(prompt)
        context = "\n\n".join(docs)

        print('Augmenting prompt...') 
        augmented_prompt = prompt + " Top documents:" + context
        print('Generating response...')
        response = llm.generate(augmented_prompt)

        print(response)

if __name__ == "__main__":
    main()