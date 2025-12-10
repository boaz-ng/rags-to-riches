from llama_cpp import Llama

class LLMGenerator:
    def __init__(self, model_path="./models/Llama-3.2-3B-Instruct-Q4_K_M.gguf"):
        self.llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=None, # use all available cpu threads
            n_gpu_layers=0,
            verbose=False
        )

    def generate(self, prompt, max_tokens=256):
        response = self.llm(
            prompt, 
            max_tokens=max_tokens,
            temperature=0.7,
        )
        return response['choices'][0]['text'].strip()

def main():
    chatgpt = LLMGenerator()
    print('making chatgpt')
    print(chatgpt.generate('how many kilograms does a cow weigh?'))

if __name__ == '__main__':
    main()