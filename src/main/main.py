import argparse
import transformers
from llama_cpp import Llama

def main():
    # parser = argparse.ArgumentParser(description="Video answering tool")
    
    # parser.add_argument(
    #     "--path",
    #     type=str,
    #     required=True,
    #     help="Path to the input video file"
    # )
    
    # args = parser.parse_args()
    
    llm = Llama(
        model_path="gemma-3-4b-it-qat-Q4_K_M.gguf",
        n_ctx=4096,
        n_threads=4,
        n_gpu_layers=-1
    )
    
    for out in llm(
        "Tell me a joke about cats.",
        max_tokens=100,
        stream=True
    ):
        print(out["choices"][0]["text"], end="", flush=True)

    
if __name__ == "__main__":
    main()