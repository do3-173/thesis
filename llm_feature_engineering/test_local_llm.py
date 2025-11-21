
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from llm_feature_engineering.llm_interface import create_llm_interface

def test_local_llm():
    print("Testing local LLM interface...")

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    
    try:
        llm = create_llm_interface(provider="huggingface", model=model_name)
        
        print(f"Model {model_name} loaded successfully.")
        
        prompt = "What is feature engineering in machine learning? Answer in one sentence."
        print(f"\nPrompt: {prompt}")
        
        response = llm.call_llm(prompt, max_tokens=50)
        
        print(f"\nResponse: {response}")
        
        if response:
            print("\nTest PASSED!")
        else:
            print("\nTest FAILED: No response generated.")
            
    except Exception as e:
        print(f"\nTest FAILED with error: {e}")

if __name__ == "__main__":
    test_local_llm()
