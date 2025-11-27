"""
LLM API Interface Module

Provides a unified interface for interacting with different LLM providers
(Anthropic Claude, OpenAI GPT) for feature engineering tasks.
"""

import os
import time
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod
from pathlib import Path

import anthropic
import openai
from dotenv import load_dotenv

# Load .env from project root (parent of llm_feature_engineering)
env_path = Path(__file__).parent.parent.parent.parent / '.env'
if env_path.exists():
    load_dotenv(env_path)
else:

    load_dotenv()


class LLMInterface(ABC):
    """
    Abstract base class for LLM API interfaces.
    
    Provides a common interface for different LLM providers while handling
    provider-specific implementation details.
    """
    
    def __init__(self, provider: str, model: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize LLM interface.
        
        Args:
            provider: LLM provider name ('anthropic' or 'openai')
            model: Model name (uses default if None)
            api_key: API key (loads from environment if None)
        """
        self.provider = provider
        self.model = model or self._get_default_model()
        self.api_key = api_key or self._load_api_key()
        self.client = self._initialize_client()
        
    @abstractmethod
    def _get_default_model(self) -> str:
        """Get default model for the provider."""
        pass
        
    @abstractmethod
    def _load_api_key(self) -> Optional[str]:
        """Load API key from environment."""
        pass
        
    @abstractmethod
    def _initialize_client(self):
        """Initialize the API client."""
        pass
        
    @abstractmethod
    def call_llm(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.1) -> Optional[str]:
        """
        Make API call to LLM.
        
        Args:
            prompt: The prompt to send to the LLM
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation (0.0 to 1.0)
            
        Returns:
            Generated text response or None if failed
        """
        pass
        
    def test_connection(self) -> bool:
        """
        Test if the LLM API connection is working.
        
        Returns:
            True if connection successful, False otherwise
        """
        test_prompt = "Hello! Please respond with 'Connection successful' if you can read this."
        response = self.call_llm(test_prompt, max_tokens=50)
        
        if response:
            print(f"{self.provider.title()} API connection successful")
            print(f"Model: {self.model}")
            print(f"Response: {response[:100]}...")
            return True
        else:
            print(f"{self.provider.title()} API connection failed")
            return False


class AnthropicInterface(LLMInterface):
    """Anthropic Claude API interface implementation."""
    
    def _get_default_model(self) -> str:
        """Get default Anthropic model."""
        return "claude-3-haiku-20240307"
    
    def _load_api_key(self) -> Optional[str]:
        """Load Anthropic API key from environment."""
        load_dotenv()
        return os.getenv('ANTHROPIC_API_KEY')
    
    def _initialize_client(self):
        """Initialize Anthropic client."""
        if not self.api_key:
            raise ValueError("Anthropic API key not found. Please set ANTHROPIC_API_KEY environment variable.")
        return anthropic.Anthropic(api_key=self.api_key)
    
    def call_llm(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.1) -> Optional[str]:
        """
        Make API call to Anthropic Claude.
        
        Args:
            prompt: The prompt to send to Claude
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation
            
        Returns:
            Generated text response or None if failed
        """
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            print(f"Anthropic API call failed: {e}")
            return None


class OpenAIInterface(LLMInterface):
    """OpenAI GPT API interface implementation."""
    
    def _get_default_model(self) -> str:
        """Get default OpenAI model."""
        return "gpt-3.5-turbo"
    
    def _load_api_key(self) -> Optional[str]:
        """Load OpenAI API key from environment."""
        load_dotenv()
        return os.getenv('OPENAI_API_KEY')
    
    def _initialize_client(self):
        """Initialize OpenAI client."""
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
        return openai.OpenAI(api_key=self.api_key)
    
    def call_llm(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.1) -> Optional[str]:
        """
        Make API call to OpenAI GPT.
        
        Args:
            prompt: The prompt to send to GPT
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation
            
        Returns:
            Generated text response or None if failed
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI API call failed: {e}")
            return None


class HuggingFaceInterface(LLMInterface):
    """Hugging Face Transformers interface implementation for local models."""

    def _get_default_model(self) -> str:
        """Get default Hugging Face model."""
        return "Qwen/Qwen2.5-7B-Instruct"

    def _load_api_key(self) -> Optional[str]:
        """Load Hugging Face token from environment (optional)."""
        load_dotenv()
        return os.getenv('HUGGINGFACE_TOKEN') or os.getenv('HF_TOKEN')

    def _initialize_client(self):
        """Initialize Hugging Face model and tokenizer."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
        except ImportError:
            raise ImportError("transformers and torch are required for HuggingFaceInterface. "
                              "Install with: pip install transformers torch")

        print(f"Loading local model: {self.model}...")
        
        # Determine device and dtype
        if torch.cuda.is_available():
            device = "cuda"
            # Use bfloat16 for better performance on modern GPUs
            dtype = torch.bfloat16
            print(f"CUDA available. GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            device = "cpu"
            dtype = torch.float32
        print(f"Using device: {device}, dtype: {dtype}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model, 
            token=self.api_key,
            trust_remote_code=True
        )
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model with appropriate settings for large models
        load_kwargs = {
            "token": self.api_key,
            "torch_dtype": dtype,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        
        # For large models (>10B params), use device_map="auto" for multi-GPU or efficient loading
        if any(size in self.model.lower() for size in ["70b", "72b", "65b", "34b", "32b"]):
            print(f"Large model detected, using device_map='auto' for efficient loading...")
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["device_map"] = device
        
        self.llm_model = AutoModelForCausalLM.from_pretrained(self.model, **load_kwargs)
        print(f"Model loaded successfully on {device}")
        
        return self.llm_model

    def call_llm(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.1) -> Optional[str]:
        """
        Make generation call to local Hugging Face model.
        """
        try:
            import torch
            
            # Check if chat template is available, otherwise fallback to raw prompt
            if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
                messages = [{"role": "user", "content": prompt}]
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                text = prompt

            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.llm_model.device)

            # Generation with proper settings
            with torch.no_grad():
                generated_ids = self.llm_model.generate(
                    model_inputs.input_ids,
                    attention_mask=model_inputs.attention_mask,
                    max_new_tokens=max_tokens,
                    temperature=max(temperature, 0.01),  # Avoid temperature=0 issues
                    do_sample=temperature > 0.01,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=False,  # Disable KV cache to avoid compatibility issues
                )
            
            # Extract only the generated tokens (not the input)
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return response.strip()
        except Exception as e:
            import traceback
            print(f"Hugging Face generation failed: {e}")
            traceback.print_exc()
            return None


def create_llm_interface(provider: str, model: Optional[str] = None, api_key: Optional[str] = None) -> LLMInterface:
    """
    Factory function to create LLM interface instances.
    
    Args:
        provider: LLM provider name ('anthropic', 'openai', 'huggingface')
        model: Model name (uses default if None)
        api_key: API key (loads from environment if None)
        
    Returns:
        LLM interface instance
        
    Raises:
        ValueError: If provider is not supported
    """
    if provider.lower() == "anthropic":
        return AnthropicInterface(provider, model, api_key)
    elif provider.lower() == "openai":
        return OpenAIInterface(provider, model, api_key)
    elif provider.lower() == "huggingface":
        return HuggingFaceInterface(provider, model, api_key)
    else:
        raise ValueError(f"Unsupported provider: {provider}. Supported providers: 'anthropic', 'openai', 'huggingface'")