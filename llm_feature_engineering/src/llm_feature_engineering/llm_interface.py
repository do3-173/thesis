"""
LLM API Interface Module

Provides a unified interface for interacting with different LLM providers
(Anthropic Claude, OpenAI GPT) for feature engineering tasks.
"""

import os
import time
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod

import anthropic
import openai
from dotenv import load_dotenv


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


def create_llm_interface(provider: str, model: Optional[str] = None, api_key: Optional[str] = None) -> LLMInterface:
    """
    Factory function to create LLM interface instances.
    
    Args:
        provider: LLM provider name ('anthropic' or 'openai')
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
    else:
        raise ValueError(f"Unsupported provider: {provider}. Supported providers: 'anthropic', 'openai'")