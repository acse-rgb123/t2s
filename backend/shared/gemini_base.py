"""
Base Gemini model class for easy integration across the application.
Provides a simple interface for calling Gemini models with system prompts and configuration.
"""

import os
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import google.generativeai as genai
from google.generativeai.types import GenerateContentResponse


@dataclass
class GeminiConfig:
    """Configuration class for Gemini model settings."""
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    max_output_tokens: int = 2048
    stop_sequences: Optional[List[str]] = None


class GeminiBase:
    """
    Base class for Gemini model interactions.
    
    Usage:
        gemini = GeminiBase(model_name="gemini-1.5-pro")
        response = gemini.generate(
            system_prompt="You are a helpful SQL assistant",
            prompt="Convert this to SQL: Show me all users",
            config=GeminiConfig(temperature=0.3)
        )
    """
    
    def __init__(
        self, 
        model_name: str = "gemini-1.5-pro",
        api_key: Optional[str] = None
    ):
        """
        Initialize the Gemini model.
        
        Args:
            model_name: The Gemini model to use (default: gemini-1.5-pro)
            api_key: Google API key (if not provided, will use GOOGLE_API_KEY env var)
        """
        self.model_name = model_name
        
        # Configure API key
        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Google API key must be provided either as parameter or GOOGLE_API_KEY environment variable")
        
        genai.configure(api_key=api_key)
        
        # Initialize the model
        self.model = genai.GenerativeModel(model_name)
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        config: Optional[GeminiConfig] = None
    ) -> str:
        """
        Generate content using the Gemini model.
        
        Args:
            prompt: The user prompt/question
            system_prompt: Optional system prompt to set context/behavior
            config: Optional configuration for generation parameters
            
        Returns:
            Generated text response
            
        Raises:
            Exception: If the API call fails
        """
        try:
            # Use default config if none provided
            if config is None:
                config = GeminiConfig()
            
            # Prepare generation config
            generation_config = genai.types.GenerationConfig(
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                max_output_tokens=config.max_output_tokens,
                stop_sequences=config.stop_sequences
            )
            
            # Combine system prompt and user prompt
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            
            # Generate content
            response = self.model.generate_content(
                full_prompt,
                generation_config=generation_config
            )
            
            return response.text
            
        except Exception as e:
            raise Exception(f"Gemini API call failed: {str(e)}")
    
    async def generate_async(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        config: Optional[GeminiConfig] = None
    ) -> str:
        """
        Async version of generate method.
        
        Args:
            prompt: The user prompt/question
            system_prompt: Optional system prompt to set context/behavior
            config: Optional configuration for generation parameters
            
        Returns:
            Generated text response
        """
        try:
            # Use default config if none provided
            if config is None:
                config = GeminiConfig()
            
            # Prepare generation config
            generation_config = genai.types.GenerationConfig(
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                max_output_tokens=config.max_output_tokens,
                stop_sequences=config.stop_sequences
            )
            
            # Combine system prompt and user prompt
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            
            # Generate content asynchronously
            response = await self.model.generate_content_async(
                full_prompt,
                generation_config=generation_config
            )
            
            return response.text
            
        except Exception as e:
            raise Exception(f"Gemini API call failed: {str(e)}")
    
    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        config: Optional[GeminiConfig] = None
    ):
        """
        Generate content with streaming response.
        
        Args:
            prompt: The user prompt/question
            system_prompt: Optional system prompt to set context/behavior
            config: Optional configuration for generation parameters
            
        Yields:
            Streaming text chunks
        """
        try:
            # Use default config if none provided
            if config is None:
                config = GeminiConfig()
            
            # Prepare generation config
            generation_config = genai.types.GenerationConfig(
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                max_output_tokens=config.max_output_tokens,
                stop_sequences=config.stop_sequences
            )
            
            # Combine system prompt and user prompt
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            
            # Generate content with streaming
            response = self.model.generate_content(
                full_prompt,
                generation_config=generation_config,
                stream=True
            )
            
            for chunk in response:
                if chunk.text:
                    yield chunk.text
                    
        except Exception as e:
            raise Exception(f"Gemini streaming API call failed: {str(e)}")


# Convenience function for quick usage
def quick_gemini_call(
    prompt: str,
    system_prompt: Optional[str] = None,
    model_name: str = "gemini-1.5-pro",
    temperature: float = 0.7
) -> str:
    """
    Quick function for simple Gemini calls without class instantiation.
    
    Args:
        prompt: The user prompt/question
        system_prompt: Optional system prompt
        model_name: Gemini model to use
        temperature: Generation temperature
        
    Returns:
        Generated text response
    """
    gemini = GeminiBase(model_name=model_name)
    config = GeminiConfig(temperature=temperature)
    return gemini.generate(prompt, system_prompt, config)