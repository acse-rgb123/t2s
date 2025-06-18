"""
Conversation service for handling chat interactions using Gemini.
"""

from typing import Optional, List, Dict, Any
from ..shared.gemini_base import GeminiBase, GeminiConfig


class ConversationService:
    """Service for managing conversations and context with Gemini."""
    
    def __init__(self, model_name: str = "gemini-1.5-pro"):
        self.gemini = GeminiBase(model_name=model_name)
        self.system_prompt = """You are a helpful assistant for a text-to-SQL application. Your role is to:

1. Help users formulate their data questions clearly
2. Provide context and guidance for database queries
3. Explain SQL results in plain language
4. Maintain conversation context and history
5. Ask clarifying questions when needed

Keep responses conversational but informative. Always be helpful and patient with users who may not be familiar with databases or SQL."""
    
    def generate_response(
        self,
        user_message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.7
    ) -> str:
        """
        Generate a conversational response.
        
        Args:
            user_message: The user's current message
            conversation_history: Previous messages in the conversation
            temperature: Model temperature for creativity
            
        Returns:
            Generated response
        """
        # Build context from conversation history
        context = ""
        if conversation_history:
            for msg in conversation_history[-5:]:  # Last 5 messages for context
                role = msg.get("role", "user")
                content = msg.get("content", "")
                context += f"{role.capitalize()}: {content}\n"
        
        prompt = f"{context}User: {user_message}"
        
        config = GeminiConfig(temperature=temperature, max_output_tokens=1024)
        
        return self.gemini.generate(
            prompt=prompt,
            system_prompt=self.system_prompt,
            config=config
        )
    
    async def generate_response_async(
        self,
        user_message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.7
    ) -> str:
        """Async version of generate_response."""
        # Build context from conversation history
        context = ""
        if conversation_history:
            for msg in conversation_history[-5:]:  # Last 5 messages for context
                role = msg.get("role", "user")
                content = msg.get("content", "")
                context += f"{role.capitalize()}: {content}\n"
        
        prompt = f"{context}User: {user_message}"
        
        config = GeminiConfig(temperature=temperature, max_output_tokens=1024)
        
        return await self.gemini.generate_async(
            prompt=prompt,
            system_prompt=self.system_prompt,
            config=config
        )
    
    def stream_response(
        self,
        user_message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.7
    ):
        """
        Generate streaming response for real-time chat.
        
        Args:
            user_message: The user's current message
            conversation_history: Previous messages in the conversation
            temperature: Model temperature for creativity
            
        Yields:
            Streaming text chunks
        """
        # Build context from conversation history
        context = ""
        if conversation_history:
            for msg in conversation_history[-5:]:  # Last 5 messages for context
                role = msg.get("role", "user")
                content = msg.get("content", "")
                context += f"{role.capitalize()}: {content}\n"
        
        prompt = f"{context}User: {user_message}"
        
        config = GeminiConfig(temperature=temperature, max_output_tokens=1024)
        
        yield from self.gemini.generate_stream(
            prompt=prompt,
            system_prompt=self.system_prompt,
            config=config
        )