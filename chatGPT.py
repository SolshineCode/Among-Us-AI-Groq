import openai
from utility import getGameData, in_meeting, get_chat_messages, clear_chat, translatePlayerColorID, allTasksDone, get_nearby_players, load_G, get_kill_list, get_num_alive_players
import time
import pyautogui
import networkx as nx
import re
import sys, os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/task-solvers")
from task_utility import get_dimensions, get_screen_coords, wake, get_screen_ratio
from groq import Groq
import requests
import json
from enum import Enum
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Simple LLM selection
Chosen_LLM = "openai"  # Change to "groq" or "ollama" for other LLM sources

class LLMProvider(Enum):
    OPENAI = "openai"
    GROQ = "groq"
    OLLAMA = "ollama"

@dataclass
class LLMConfig:
    provider: LLMProvider
    api_key: Optional[str] = None
    model_name: str = "gpt-3.5-turbo"
    temperature: float = 1.0
    max_tokens: Optional[int] = None
    top_p: float = 1.0
    api_base: str = "http://localhost:11434"

# Provider-specific configurations
LLM_CONFIGS = {
    "openai": {
        "model_name": "gpt-3.5-turbo",
        "temperature": 1.0,
        "max_tokens": None,
        "top_p": 1.0
    },
    "groq": {
        "model_name": "mixtral-8x7b-32768",
        "temperature": 0.7,
        "max_tokens": None,
        "top_p": 1.0
    },
    "ollama": {
        "model_name": "llama2",
        "temperature": 0.7,
        "max_tokens": None,
        "top_p": 1.0,
        "api_base": "http://localhost:11434"
    }
}

class LLMManager:
    def __init__(self):
        self.config = self._load_config()
        self._setup_client()
    
    def _load_config(self) -> LLMConfig:
        """Load LLM configuration based on Chosen_LLM"""
        provider = Chosen_LLM.lower()
        if provider not in LLM_CONFIGS:
            raise ValueError(f"Unsupported LLM provider: {provider}")
        
        # Get provider-specific config
        config_data = LLM_CONFIGS[provider].copy()
        
        # Load API key if needed
        api_key = None
        if provider in ["openai", "groq"]:
            try:
                with open("APIkey.txt") as f:
                    api_key = f.readline().rstrip()
            except FileNotFoundError:
                print(f"No API key found for {provider}")
                raise SystemExit(0)
        
        return LLMConfig(
            provider=LLMProvider(provider),
            api_key=api_key,
            **config_data
        )
    
    def _setup_client(self):
        if self.config.provider == LLMProvider.OPENAI:
            openai.api_key = self.config.api_key
        elif self.config.provider == LLMProvider.GROQ:
            self.client = Groq(api_key=self.config.api_key)
    
    def generate_response(self, messages: List[Dict[str, str]]) -> str:
        try:
            if self.config.provider == LLMProvider.OPENAI:
                response = openai.ChatCompletion.create(
                    model=self.config.model_name,
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    top_p=self.config.top_p
                )
                return response['choices'][0]['message']['content'].rstrip()
            
            elif self.config.provider == LLMProvider.GROQ:
                response = self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    top_p=self.config.top_p
                )
                return response.choices[0].message.content.rstrip()
            
            elif self.config.provider == LLMProvider.OLLAMA:
                response = requests.post(
                    f"{self.config.api_base}/api/chat",
                    json={
                        "model": self.config.model_name,
                        "messages": messages,
                        "options": {
                            "temperature": self.config.temperature,
                            "top_p": self.config.top_p
                        },
                        "stream": False
                    }
                )
                if response.status_code == 200:
                    return response.json()['message']['content'].rstrip()
                else:
                    print(f"Ollama API error: {response.text}")
                    return ""
            
        except Exception as e:
            print(f"Error generating response with {self.config.provider.value}: {str(e)}")
            return ""

# Initialize the LLM manager
llm_manager = LLMManager()

def ask_llm(prompts: List[Dict[str, str]]) -> str:
    """Drop-in replacement for ask_gpt that works with multiple providers"""
    print(f"Sending prompt to {llm_manager.config.provider.value}")
    try:
        response = llm_manager.generate_response(prompts)
        print("Returned message")
        return response
    except Exception as e:
        print(f"Error in LLM response: {str(e)}")
        return ""
