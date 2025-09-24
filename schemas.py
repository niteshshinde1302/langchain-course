from typing import List
from pydantic import BaseModel, Field

class Source(BaseModel):
    """Schema for a source used by the agent"""
    url: str = Field(description="The URL of the source")

class AgentResponse(BaseModel):
    """Schema for the response of the agent with sources and answer"""
    
    sources: List[Source] = Field(default_factory=list, description="List of sources to generate the answer")
    answer: str = Field(description="The answer to the question")
    