from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field


class QueryIntent(BaseModel):
    """Structured intent classification"""
    query_type: str = Field(description="Type of spending query: summary, category, time_pattern, merchant, comparison, anomaly, custom")
    confidence: float = Field(description="Confidence score 0-1")
    approach: str = Field(description="Processing approach: predefined or dynamic")
    parameters: Dict[str, Any] = Field(description="Extracted parameters from query")
    time_period: Optional[str] = Field(description="Time period if specified")
    specific_category: Optional[str] = Field(description="Specific category if mentioned")

class AnalysisResult(BaseModel):
    """Structured analysis result"""
    summary: str = Field(description="Brief summary of findings")
    key_metrics: Dict[str, Any] = Field(description="Key numerical findings")
    insights: List[str] = Field(description="Generated insights")
    recommendations: List[str] = Field(description="Actionable recommendations")
    data_points: int = Field(description="Number of transactions analyzed")