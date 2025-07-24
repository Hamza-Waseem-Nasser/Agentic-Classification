"""
STEP 2: STATE MANAGEMENT SYSTEM - TICKET STATE MODELS

This file defines the core data models for our ITSM ticket classification system.
It represents the state that flows through our multi-agent pipeline, tracking
the ticket's journey from raw text to final classification.

KEY CONCEPTS:
1. TicketState: The main state object that holds all ticket information
2. ProcessingStatus: Tracks which agents have processed the ticket
3. ValidationResult: Stores validation outcomes from each step
4. Pydantic Models: Provides type safety and automatic validation

DESIGN DECISIONS:
- Mutable state pattern: Agents modify the same state object
- Validation after every agent: Ensures data integrity
- Rich metadata tracking: Supports debugging and analytics
- Arabic text support: Handles UTF-8 encoding properly

This state flows through our 6-agent pipeline:
Orchestrator → Arabic Processing → Category Classifier → Subcategory Classifier → Validation → Learning
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum
from pydantic import BaseModel, Field, validator
import uuid


class AgentType(str, Enum):
    """Enumeration of all agents in our pipeline"""
    ORCHESTRATOR = "orchestrator"
    ARABIC_PROCESSOR = "arabic_processor"
    CATEGORY_CLASSIFIER = "category_classifier"
    SUBCATEGORY_CLASSIFIER = "subcategory_classifier"
    VALIDATION_AGENT = "validation_agent"
    LEARNING_AGENT = "learning_agent"


class ProcessingStatus(str, Enum):
    """Status of ticket processing at each stage"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ValidationResult(BaseModel):
    """Result of validation performed by an agent"""
    is_valid: bool = Field(description="Whether the validation passed")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score (0-1)")
    issues: List[str] = Field(default_factory=list, description="List of validation issues found")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations for improvement")
    
    class Config:
        """Pydantic configuration"""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class AgentProcessingInfo(BaseModel):
    """Information about processing by a specific agent"""
    agent_type: AgentType
    status: ProcessingStatus = ProcessingStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    processing_time_ms: Optional[int] = None
    validation_result: Optional[ValidationResult] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None
    
    @validator('completed_at')
    def validate_completion_time(cls, v, values):
        """Ensure completion time is after start time"""
        if v and values.get('started_at') and v < values['started_at']:
            raise ValueError("Completion time cannot be before start time")
        return v


class TicketClassification(BaseModel):
    """Classification results for the ticket"""
    main_category: Optional[str] = Field(None, description="Primary category (SubCategory)")
    main_category_description: Optional[str] = Field(None, description="Description of main category")
    subcategory: Optional[str] = Field(None, description="Secondary category (SubCategory2)")
    subcategory_description: Optional[str] = Field(None, description="Description of subcategory")
    confidence_score: float = Field(0.0, ge=0.0, le=1.0, description="Overall classification confidence")
    alternative_classifications: List[Dict[str, Any]] = Field(default_factory=list, description="Alternative possible classifications")


class TicketState(BaseModel):
    """
    Main state object that flows through our multi-agent pipeline.
    
    This represents a single ticket's complete processing state, including:
    - Original ticket data
    - Processing status for each agent
    - Classification results
    - Validation outcomes
    - Performance metrics
    """
    
    # Unique identifier
    ticket_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique ticket identifier")
    
    # Original ticket data
    original_text: str = Field(description="Original ticket text in Arabic")
    customer_id: Optional[str] = Field(None, description="Customer identifier")
    priority: Optional[str] = Field(None, description="Ticket priority")
    created_at: datetime = Field(default_factory=datetime.now, description="Ticket creation timestamp")
    
    # Processed text (after Arabic processing)
    processed_text: Optional[str] = Field(None, description="Text after Arabic processing/normalization")
    extracted_keywords: List[str] = Field(default_factory=list, description="Keywords extracted from text")
    language_confidence: float = Field(0.0, ge=0.0, le=1.0, description="Confidence that text is Arabic")
    
    # Arabic processing results
    entities: List[str] = Field(default_factory=list, description="Entities extracted from Arabic text")
    technical_terms: List[str] = Field(default_factory=list, description="Technical terms identified")
    
    # Processing metadata
    processing_metadata: Dict[str, Any] = Field(default_factory=dict, description="Processing metadata from agents")
    processing_started: Optional[datetime] = Field(None, description="When processing started")
    processing_completed: Optional[datetime] = Field(None, description="When processing completed")
    routing_decisions: Dict[str, Any] = Field(default_factory=dict, description="Routing decisions made by orchestrator")
    agent_status: Dict[str, str] = Field(default_factory=dict, description="Status of each agent")
    classification_metadata: Dict[str, Any] = Field(default_factory=dict, description="Classification metadata")
    
    # Arabic processing specific results
    arabic_processing: Dict[str, Any] = Field(default_factory=dict, description="Arabic processing results")
    
    # Classification fields for direct access
    predicted_category: Optional[str] = Field(None, description="Predicted main category")
    category_confidence: float = Field(0.0, ge=0.0, le=1.0, description="Category confidence score")
    predicted_subcategory: Optional[str] = Field(None, description="Predicted subcategory")
    subcategory_confidence: float = Field(0.0, ge=0.0, le=1.0, description="Subcategory confidence score")
    
    # Classification results
    classification: TicketClassification = Field(default_factory=TicketClassification)
    
    # Processing information for each agent
    agent_processing: Dict[AgentType, AgentProcessingInfo] = Field(
        default_factory=lambda: {
            agent_type: AgentProcessingInfo(agent_type=agent_type)
            for agent_type in AgentType
        }
    )
    
    # Overall processing status
    overall_status: ProcessingStatus = ProcessingStatus.PENDING
    total_processing_time_ms: Optional[int] = None
    
    # Validation and quality metrics
    final_validation_score: float = Field(0.0, ge=0.0, le=1.0, description="Final validation score")
    quality_metrics: Dict[str, float] = Field(default_factory=dict, description="Various quality metrics")
    
    # Metadata
    version: str = Field("1.0", description="State schema version")
    last_updated: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        """Pydantic configuration"""
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        json_schema_extra = {
            "example": {
                "ticket_id": "ticket_123",
                "original_text": "مشكلة في تسجيل الدخول للمنصة",
                "classification": {
                    "main_category": "تسجيل الدخول",
                    "subcategory": "عدم القدرة على تسجيل الدخول",
                    "confidence_score": 0.95
                }
            }
        }
    
    def update_agent_status(self, agent_type: AgentType, status: ProcessingStatus, 
                           validation_result: Optional[ValidationResult] = None,
                           metadata: Optional[Dict[str, Any]] = None,
                           error_message: Optional[str] = None) -> None:
        """Update the processing status for a specific agent"""
        agent_info = self.agent_processing[agent_type]
        
        # Set timestamps based on status
        if status == ProcessingStatus.IN_PROGRESS and not agent_info.started_at:
            agent_info.started_at = datetime.now()
        elif status in [ProcessingStatus.COMPLETED, ProcessingStatus.FAILED]:
            if not agent_info.completed_at:
                agent_info.completed_at = datetime.now()
            
            # Calculate processing time
            if agent_info.started_at:
                time_diff = agent_info.completed_at - agent_info.started_at
                agent_info.processing_time_ms = int(time_diff.total_seconds() * 1000)
        
        # Update status and other fields
        agent_info.status = status
        if validation_result:
            agent_info.validation_result = validation_result
        if metadata:
            agent_info.metadata.update(metadata)
        if error_message:
            agent_info.error_message = error_message
        
        # Update last modified timestamp
        self.last_updated = datetime.now()
    
    def update_state(self, agent_result: Dict[str, Any]) -> None:
        """Update state with results from an agent"""
        # Update fields based on agent result
        for key, value in agent_result.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        # Update last modified timestamp
        self.last_updated = datetime.now()
    
    def get_next_agent(self) -> Optional[AgentType]:
        """Get the next agent that needs to process this ticket"""
        agent_order = [
            AgentType.ORCHESTRATOR,
            AgentType.ARABIC_PROCESSOR,
            AgentType.CATEGORY_CLASSIFIER,
            AgentType.SUBCATEGORY_CLASSIFIER,
            AgentType.VALIDATION_AGENT,
            AgentType.LEARNING_AGENT
        ]
        
        for agent_type in agent_order:
            if self.agent_processing[agent_type].status == ProcessingStatus.PENDING:
                return agent_type
        
        return None  # All agents completed
    
    def is_processing_complete(self) -> bool:
        """Check if all agents have completed processing"""
        return all(
            info.status in [ProcessingStatus.COMPLETED, ProcessingStatus.SKIPPED]
            for info in self.agent_processing.values()
        )
    
    def has_failed(self) -> bool:
        """Check if any agent has failed"""
        return any(
            info.status == ProcessingStatus.FAILED
            for info in self.agent_processing.values()
        )
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """Get a summary of processing status"""
        return {
            "ticket_id": self.ticket_id,
            "overall_status": self.overall_status,
            "agents_completed": sum(1 for info in self.agent_processing.values() 
                                  if info.status == ProcessingStatus.COMPLETED),
            "agents_failed": sum(1 for info in self.agent_processing.values() 
                               if info.status == ProcessingStatus.FAILED),
            "total_agents": len(self.agent_processing),
            "current_classification": {
                "main_category": self.classification.main_category,
                "subcategory": self.classification.subcategory,
                "confidence": self.classification.confidence_score
            },
            "processing_complete": self.is_processing_complete(),
            "has_failures": self.has_failed()
        }


# Type aliases for convenience
TicketStateDict = Dict[str, Any]
AgentProcessingDict = Dict[AgentType, AgentProcessingInfo]
