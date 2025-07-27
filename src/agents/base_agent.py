"""
STEP 3: AGENT FRAMEWORK - BASE AGENT INFRASTRUCTURE

This file defines the foundational BaseAgent class that all specialized agents
inherit from. It provides the common interface, configuration management,
error handling, and metrics collection for our 4-agent processing pipeline.

KEY CONCEPTS:
1. Abstract Base Class: Common interface for all agents
2. Configuration Management: Per-agent settings and LLM configuration
3. Error Handling: Standardized error recovery and logging
4. Metrics Collection: Performance tracking and monitoring
5. State Validation: Ensure valid state transitions between agents

AGENT RESPONSIBILITIES:
- Process incoming TicketState and return updated state
- Track processing time and success metrics
- Handle errors gracefully with fallback strategies
- Validate state changes and maintain data integrity
- Log detailed processing information for debugging

DESIGN DECISIONS:
- Async Processing: All agents use async/await for LLM calls
- Configuration-Driven: Agents are configurable for different companies
- Error Recovery: Multiple retry strategies and fallback mechanisms
- State Immutability: Agents create new state rather than mutating existing
- Rich Logging: Comprehensive logs for debugging and analytics

INTEGRATION:
- Works with existing StateManager for persistence
- Integrates with LangChain for LLM interactions
- Supports OpenAI GPT-4 and other models
- Compatible with Qdrant for vector operations
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
import time
import logging
import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

# LangChain imports for LLM integration
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

# Local imports
from ..config.agent_config import BaseAgentConfig


class AgentType(str, Enum):
    """Enumeration of agent types in our 4-agent pipeline"""
    ORCHESTRATOR = "orchestrator"
    ARABIC_PROCESSOR = "arabic_processor"
    CATEGORY_CLASSIFIER = "category_classifier"
    SUBCATEGORY_CLASSIFIER = "subcategory_classifier"


class AgentException(Exception):
    """Custom exception for agent processing errors"""
    def __init__(self, agent_type: str, message: str, original_error: Exception = None):
        self.agent_type = agent_type
        self.original_error = original_error
        super().__init__(f"[{agent_type}] {message}")


class AgentMetrics(BaseModel):
    """
    Enhanced metrics tracking for agent performance monitoring.
    
    Tracks processing success/failure rates, timing, and errors for
    performance optimization and system monitoring.
    """
    
    total_processed: int = 0
    successful_processed: int = 0
    failed_processed: int = 0
    average_processing_time_ms: float = 0.0
    total_processing_time_ms: float = 0.0
    last_error: Optional[str] = None
    last_processed_at: Optional[datetime] = None
    
    def update_success(self, processing_time_ms: float):
        """Update metrics for successful processing"""
        self.total_processed += 1
        self.successful_processed += 1
        self.total_processing_time_ms += processing_time_ms
        self.average_processing_time_ms = self.total_processing_time_ms / self.total_processed
        self.last_processed_at = datetime.now()
    
    def update_failure(self, error_message: str):
        """Update metrics for failed processing"""
        self.total_processed += 1
        self.failed_processed += 1
        self.last_error = error_message
        self.last_processed_at = datetime.now()
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage"""
        if self.total_processed == 0:
            return 0.0
        return (self.successful_processed / self.total_processed) * 100


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the ITSM classification system.
    
    This class provides:
    1. Common interface that all agents must implement (process method)
    2. Shared functionality (LLM initialization, logging, metrics)
    3. Error handling and retry logic
    4. Performance monitoring
    
    Educational Concepts:
    - ABC ensures all child classes implement the process() method
    - Composition: each agent HAS an LLM, rather than IS an LLM
    - Template Method Pattern: common structure, specialized implementations
    """
    
    def __init__(self, config: BaseAgentConfig):
        """
        Initialize the base agent with configuration.
        
        Args:
            config: BaseAgentConfig containing LLM and agent settings
            
        Educational: This __init__ method will be called by all child classes
        using super().__init__(config)
        """
        self.config = config
        self.metrics = AgentMetrics()
        self.logger = self._setup_logging()
        
        # Validate OpenAI API key
        if not self.config.api_key:
            raise ValueError("OpenAI API key is required but not provided")
        
        if self.config.api_key == "your-api-key-here":
            raise ValueError("Valid OpenAI API key required - please set a real API key")
        
        # Initialize the LLM - this is the core AI component
        self.llm = self._initialize_llm()
        
        self.logger.info(f"Initialized {self.config.agent_name} agent")
    
    def _setup_logging(self) -> logging.Logger:
        """
        Set up logging for this agent.
        
        Educational: Each agent gets its own logger with its name,
        making debugging much easier when you have multiple agents running.
        """
        logger = logging.getLogger(f"agent.{self.config.agent_name}")
        
        # Only add handler if it doesn't already exist (prevents duplicate logs)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        
        return logger
    
    def _initialize_llm(self) -> ChatOpenAI:
        """
        Initialize the Language Model using LangChain's ChatOpenAI wrapper.
        
        Returns:
            ChatOpenAI instance configured with agent settings
            
        Educational: We use LangChain's ChatOpenAI instead of raw OpenAI client
        because it provides better integration with LangGraph and other tools.
        """
        try:
            llm = ChatOpenAI(
                model=self.config.model_name,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                api_key=self.config.api_key,
                organization=self.config.organization_id,
                timeout=self.config.timeout_seconds,
                max_retries=self.config.retry_attempts
            )
            
            self.logger.info(f"Initialized LLM: {self.config.model_name}")
            return llm
            
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM: {e}")
            raise
    
    @abstractmethod
    async def process(self, state: Any) -> Any:
        """
        Process the input state and return the updated state.
        
        This is the main method that each agent must implement.
        It defines what the agent actually does with the data.
        
        Args:
            state: The current state of the ticket processing workflow
            
        Returns:
            Updated state after this agent's processing
            
        Educational: @abstractmethod means:
        1. This method MUST be implemented by child classes
        2. You cannot create an instance of BaseAgent directly
        3. Python will error if a child class doesn't implement this method
        """
        pass
    
    async def _safe_llm_call(self, messages: list, **kwargs) -> Any:
        """
        Make a safe call to the LLM with error handling and metrics tracking.
        
        Args:
            messages: List of messages to send to the LLM
            **kwargs: Additional arguments for the LLM call
            
        Returns:
            LLM response
            
        Educational: This method demonstrates:
        - Error handling with try/except
        - Performance monitoring with timing
        - Retry logic for resilience
        - Logging for debugging
        """
        start_time = time.time()
        
        try:
            self.logger.debug(f"Making LLM call with {len(messages)} messages")
            
            # Make the actual LLM call
            response = await self.llm.ainvoke(messages, **kwargs)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Update success metrics
            self.metrics.update_success(processing_time)
            
            self.logger.debug(f"LLM call succeeded in {processing_time:.2f}s")
            return response
            
        except Exception as e:
            # Calculate processing time even for failures
            processing_time = time.time() - start_time
            
            # Update failure metrics
            self.metrics.update_failure(processing_time)
            
            self.logger.error(f"LLM call failed after {processing_time:.2f}s: {e}")
            raise
    
    async def _call_llm(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        """
        Convenience method for calling LLM with system and user prompts.
        
        Args:
            system_prompt: System message content
            user_prompt: User message content
            **kwargs: Additional arguments for the LLM call
            
        Returns:
            LLM response as string
        """
        messages = [
            self._create_system_message(system_prompt),
            self._create_human_message(user_prompt)
        ]
        
        response = await self._safe_llm_call(messages, **kwargs)
        return response.content
    
    def _create_system_message(self, content: str) -> SystemMessage:
        """
        Create a system message for the LLM.
        
        Args:
            content: The system prompt content
            
        Returns:
            SystemMessage instance
            
        Educational: System messages set the context and behavior for the LLM.
        They're like instructions that tell the LLM how to behave.
        """
        return SystemMessage(content=content)
    
    def _create_human_message(self, content: str) -> HumanMessage:
        """
        Create a human message for the LLM.
        
        Args:
            content: The user/human input content
            
        Returns:
            HumanMessage instance
            
        Educational: Human messages represent user input or data to be processed.
        """
        return HumanMessage(content=content)
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current performance metrics for this agent.
        
        Returns:
            Dictionary containing metrics data
            
        Educational: This allows monitoring systems to track agent performance.
        """
        return {
            "agent_name": self.config.agent_name,
            "total_requests": self.metrics.total_requests,
            "success_rate": self.metrics.success_rate,
            "average_processing_time": self.metrics.average_processing_time,
            "last_request_time": self.metrics.last_request_time.isoformat() if self.metrics.last_request_time else None
        }
    
    def __repr__(self) -> str:
        """
        String representation of the agent for debugging.
        
        Educational: __repr__ is called when you print() an object
        or view it in a debugger. Make it informative!
        """
        return (f"{self.__class__.__name__}("
                f"name='{self.config.agent_name}', "
                f"model='{self.config.model_name}', "
                f"requests={self.metrics.total_requests})")


# Educational Example: How to create a concrete agent
class ExampleAgent(BaseAgent):
    """
    Example concrete implementation of BaseAgent.
    
    This shows the minimum required implementation - just the process() method.
    Real agents will have much more sophisticated logic!
    """
    
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Example implementation of the required process() method.
        
        Args:
            state: Dictionary containing the current processing state
            
        Returns:
            Updated state dictionary
        """
        self.logger.info(f"Processing state with keys: {list(state.keys())}")
        
        # Example: Add a timestamp to show this agent processed the state
        state["processed_by"] = self.config.agent_name
        state["processed_at"] = datetime.now().isoformat()
        
        return state


# Educational demonstration
if __name__ == "__main__":
    """
    This section demonstrates how to use the BaseAgent class.
    Run this file directly to see examples in action!
    
    Note: This will only work if you have set up your .env file with OPENAI_API_KEY
    """
    import asyncio
    from ..config.agent_config import create_default_agent_config
    
    async def demo():
        print("=== Base Agent Demonstration ===\n")
        
        try:
            # Create a configuration
            config = create_default_agent_config("demo_agent")
            print(f"✅ Created config: {config.agent_name}")
            
            # Create an example agent
            agent = ExampleAgent(config)
            print(f"✅ Created agent: {agent}")
            
            # Test processing some state
            test_state = {
                "ticket_id": "TEST-001",
                "text": "مرحبا، أحتاج مساعدة في الطابعة",  # "Hello, I need help with the printer"
                "timestamp": datetime.now().isoformat()
            }
            
            print(f"\n📥 Input state: {test_state}")
            
            # Process the state
            result_state = await agent.process(test_state)
            print(f"📤 Output state: {result_state}")
            
            # Show metrics
            metrics = agent.get_metrics()
            print(f"\n📊 Agent metrics: {metrics}")
            
        except Exception as e:
            print(f"❌ Demo failed: {e}")
            print("   Make sure your .env file has a valid OPENAI_API_KEY!")
    
    # Run the demo
    asyncio.run(demo())
