"""
STEP 3: ORCHESTRATOR AGENT - WORKFLOW COORDINATION

This agent manages the overall ticket classification workflow, initializes
the processing pipeline, and coordinates between the specialized agents.

KEY RESPONSIBILITIES:
1. Initialize ticket processing state with metadata
2. Load company-specific configurations and business rules
3. Route tickets based on complexity, priority, and type
4. Handle overall workflow errors and recovery strategies
5. Coordinate agent sequence execution and state management

ORCHESTRATION LOGIC:
- Validates incoming tickets and prepares initial state
- Determines optimal processing path based on ticket characteristics
- Sets up context for downstream agents (Arabic, Classification)
- Manages workflow routing and conditional execution
- Handles escalation rules and business logic

DESIGN DECISIONS:
- Lightweight Processing: Minimal computation, focus on coordination
- Business Rules Engine: Company-specific routing and escalation logic
- Error Recovery: Centralized error handling for the entire pipeline
- State Initialization: Prepares comprehensive state for all agents
- Configuration Loading: Company-specific settings and thresholds

INTEGRATION POINTS:
- StateManager: Persists workflow state and configuration
- ClassificationHierarchy: Loads and manages category hierarchy
- Agent Pipeline: Initiates the multi-agent processing sequence
- Business Rules: Applies company-specific routing and priority logic
"""

import json
from typing import Dict, Any, Optional
from datetime import datetime

from .base_agent import BaseAgent, BaseAgentConfig, AgentType
from ..models.ticket_state import TicketState
from ..state.state_manager import StateManager
from ..models.entities import ClassificationHierarchy


class OrchestratorAgent(BaseAgent):
    """
    Orchestrator Agent: Manages workflow coordination and routing.
    
    This agent initializes the processing pipeline and sets up the context
    for all downstream agents to perform their specialized tasks.
    """
    
    def __init__(self, config: BaseAgentConfig, state_manager: StateManager):
        super().__init__(config)
        self.state_manager = state_manager
        self.hierarchy = None  # Will be set by pipeline if needed
        self.business_rules = self._load_business_rules()
        
        self.logger.info("Orchestrator agent initialized successfully")
    
    def _load_business_rules(self) -> Dict[str, Any]:
        """Load business rules for routing and escalation"""
        # Default business rules - can be overridden by company config
        return {
            "priority_keywords": [
                "عاجل", "مستعجل", "طارئ", "فوري", "urgent", "emergency"
            ],
            "escalation_categories": [
                "المدفوعات", "البيانات المالية", "الأمان والحماية"
            ],
            "confidence_thresholds": {
                "high_confidence": 0.85,
                "medium_confidence": 0.65,
                "low_confidence": 0.45
            },
            "timeout_rules": {
                "arabic_processing": 30,
                "category_classification": 45,
                "subcategory_classification": 45
            }
        }
    
    async def process(self, state: TicketState) -> TicketState:
        """
        Initialize and coordinate the ticket processing workflow.
        
        Args:
            state: Incoming ticket state with basic information
            
        Returns:
            Enhanced state with orchestration metadata and routing info
        """
        self.logger.info(f"Orchestrating workflow for ticket {state.ticket_id}")
        
        # 1. Validate and enrich incoming state
        await self._validate_and_enrich_state(state)
        
        # 2. Apply business rules and determine routing
        await self._apply_business_rules(state)
        
        # 3. Initialize processing context
        await self._initialize_processing_context(state)
        
        # 4. Set workflow metadata
        await self._set_workflow_metadata(state)
        
        self.logger.info(f"Orchestration complete for ticket {state.ticket_id}")
        return state
    
    async def _validate_and_enrich_state(self, state: TicketState) -> None:
        """Validate incoming state and add missing required fields"""
        
        # Ensure required fields are present
        if not state.original_text or not state.original_text.strip():
            raise ValueError("Original text is required and cannot be empty")
        
        if not state.ticket_id:
            raise ValueError("Ticket ID is required")
        
        # Enrich with missing metadata
        if not state.created_at:
            state.created_at = datetime.now()
        
        # Initialize processing fields if missing
        if not state.processing_metadata:
            state.processing_metadata = {}
        
        # Initialize processing timestamps
        if not state.processing_started:
            state.processing_started = datetime.now()
        
        # Set orchestrator timestamp
        state.processing_metadata['orchestrator_start'] = datetime.now().isoformat()
        
        self.logger.debug(f"State validation complete for ticket {state.ticket_id}")
    
    async def _apply_business_rules(self, state: TicketState) -> None:
        """Apply company-specific business rules and routing logic"""
        
        text = state.original_text.lower()
        
        # 1. Priority Detection
        priority_score = 0
        for keyword in self.business_rules["priority_keywords"]:
            if keyword in text:
                priority_score += 1
        
        # Set priority based on keyword matches
        if priority_score > 0:
            state.priority = "high" if priority_score >= 2 else "medium"
            state.routing_decisions['priority_detected'] = True
            state.routing_decisions['priority_keywords_found'] = priority_score
        else:
            state.priority = "normal"
            state.routing_decisions['priority_detected'] = False
        
        # 2. Complexity Assessment
        complexity_factors = {
            'text_length': len(state.original_text),
            'has_arabic_mixed_languages': self._detect_mixed_languages(state.original_text),
            'has_technical_terms': self._has_technical_terms(state.original_text),
            'has_numbers_codes': self._has_numbers_or_codes(state.original_text)
        }
        
        complexity_score = self._calculate_complexity_score(complexity_factors)
        state.routing_decisions['complexity_score'] = complexity_score
        state.routing_decisions['complexity_factors'] = complexity_factors
        
        # 3. Routing Decisions
        if complexity_score > 0.7:
            state.routing_decisions['processing_path'] = 'complex'
            state.routing_decisions['requires_extended_processing'] = True
        elif complexity_score < 0.3:
            state.routing_decisions['processing_path'] = 'simple'
            state.routing_decisions['requires_extended_processing'] = False
        else:
            state.routing_decisions['processing_path'] = 'standard'
            state.routing_decisions['requires_extended_processing'] = False
        
        self.logger.debug(f"Business rules applied: priority={state.priority}, complexity={complexity_score:.2f}")
    
    def _detect_mixed_languages(self, text: str) -> bool:
        """Detect if text contains mixed Arabic and other languages"""
        arabic_chars = sum(1 for char in text if '\u0600' <= char <= '\u06FF')
        total_chars = len([char for char in text if char.isalpha()])
        
        if total_chars == 0:
            return False
        
        arabic_ratio = arabic_chars / total_chars
        # Mixed if Arabic is between 20% and 80% of alphabetic characters
        return 0.2 <= arabic_ratio <= 0.8
    
    def _has_technical_terms(self, text: str) -> bool:
        """Check if text contains technical terms or ITSM-specific vocabulary"""
        technical_terms = [
            "api", "ssl", "vpn", "dns", "ip", "url", "http", "https",
            "نظام", "تطبيق", "قاعدة البيانات", "خادم", "شبكة", "برنامج",
            "رمز الخطأ", "كلمة المرور", "تسجيل الدخول", "صفحة"
        ]
        
        text_lower = text.lower()
        return any(term in text_lower for term in technical_terms)
    
    def _has_numbers_or_codes(self, text: str) -> bool:
        """Check if text contains numbers, error codes, or identifiers"""
        import re
        # Look for patterns like error codes, IDs, version numbers
        patterns = [
            r'\d{3,}',  # Numbers with 3+ digits
            r'[A-Z]{2,}\d+',  # Letter-number combinations
            r'\d+\.\d+',  # Version numbers
            r'#\d+',  # Hash-prefixed numbers
        ]
        
        return any(re.search(pattern, text) for pattern in patterns)
    
    def _calculate_complexity_score(self, factors: Dict[str, Any]) -> float:
        """Calculate overall complexity score from various factors"""
        score = 0.0
        
        # Text length factor (longer text = more complex)
        if factors['text_length'] > 200:
            score += 0.3
        elif factors['text_length'] > 100:
            score += 0.2
        elif factors['text_length'] > 50:
            score += 0.1
        
        # Mixed languages increase complexity
        if factors['has_arabic_mixed_languages']:
            score += 0.25
        
        # Technical terms increase complexity
        if factors['has_technical_terms']:
            score += 0.25
        
        # Numbers and codes increase complexity
        if factors['has_numbers_codes']:
            score += 0.2
        
        return min(score, 1.0)  # Cap at 1.0
    
    async def _initialize_processing_context(self, state: TicketState) -> None:
        """Initialize context for downstream agents"""
        
        # Load category hierarchy if available
        if self.hierarchy:
            try:
                # Set hierarchy information in processing metadata
                stats = self.hierarchy.get_statistics()
                state.processing_metadata['hierarchy_loaded'] = True
                state.processing_metadata['total_categories'] = stats.get('total_categories', 0)
                state.processing_metadata['total_subcategories'] = stats.get('total_subcategories', 0)
                    
            except Exception as e:
                self.logger.warning(f"Failed to access hierarchy: {e}")
                state.processing_metadata['hierarchy_loaded'] = False
                state.processing_metadata['hierarchy_error'] = str(e)
        else:
            # No hierarchy available - will be loaded by pipeline
            state.processing_metadata['hierarchy_loaded'] = False
        
        # Set processing timeouts based on business rules
        timeouts = self.business_rules["timeout_rules"]
        state.processing_metadata['timeouts'] = timeouts
        
        # Initialize agent status tracking
        if not hasattr(state, 'agent_status'):
            state.agent_status = {}
        
        state.agent_status['orchestrator'] = {
            'status': 'completed',
            'timestamp': datetime.now().isoformat(),
            'processing_time_ms': 0  # Will be calculated by base class
        }

    def set_hierarchy(self, hierarchy: ClassificationHierarchy):
        """Set the classification hierarchy for this agent"""
        self.hierarchy = hierarchy
        self.logger.info("Classification hierarchy set successfully")
    
    async def _set_workflow_metadata(self, state: TicketState) -> None:
        """Set metadata for workflow tracking and debugging"""
        
        workflow_info = {
            'orchestrator_version': '1.0',
            'workflow_id': f"wf_{state.ticket_id}_{int(datetime.now().timestamp())}",
            'agent_sequence': ['orchestrator', 'arabic_processor', 'category_classifier', 'subcategory_classifier'],
            'estimated_processing_time_ms': self._estimate_processing_time(state),
            'routing_strategy': state.routing_decisions.get('processing_path', 'standard')
        }
        
        state.processing_metadata['workflow'] = workflow_info
        
        self.logger.info(f"Workflow metadata set: {workflow_info['workflow_id']}")
    
    def _estimate_processing_time(self, state: TicketState) -> int:
        """Estimate total processing time based on complexity"""
        base_time = 2000  # 2 seconds base time
        
        complexity_score = state.routing_decisions.get('complexity_score', 0.5)
        complexity_multiplier = 1 + complexity_score  # 1.0 to 2.0
        
        priority = state.priority or 'normal'
        priority_multiplier = {
            'high': 0.8,  # High priority gets faster processing
            'medium': 1.0,
            'normal': 1.2
        }.get(priority, 1.0)
        
        estimated_time = int(base_time * complexity_multiplier * priority_multiplier)
        return estimated_time
    
    def _validate_output_state(self, state: TicketState) -> None:
        """Validate that orchestrator has properly initialized the state"""
        super()._validate_output_state(state)
        
        # Orchestrator-specific validations
        if not hasattr(state, 'processing_metadata'):
            raise ValueError("Processing metadata not initialized")
        
        if not hasattr(state, 'routing_decisions'):
            raise ValueError("Routing decisions not initialized")
        
        if not state.priority:
            raise ValueError("Priority not set")
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics"""
        stats = {
            'agent_metrics': self.metrics.dict(),
            'business_rules': self.business_rules,
            'hierarchy_status': self.hierarchy is not None
        }
        
        return stats
