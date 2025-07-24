"""
STEP 2: STATE MANAGEMENT SYSTEM - STATE VALIDATOR

This file implements comprehensive validation for our ticket state throughout
the multi-agent pipeline. It ensures data integrity, classification accuracy,
and system reliability by validating state at each processing step.

KEY CONCEPTS:
1. Multi-level Validation: Different validation rules for each agent
2. Business Rules: Validates against ITSM domain knowledge
3. Data Integrity: Ensures state consistency and completeness
4. Performance Monitoring: Tracks validation performance

DESIGN DECISIONS:
- Modular validation: Each agent has specific validation rules
- Configurable strictness: Can adjust validation sensitivity
- Rich feedback: Provides detailed validation results with recommendations
- Early failure detection: Catches issues before they propagate

Validation happens after each agent:
Orchestrator → Arabic Processing → Category → Subcategory → Validation → Learning
     ↓              ↓              ↓           ↓             ↓          ↓
  Basic Val.    Text Val.      Cat Val.    SubCat Val.   Final Val.  Learning Val.
"""

from typing import List, Dict, Optional, Set, Callable
from datetime import datetime, timedelta
import re
from dataclasses import dataclass

from ..models.ticket_state import (
    TicketState, AgentType, ProcessingStatus, ValidationResult,
    TicketClassification, AgentProcessingInfo
)
from ..models.entities import ClassificationHierarchy, Category, Subcategory


@dataclass
class ValidationConfig:
    """Configuration for validation behavior"""
    # Strictness levels
    require_arabic_text: bool = True
    min_text_length: int = 5
    max_text_length: int = 5000
    min_confidence_threshold: float = 0.6
    max_processing_time_ms: int = 30000  # 30 seconds
    
    # Validation rules
    require_category_description: bool = True
    require_subcategory_match: bool = True
    validate_against_hierarchy: bool = True
    check_processing_times: bool = True
    
    # Performance thresholds
    warning_processing_time_ms: int = 10000  # 10 seconds
    max_validation_errors: int = 5


class StateValidator:
    """
    Comprehensive validator for ticket state throughout the processing pipeline.
    
    This validator ensures that the ticket state remains consistent and valid
    as it flows through our multi-agent system. It provides different validation
    rules for each processing stage.
    """
    
    def __init__(self, hierarchy: ClassificationHierarchy, config: ValidationConfig = None):
        """
        Initialize the validator with classification hierarchy and configuration.
        
        Args:
            hierarchy: The loaded classification hierarchy for validation
            config: Validation configuration (uses defaults if not provided)
        """
        self.hierarchy = hierarchy
        self.config = config or ValidationConfig()
        
        # Arabic text pattern for validation
        self.arabic_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+')
        
        # Validation rule registry
        self.validation_rules: Dict[AgentType, List[Callable]] = {
            AgentType.ORCHESTRATOR: [
                self._validate_basic_state,
                self._validate_original_text,
                self._validate_timestamps
            ],
            AgentType.ARABIC_PROCESSOR: [
                self._validate_basic_state,
                self._validate_arabic_processing,
                self._validate_language_confidence,
                self._validate_keywords_extraction
            ],
            AgentType.CATEGORY_CLASSIFIER: [
                self._validate_basic_state,
                self._validate_category_classification,
                self._validate_classification_confidence,
                self._validate_against_hierarchy
            ],
            AgentType.SUBCATEGORY_CLASSIFIER: [
                self._validate_basic_state,
                self._validate_subcategory_classification,
                self._validate_classification_hierarchy_match,
                self._validate_final_classification_confidence
            ],
            AgentType.VALIDATION_AGENT: [
                self._validate_basic_state,
                self._validate_complete_classification,
                self._validate_processing_quality,
                self._validate_performance_metrics
            ],
            AgentType.LEARNING_AGENT: [
                self._validate_basic_state,
                self._validate_learning_metadata,
                self._validate_final_state
            ]
        }
    
    def validate_state(self, state: TicketState, agent_type: AgentType) -> ValidationResult:
        """
        Validate the ticket state for a specific agent.
        
        Args:
            state: The ticket state to validate
            agent_type: The agent type that just processed the state
            
        Returns:
            ValidationResult with validation outcome and details
        """
        validation_start = datetime.now()
        issues = []
        recommendations = []
        confidence_scores = []
        
        try:
            # Get validation rules for this agent
            rules = self.validation_rules.get(agent_type, [])
            
            # Run all validation rules
            for rule in rules:
                try:
                    rule_result = rule(state)
                    if rule_result:
                        issues.extend(rule_result.get('issues', []))
                        recommendations.extend(rule_result.get('recommendations', []))
                        if 'confidence' in rule_result:
                            confidence_scores.append(rule_result['confidence'])
                except Exception as e:
                    issues.append(f"Validation rule error: {str(e)}")
                    confidence_scores.append(0.0)
            
            # Calculate overall confidence
            overall_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 1.0
            
            # Determine if validation passed
            is_valid = len(issues) == 0 or len(issues) <= self.config.max_validation_errors
            
            # Adjust confidence based on issues
            if issues:
                confidence_penalty = min(0.3, len(issues) * 0.1)
                overall_confidence = max(0.0, overall_confidence - confidence_penalty)
            
            validation_time = datetime.now() - validation_start
            
            return ValidationResult(
                is_valid=is_valid,
                confidence=overall_confidence,
                issues=issues,
                recommendations=recommendations
            )
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                confidence=0.0,
                issues=[f"Validation failed with error: {str(e)}"],
                recommendations=["Review the ticket state and processing logic"]
            )
    
    def validate_complete_pipeline(self, state: TicketState) -> ValidationResult:
        """
        Validate the complete pipeline processing for a ticket.
        
        Args:
            state: The ticket state after complete processing
            
        Returns:
            ValidationResult for the entire pipeline
        """
        issues = []
        recommendations = []
        
        # Check that all agents have been processed
        for agent_type in AgentType:
            agent_info = state.agent_processing.get(agent_type)
            if not agent_info or agent_info.status not in [ProcessingStatus.COMPLETED, ProcessingStatus.SKIPPED]:
                issues.append(f"Agent {agent_type.value} has not completed processing")
        
        # Validate final classification
        if not state.classification.main_category:
            issues.append("No main category classification found")
        
        if not state.classification.subcategory:
            issues.append("No subcategory classification found")
        
        # Validate classification exists in hierarchy
        if (state.classification.main_category and state.classification.subcategory and
            not self.hierarchy.validate_classification(
                state.classification.main_category,
                state.classification.subcategory
            )):
            issues.append("Classification combination does not exist in hierarchy")
        
        # Check overall confidence
        if state.classification.confidence_score < self.config.min_confidence_threshold:
            issues.append(f"Classification confidence {state.classification.confidence_score:.2f} below threshold {self.config.min_confidence_threshold}")
            recommendations.append("Consider manual review or reprocessing with different parameters")
        
        # Check processing times
        total_time = sum(
            info.processing_time_ms or 0
            for info in state.agent_processing.values()
            if info.processing_time_ms
        )
        
        if total_time > self.config.max_processing_time_ms:
            issues.append(f"Total processing time {total_time}ms exceeds maximum {self.config.max_processing_time_ms}ms")
            recommendations.append("Optimize agent processing performance")
        
        is_valid = len(issues) == 0
        confidence = 1.0 - (len(issues) * 0.2)  # Reduce confidence for each issue
        
        return ValidationResult(
            is_valid=is_valid,
            confidence=max(0.0, confidence),
            issues=issues,
            recommendations=recommendations
        )
    
    # Validation rule implementations
    
    def _validate_basic_state(self, state: TicketState) -> Optional[Dict]:
        """Validate basic state structure and required fields"""
        issues = []
        
        if not state.ticket_id:
            issues.append("Missing ticket ID")
        
        if not state.original_text or len(state.original_text.strip()) == 0:
            issues.append("Missing or empty original text")
        
        if len(state.original_text) < self.config.min_text_length:
            issues.append(f"Original text too short (minimum {self.config.min_text_length} characters)")
        
        if len(state.original_text) > self.config.max_text_length:
            issues.append(f"Original text too long (maximum {self.config.max_text_length} characters)")
        
        return {"issues": issues, "confidence": 1.0 if not issues else 0.5}
    
    def _validate_original_text(self, state: TicketState) -> Optional[Dict]:
        """Validate the original ticket text"""
        issues = []
        recommendations = []
        
        if self.config.require_arabic_text:
            if not self.arabic_pattern.search(state.original_text):
                issues.append("No Arabic text detected in original text")
                recommendations.append("Verify that the ticket text is in Arabic")
        
        # Check for suspicious content
        suspicious_patterns = ['test', 'testing', '123', 'xxx']
        text_lower = state.original_text.lower()
        for pattern in suspicious_patterns:
            if pattern in text_lower:
                recommendations.append(f"Text contains potentially test content: '{pattern}'")
        
        return {"issues": issues, "recommendations": recommendations, "confidence": 0.9}
    
    def _validate_timestamps(self, state: TicketState) -> Optional[Dict]:
        """Validate timestamp consistency"""
        issues = []
        
        if state.created_at > datetime.now():
            issues.append("Created timestamp is in the future")
        
        if state.last_updated < state.created_at:
            issues.append("Last updated timestamp is before created timestamp")
        
        return {"issues": issues, "confidence": 1.0 if not issues else 0.0}
    
    def _validate_arabic_processing(self, state: TicketState) -> Optional[Dict]:
        """Validate Arabic text processing results"""
        issues = []
        recommendations = []
        
        if not state.processed_text:
            issues.append("No processed text found after Arabic processing")
        elif len(state.processed_text) < len(state.original_text) * 0.5:
            recommendations.append("Processed text significantly shorter than original - verify processing")
        
        return {"issues": issues, "recommendations": recommendations, "confidence": 0.9}
    
    def _validate_language_confidence(self, state: TicketState) -> Optional[Dict]:
        """Validate language detection confidence"""
        issues = []
        recommendations = []
        
        if state.language_confidence < 0.7:
            issues.append(f"Low language confidence: {state.language_confidence:.2f}")
            recommendations.append("Consider manual language verification")
        
        return {"issues": issues, "recommendations": recommendations, "confidence": state.language_confidence}
    
    def _validate_keywords_extraction(self, state: TicketState) -> Optional[Dict]:
        """Validate keyword extraction"""
        issues = []
        recommendations = []
        
        if len(state.extracted_keywords) == 0:
            issues.append("No keywords extracted from text")
        elif len(state.extracted_keywords) > 20:
            recommendations.append(f"Many keywords extracted ({len(state.extracted_keywords)}) - consider filtering")
        
        return {"issues": issues, "recommendations": recommendations, "confidence": 0.8}
    
    def _validate_category_classification(self, state: TicketState) -> Optional[Dict]:
        """Validate main category classification"""
        issues = []
        recommendations = []
        
        if not state.classification.main_category:
            issues.append("No main category classification")
        elif not self.hierarchy.get_category(state.classification.main_category):
            issues.append(f"Main category '{state.classification.main_category}' not found in hierarchy")
            recommendations.append("Verify category name against loaded hierarchy")
        
        return {"issues": issues, "recommendations": recommendations, "confidence": 0.9}
    
    def _validate_classification_confidence(self, state: TicketState) -> Optional[Dict]:
        """Validate classification confidence scores"""
        issues = []
        
        if state.classification.confidence_score < self.config.min_confidence_threshold:
            issues.append(f"Classification confidence {state.classification.confidence_score:.2f} below threshold")
        
        return {"issues": issues, "confidence": state.classification.confidence_score}
    
    def _validate_against_hierarchy(self, state: TicketState) -> Optional[Dict]:
        """Validate classification against loaded hierarchy"""
        issues = []
        
        if state.classification.main_category:
            category = self.hierarchy.get_category(state.classification.main_category)
            if not category:
                issues.append(f"Category '{state.classification.main_category}' not in hierarchy")
        
        return {"issues": issues, "confidence": 1.0 if not issues else 0.0}
    
    def _validate_subcategory_classification(self, state: TicketState) -> Optional[Dict]:
        """Validate subcategory classification"""
        issues = []
        
        if not state.classification.subcategory:
            issues.append("No subcategory classification")
        
        return {"issues": issues, "confidence": 0.9}
    
    def _validate_classification_hierarchy_match(self, state: TicketState) -> Optional[Dict]:
        """Validate that category and subcategory match in hierarchy"""
        issues = []
        
        if (state.classification.main_category and state.classification.subcategory and
            not self.hierarchy.validate_classification(
                state.classification.main_category,
                state.classification.subcategory
            )):
            issues.append("Category and subcategory combination not valid in hierarchy")
        
        return {"issues": issues, "confidence": 1.0 if not issues else 0.0}
    
    def _validate_final_classification_confidence(self, state: TicketState) -> Optional[Dict]:
        """Validate final classification confidence"""
        issues = []
        
        if state.classification.confidence_score < self.config.min_confidence_threshold:
            issues.append("Final classification confidence too low")
        
        return {"issues": issues, "confidence": state.classification.confidence_score}
    
    def _validate_complete_classification(self, state: TicketState) -> Optional[Dict]:
        """Validate that classification is complete"""
        issues = []
        
        required_fields = [
            ('main_category', state.classification.main_category),
            ('subcategory', state.classification.subcategory),
            ('confidence_score', state.classification.confidence_score)
        ]
        
        for field_name, field_value in required_fields:
            if not field_value:
                issues.append(f"Missing {field_name} in final classification")
        
        return {"issues": issues, "confidence": 1.0 if not issues else 0.5}
    
    def _validate_processing_quality(self, state: TicketState) -> Optional[Dict]:
        """Validate overall processing quality"""
        issues = []
        recommendations = []
        
        # Check for agent failures
        failed_agents = [
            agent_type.value for agent_type, info in state.agent_processing.items()
            if info.status == ProcessingStatus.FAILED
        ]
        
        if failed_agents:
            issues.append(f"Failed agents: {', '.join(failed_agents)}")
            recommendations.append("Review failed agent logs and retry processing")
        
        return {"issues": issues, "recommendations": recommendations, "confidence": 0.9}
    
    def _validate_performance_metrics(self, state: TicketState) -> Optional[Dict]:
        """Validate performance metrics"""
        issues = []
        recommendations = []
        
        for agent_type, info in state.agent_processing.items():
            if info.processing_time_ms and info.processing_time_ms > self.config.warning_processing_time_ms:
                recommendations.append(f"Agent {agent_type.value} took {info.processing_time_ms}ms (consider optimization)")
        
        return {"issues": issues, "recommendations": recommendations, "confidence": 0.8}
    
    def _validate_learning_metadata(self, state: TicketState) -> Optional[Dict]:
        """Validate learning agent metadata"""
        issues = []
        
        learning_info = state.agent_processing.get(AgentType.LEARNING_AGENT)
        if learning_info and learning_info.status == ProcessingStatus.COMPLETED:
            if not learning_info.metadata:
                issues.append("Learning agent completed but no metadata stored")
        
        return {"issues": issues, "confidence": 0.9}
    
    def _validate_final_state(self, state: TicketState) -> Optional[Dict]:
        """Validate final state completeness"""
        issues = []
        
        if not state.is_processing_complete():
            issues.append("Processing not complete but learning agent finished")
        
        if state.has_failed():
            issues.append("State has failures but reached learning agent")
        
        return {"issues": issues, "confidence": 1.0 if not issues else 0.0}
