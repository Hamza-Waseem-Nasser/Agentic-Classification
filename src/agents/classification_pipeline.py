"""
STEP 3: CLASSIFICATION PIPELINE - AGENT ORCHESTRATION

This module provides the complete 4-agent classification pipeline that processes
Arabic ITSM tickets from raw text to final categorization. It coordinates the
workflow between all agents and handles the data flow and error management.

PIPELINE FLOW:
Raw Arabic Ticket → Orchestrator → Arabic Processing → Category Classification → Subcategory Classification → Final Result

AGENT COORDINATION:
1. OrchestratorAgent: Workflow management, routing, and business rules
2. ArabicProcessingAgent: Language processing, normalization, and entity extraction
3. CategoryClassifierAgent: Main category classification using vector search + LLM
4. SubcategoryClassifierAgent: Hierarchical subcategory classification

KEY FEATURES:
- End-to-End Processing: Complete ticket classification workflow
- Error Recovery: Graceful handling of agent failures with fallbacks
- Performance Monitoring: Timing and metrics collection across agents
- State Management: Consistent state tracking throughout the pipeline
- Configuration Management: Centralized configuration for all agents

DESIGN DECISIONS:
- Sequential Processing: Each agent builds upon the previous agent's output
- State Persistence: Full state maintained throughout the pipeline
- Error Isolation: Agent failures don't cascade to other agents
- Metrics Collection: Comprehensive performance and accuracy tracking
- Flexible Configuration: Easy customization of agent behaviors

INTEGRATION STRATEGY:
- Dependency Injection: Agents receive dependencies through constructor
- Event-Driven: Agents emit events for monitoring and debugging
- Configurable: Pipeline behavior adjustable through configuration
- Extensible: Easy to add new agents or modify existing ones
"""

import asyncio
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import logging

from .orchestrator_agent import OrchestratorAgent
from .arabic_processing_agent import ArabicProcessingAgent
from .category_classifier_agent import CategoryClassifierAgent
from .subcategory_classifier_agent import SubcategoryClassifierAgent
from .base_agent import BaseAgentConfig, AgentException
from ..models.ticket_state import TicketState
from ..models.entities import ClassificationHierarchy, Category, Subcategory
from ..state.state_manager import StateManager
import json
import os


class ClassificationPipeline:
    """
    Complete ITSM Classification Pipeline
    
    Coordinates the 4-agent workflow to provide end-to-end ticket classification
    with comprehensive error handling and performance monitoring.
    """
    
    def __init__(self, 
                 config_path: Optional[str] = None,
                 hierarchy: Optional[ClassificationHierarchy] = None,
                 qdrant_client=None):
        """
        Initialize the classification pipeline with all agents.
        
        Args:
            config_path: Path to configuration file
            hierarchy: Classification hierarchy (optional, will load if not provided)
            qdrant_client: Qdrant client for vector operations (optional)
        """
        from src.utils.logging_config import get_logger
        self.logger = get_logger(__name__)
        
        # Load configuration - use simple approach for now
        self.config = self._load_config(config_path)
        
        # Load classification hierarchy
        self.hierarchy = hierarchy or self._load_hierarchy()
        
        # Initialize state manager
        self.state_manager = StateManager(
            storage_path="state_storage",
            hierarchy=self.hierarchy
        )
        
        # Initialize agents
        self._initialize_agents(qdrant_client)
        
        # Pipeline metrics
        self.metrics = {
            'total_processed': 0,
            'successful_classifications': 0,
            'failed_classifications': 0,
            'average_processing_time': 0.0,
            'agent_performance': {
                'orchestrator': {'calls': 0, 'failures': 0, 'avg_time': 0.0},
                'arabic_processing': {'calls': 0, 'failures': 0, 'avg_time': 0.0},
                'category_classifier': {'calls': 0, 'failures': 0, 'avg_time': 0.0},
                'subcategory_classifier': {'calls': 0, 'failures': 0, 'avg_time': 0.0}
            }
        }
        
        # Add accuracy tracking
        self.accuracy_stats = {
            'total': 0,
            'correct': 0,
            'by_category': {},
            'confidence_distribution': []
        }
        
        self.logger.info("Classification pipeline initialized successfully")
    
    @classmethod
    async def create(cls, 
                     config_path: Optional[str] = None,
                     hierarchy: Optional[ClassificationHierarchy] = None,
                     qdrant_client=None):
        """
        Async factory method to properly initialize ClassificationPipeline.
        
        Args:
            config_path: Path to configuration file
            hierarchy: Classification hierarchy (optional, will load if not provided)
            qdrant_client: Qdrant client for vector operations (optional)
            
        Returns:
            Fully initialized ClassificationPipeline
        """
        instance = cls(config_path, hierarchy, qdrant_client)
        await instance._async_initialize()
        return instance
    
    async def _async_initialize(self):
        """Complete async initialization of the pipeline"""
        # Initialize category classifier with async factory (only if not already created)
        if self.category_classifier is None:
            api_key = self.config.get('openai', {}).get('api_key')
            category_config = BaseAgentConfig(
                agent_name="category_classifier",
                model_name=self.config.get('openai', {}).get('model', 'gpt-4'),
                temperature=self.config.get('openai', {}).get('temperature', 0.1),
                max_tokens=self.config.get('openai', {}).get('max_tokens', 1000),
                api_key=api_key
            )
            
            self.category_classifier = await CategoryClassifierAgent.create(
                config=category_config,
                hierarchy=self.hierarchy,
                qdrant_url=self.config.get('qdrant', {}).get('host', 'http://localhost:6333'),
                collection_name=self.config.get('qdrant', {}).get('collection_name', 'itsm_categories')
            )
            
            self.logger.info("Async initialization completed for category classifier")
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        default_config = {
            'openai': {
                'api_key': os.getenv('OPENAI_API_KEY'),
                'model': 'gpt-4o-mini',
                'temperature': 0.1,
                'max_tokens': 1000
            },
            'classification': {
                'hierarchy_path': 'data/classification_hierarchy.json',
                'confidence_threshold': 0.7
            },
            'qdrant': {
                'host': 'localhost',
                'port': 6333,
                'collection_name': 'itsm_categories'
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                # Merge with defaults
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                return config
            except Exception as e:
                self.logger.warning(f"Could not load config from {config_path}: {e}")
        
        return default_config
    
    def _load_hierarchy(self) -> ClassificationHierarchy:
        """Load classification hierarchy from configuration."""
        try:
            # Load from configuration file or default CSV location
            hierarchy_path = self.config.get('classification', {}).get('hierarchy_path', 'Category + Subcategory.csv')
            self.logger.info(f"Loading hierarchy from: {hierarchy_path}")
            return ClassificationHierarchy.load_from_file(hierarchy_path)
        except Exception as e:
            self.logger.warning(f"Could not load hierarchy from file: {e}")
            # Return a basic hierarchy for demo purposes
            return self._create_demo_hierarchy()
    
    def _create_demo_hierarchy(self) -> ClassificationHierarchy:
        """Create a demo hierarchy for testing purposes."""
        categories = [
            Category(
                id="hardware_issues",
                name="مشاكل الأجهزة",
                description="مشاكل متعلقة بالأجهزة والمعدات",
                keywords=["جهاز", "هاردوير", "معدات", "مكونات"],
                subcategories=[
                    Subcategory(
                        id="computer_hardware",
                        name="أجهزة الكمبيوتر",
                        description="مشاكل أجهزة الكمبيوتر والمكونات",
                        keywords=["كمبيوتر", "معالج", "ذاكرة", "قرص صلب"]
                    ),
                    Subcategory(
                        id="network_hardware",
                        name="أجهزة الشبكة",
                        description="مشاكل أجهزة الشبكة والاتصالات",
                        keywords=["راوتر", "سويتش", "شبكة", "كابل"]
                    )
                ]
            ),
            Category(
                id="software_issues",
                name="مشاكل البرمجيات",
                description="مشاكل متعلقة بالبرمجيات والتطبيقات",
                keywords=["برنامج", "تطبيق", "نظام", "سوفتوير"],
                subcategories=[
                    Subcategory(
                        id="application_issues",
                        name="مشاكل التطبيقات",
                        description="مشاكل في التطبيقات والبرامج",
                        keywords=["تطبيق", "برنامج", "خطأ", "تعطل"]
                    ),
                    Subcategory(
                        id="system_issues",
                        name="مشاكل النظام",
                        description="مشاكل نظام التشغيل",
                        keywords=["نظام", "ويندوز", "تشغيل", "إقلاع"]
                    )
                ]
            )
        ]
        
        return ClassificationHierarchy(categories=categories)
    
    def _initialize_agents(self, qdrant_client=None):
        """Initialize all agents with proper configuration."""
        # Get OpenAI API key from config
        api_key = self.config.get('openai', {}).get('api_key')
        if not api_key:
            raise ValueError("OpenAI API key is required in configuration")
        
        # Create base configs for each agent
        orchestrator_config = BaseAgentConfig(
            agent_name="orchestrator",
            model_name=self.config.get('openai', {}).get('model', 'gpt-4'),
            temperature=self.config.get('openai', {}).get('temperature', 0.1),
            max_tokens=self.config.get('openai', {}).get('max_tokens', 1000),
            api_key=api_key
        )
        
        arabic_config = BaseAgentConfig(
            agent_name="arabic_processor",
            model_name=self.config.get('openai', {}).get('model', 'gpt-4'),
            temperature=self.config.get('openai', {}).get('temperature', 0.1),
            max_tokens=self.config.get('openai', {}).get('max_tokens', 1000),
            api_key=api_key
        )
        
        category_config = BaseAgentConfig(
            agent_name="category_classifier",
            model_name=self.config.get('openai', {}).get('model', 'gpt-4'),
            temperature=self.config.get('openai', {}).get('temperature', 0.1),
            max_tokens=self.config.get('openai', {}).get('max_tokens', 1000),
            api_key=api_key
        )
        
        subcategory_config = BaseAgentConfig(
            agent_name="subcategory_classifier",
            model_name=self.config.get('openai', {}).get('model', 'gpt-4'),
            temperature=self.config.get('openai', {}).get('temperature', 0.1),
            max_tokens=self.config.get('openai', {}).get('max_tokens', 1000),
            api_key=api_key
        )
        
        # Initialize agents
        try:
            self.orchestrator = OrchestratorAgent(
                config=orchestrator_config,
                state_manager=self.state_manager
            )
            
            self.arabic_processor = ArabicProcessingAgent(config=arabic_config)
            
            # Note: CategoryClassifierAgent will need async initialization
            # This will be handled in the async_initialize method
            # Don't create the instance here to avoid duplicates
            self.category_classifier = None
            
            self.subcategory_classifier = SubcategoryClassifierAgent(
                config=subcategory_config,
                hierarchy=self.hierarchy
            )
            
            self.logger.info("All agents initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize agents: {e}")
            raise Exception(f"Agent initialization failed: {e}")
    
    async def classify_ticket(self, ticket_text: str, ticket_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a ticket through the complete classification pipeline.
        
        Args:
            ticket_text: The Arabic ticket text to classify
            ticket_id: Optional ticket ID for tracking
            
        Returns:
            Dictionary containing classification results and metadata
        """
        start_time = time.time()
        ticket_id = ticket_id or f"ticket_{int(time.time())}"
        
        try:
            self.logger.info(f"Starting classification for ticket {ticket_id}")
            
            # Initialize ticket state
            ticket_state = TicketState(
                ticket_id=ticket_id,
                original_text=ticket_text,
                processing_started=datetime.now()
            )
            
            # Step 1: Orchestrator - Initial assessment and routing
            self.logger.info("Step 1: Orchestrator processing")
            orchestrator_start = time.time()
            
            try:
                orchestrator_result = await self.orchestrator.process(ticket_state)
                # orchestrator returns the updated state object
                ticket_state = orchestrator_result
                self._update_agent_metrics('orchestrator', time.time() - orchestrator_start, success=True)
                
            except Exception as e:
                self._update_agent_metrics('orchestrator', time.time() - orchestrator_start, success=False)
                self.logger.error(f"Orchestrator failed: {e}")
                # Continue with default routing
                ticket_state.metadata['orchestrator_error'] = str(e)
            
            # Step 2: Arabic Processing - Language analysis and normalization
            self.logger.info("Step 2: Arabic text processing")
            arabic_start = time.time()
            
            try:
                arabic_result = await self.arabic_processor.process(ticket_state)
                # arabic processor returns the updated state object
                ticket_state = arabic_result
                self._update_agent_metrics('arabic_processing', time.time() - arabic_start, success=True)
                
            except Exception as e:
                self._update_agent_metrics('arabic_processing', time.time() - arabic_start, success=False)
                self.logger.error(f"Arabic processing failed: {e}")
                # Continue with original text
                ticket_state.processed_text = ticket_text
                ticket_state.metadata['arabic_processing_error'] = str(e)
            
            # Step 3: Category Classification - Main category identification
            self.logger.info("Step 3: Category classification")
            category_start = time.time()
            
            try:
                category_result = await self.category_classifier.process(ticket_state)
                # category classifier returns the updated state object
                ticket_state = category_result
                self._update_agent_metrics('category_classifier', time.time() - category_start, success=True)
                
            except Exception as e:
                self._update_agent_metrics('category_classifier', time.time() - category_start, success=False)
                self.logger.error(f"Category classification failed: {e}")
                # Set default category in classification object
                if not hasattr(ticket_state, 'classification') or ticket_state.classification is None:
                    from ..models.ticket_state import TicketClassification
                    ticket_state.classification = TicketClassification()
                ticket_state.classification.main_category = "عام"
                ticket_state.category_confidence = 0.1
                ticket_state.metadata['category_classification_error'] = str(e)
            
            # Step 4: Subcategory Classification - Detailed subcategory identification
            self.logger.info("Step 4: Subcategory classification")
            subcategory_start = time.time()
            
            try:
                subcategory_result = await self.subcategory_classifier.process(ticket_state)
                # subcategory classifier returns the updated state object
                ticket_state = subcategory_result
                self._update_agent_metrics('subcategory_classifier', time.time() - subcategory_start, success=True)
                
            except Exception as e:
                self._update_agent_metrics('subcategory_classifier', time.time() - subcategory_start, success=False)
                self.logger.error(f"Subcategory classification failed: {e}")
                # Set default subcategory in classification object
                if not hasattr(ticket_state, 'classification') or ticket_state.classification is None:
                    from ..models.ticket_state import TicketClassification
                    ticket_state.classification = TicketClassification()
                ticket_state.classification.subcategory = "عام"
                ticket_state.subcategory_confidence = 0.1
                ticket_state.metadata['subcategory_classification_error'] = str(e)
            
            # Finalize processing
            ticket_state.processing_completed = datetime.now()
            total_time = time.time() - start_time
            
            # Update metrics
            self.metrics['total_processed'] += 1
            self.metrics['successful_classifications'] += 1
            self._update_processing_time(total_time)
            
            # Prepare final result
            result = {
                'ticket_id': ticket_id,
                'classification': {
                    'category': ticket_state.classification.main_category if ticket_state.classification else None,
                    'category_confidence': ticket_state.category_confidence,
                    'subcategory': ticket_state.classification.subcategory if ticket_state.classification else None,
                    'subcategory_confidence': ticket_state.subcategory_confidence
                },
                'processing': {
                    'original_text': ticket_state.original_text,
                    'processed_text': ticket_state.processed_text,
                    'entities': ticket_state.entities,
                    'technical_terms': ticket_state.technical_terms,
                    'processing_time': total_time,
                    'started': ticket_state.processing_started.isoformat(),
                    'completed': ticket_state.processing_completed.isoformat()
                },
                'metadata': ticket_state.metadata,
                'success': True
            }
            
            self.logger.info(f"Ticket {ticket_id} classified successfully in {total_time:.2f}s")
            return result
            
        except Exception as e:
            # Handle complete pipeline failure
            self.metrics['total_processed'] += 1
            self.metrics['failed_classifications'] += 1
            
            error_result = {
                'ticket_id': ticket_id,
                'classification': {
                    'category': 'خطأ في المعالجة',
                    'category_confidence': 0.0,
                    'subcategory': 'خطأ نظام',
                    'subcategory_confidence': 0.0
                },
                'processing': {
                    'original_text': ticket_text,
                    'processed_text': ticket_text,
                    'entities': [],
                    'technical_terms': [],
                    'processing_time': time.time() - start_time,
                    'started': datetime.now().isoformat(),
                    'completed': datetime.now().isoformat()
                },
                'error': str(e),
                'success': False
            }
            
            self.logger.error(f"Pipeline failed for ticket {ticket_id}: {e}")
            return error_result
    
    def _update_agent_metrics(self, agent_name: str, execution_time: float, success: bool):
        """Update performance metrics for an agent."""
        metrics = self.metrics['agent_performance'][agent_name]
        metrics['calls'] += 1
        
        if not success:
            metrics['failures'] += 1
        
        # Update average time
        if metrics['calls'] == 1:
            metrics['avg_time'] = execution_time
        else:
            metrics['avg_time'] = (metrics['avg_time'] * (metrics['calls'] - 1) + execution_time) / metrics['calls']
    
    def _update_processing_time(self, total_time: float):
        """Update average processing time."""
        if self.metrics['total_processed'] == 1:
            self.metrics['average_processing_time'] = total_time
        else:
            current_avg = self.metrics['average_processing_time']
            count = self.metrics['total_processed']
            self.metrics['average_processing_time'] = (current_avg * (count - 1) + total_time) / count
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics for the pipeline."""
        success_rate = 0.0
        if self.metrics['total_processed'] > 0:
            success_rate = self.metrics['successful_classifications'] / self.metrics['total_processed']
        
        return {
            'overall': {
                'total_processed': self.metrics['total_processed'],
                'successful_classifications': self.metrics['successful_classifications'],
                'failed_classifications': self.metrics['failed_classifications'],
                'success_rate': success_rate,
                'average_processing_time': self.metrics['average_processing_time']
            },
            'agent_performance': self.metrics['agent_performance']
        }
    
    async def batch_classify(self, tickets: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
        """
        Process multiple tickets in batch.
        
        Args:
            tickets: List of (ticket_id, ticket_text) tuples
            
        Returns:
            List of classification results
        """
        results = []
        
        for ticket_id, ticket_text in tickets:
            try:
                result = await self.classify_ticket(ticket_text, ticket_id)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Failed to process ticket {ticket_id}: {e}")
                results.append({
                    'ticket_id': ticket_id,
                    'error': str(e),
                    'success': False
                })
        
        return results
    
    def track_classification_accuracy(self, expected: str, predicted: str, confidence: float):
        """Track classification accuracy for monitoring"""
        self.accuracy_stats['total'] += 1
        if expected == predicted:
            self.accuracy_stats['correct'] += 1
        
        if expected not in self.accuracy_stats['by_category']:
            self.accuracy_stats['by_category'][expected] = {'total': 0, 'correct': 0}
        
        self.accuracy_stats['by_category'][expected]['total'] += 1
        if expected == predicted:
            self.accuracy_stats['by_category'][expected]['correct'] += 1
        
        self.accuracy_stats['confidence_distribution'].append(confidence)
        
        # Log accuracy update
        overall_accuracy = self.accuracy_stats['correct'] / self.accuracy_stats['total'] * 100
        self.logger.debug(f"Classification accuracy updated: {overall_accuracy:.1f}%")
    
    def reset_metrics(self):
        """Reset all performance metrics."""
        self.metrics = {
            'total_processed': 0,
            'successful_classifications': 0,
            'failed_classifications': 0,
            'average_processing_time': 0.0,
            'agent_performance': {
                'orchestrator': {'calls': 0, 'failures': 0, 'avg_time': 0.0},
                'arabic_processing': {'calls': 0, 'failures': 0, 'avg_time': 0.0},
                'category_classifier': {'calls': 0, 'failures': 0, 'avg_time': 0.0},
                'subcategory_classifier': {'calls': 0, 'failures': 0, 'avg_time': 0.0}
            }
        }
        self.logger.info("Performance metrics reset")
