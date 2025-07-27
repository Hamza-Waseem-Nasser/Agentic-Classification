"""
STEP 2: STATE MANAGEMENT SYSTEM - STATE MANAGER

This file implements the StateManager class responsible for managing ticket state
throughout the multi-agent pipeline. It handles state persistence, loading,
validation, and provides a clean interface for agents to interact with state.

KEY CONCEPTS:
1. State Persistence: Save state to JSON files after each agent
2. State Loading: Load existing state from storage
3. State Validation: Validate state after each operation
4. Thread Safety: Handle concurrent access to state files
5. Error Recovery: Graceful handling of corrupted or missing state

DESIGN DECISIONS:
- JSON persistence: Human-readable, debuggable state storage
- File-based storage: Simple, reliable, no external dependencies
- Atomic writes: Prevent corruption during concurrent access
- Automatic backups: Keep previous versions for recovery
- Rich metadata: Track state history and performance

State lifecycle:
Create → Process (Agent 1) → Save → Process (Agent 2) → Save → ... → Complete
"""

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from threading import Lock
import tempfile
import uuid

from ..models.ticket_state import TicketState, AgentType, ProcessingStatus, ValidationResult
from ..models.entities import ClassificationHierarchy
from .state_validator import StateValidator, ValidationConfig


class StateManagerError(Exception):
    """Base exception for state manager errors"""
    pass


class StateNotFoundError(StateManagerError):
    """Raised when a ticket state is not found"""
    pass


class StateValidationError(StateManagerError):
    """Raised when state validation fails"""
    pass


class StateCorruptionError(StateManagerError):
    """Raised when state file is corrupted"""
    pass


class StateManager:
    """
    Manages ticket state throughout the multi-agent processing pipeline.
    
    The StateManager is responsible for:
    - Creating and initializing new ticket states
    - Persisting state to disk after each agent processing
    - Loading existing states from storage
    - Validating state integrity and business rules
    - Providing recovery mechanisms for corrupted states
    - Tracking processing history and performance metrics
    """
    
    def __init__(self, 
                 storage_path: str,
                 hierarchy: ClassificationHierarchy,
                 validation_config: Optional[ValidationConfig] = None,
                 enable_backups: bool = True,
                 max_backups: int = 5):
        """
        Initialize the StateManager.
        
        Args:
            storage_path: Directory path for storing state files
            hierarchy: Classification hierarchy for validation
            validation_config: Configuration for state validation
            enable_backups: Whether to create backup copies of state files
            max_backups: Maximum number of backup files to keep
        """
        self.storage_path = Path(storage_path)
        self.hierarchy = hierarchy
        self.enable_backups = enable_backups
        self.max_backups = max_backups
        
        # Create storage directory if it doesn't exist
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize validator
        self.validator = StateValidator(hierarchy, validation_config or ValidationConfig())
        
        # Thread safety for concurrent access
        self._locks: Dict[str, Lock] = {}
        self._global_lock = Lock()
        
        # Thread-safe performance tracking
        self._stats_lock = Lock()
        self.stats = {
            "states_created": 0,
            "states_loaded": 0,
            "states_saved": 0,
            "validation_failures": 0,
            "corruption_recoveries": 0
        }
    
    def create_ticket_state(self, 
                          original_text: str,
                          customer_id: Optional[str] = None,
                          priority: Optional[str] = None,
                          ticket_id: Optional[str] = None) -> TicketState:
        """
        Create a new ticket state and save it to storage.
        
        Args:
            original_text: The original ticket text
            customer_id: Optional customer identifier
            priority: Optional ticket priority
            ticket_id: Optional custom ticket ID (auto-generated if not provided)
            
        Returns:
            TicketState: The created and saved ticket state
            
        Raises:
            StateManagerError: If state creation or saving fails
        """
        try:
            # Generate ticket ID if not provided
            if not ticket_id:
                ticket_id = f"ticket_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
            
            # Create the ticket state
            state = TicketState(
                ticket_id=ticket_id,
                original_text=original_text,
                customer_id=customer_id,
                priority=priority
            )
            
            # Validate initial state
            validation_result = self.validator.validate_state(state, AgentType.ORCHESTRATOR)
            if not validation_result.is_valid:
                raise StateValidationError(f"Initial state validation failed: {validation_result.issues}")
            
            # Save the state
            self.save_state(state)
            
            self._increment_stat("states_created")
            return state
            
        except Exception as e:
            raise StateManagerError(f"Failed to create ticket state: {str(e)}") from e
    
    def load_state(self, ticket_id: str) -> TicketState:
        """
        Load a ticket state from storage.
        
        Args:
            ticket_id: The ticket ID to load
            
        Returns:
            TicketState: The loaded ticket state
            
        Raises:
            StateNotFoundError: If the state file doesn't exist
            StateCorruptionError: If the state file is corrupted
        """
        state_file = self._get_state_file_path(ticket_id)
        
        if not state_file.exists():
            raise StateNotFoundError(f"State file not found for ticket {ticket_id}")
        
        try:
            with open(state_file, 'r', encoding='utf-8') as f:
                state_data = json.load(f)
            
            # Convert ISO datetime strings back to datetime objects
            self._convert_datetime_strings(state_data)
            
            # Convert to TicketState object
            state = TicketState(**state_data)
            
            # Validate loaded state
            validation_result = self.validator.validate_state(state, AgentType.ORCHESTRATOR)
            if not validation_result.is_valid:
                # Try to load from backup if validation fails
                backup_state = self._try_load_from_backup(ticket_id)
                if backup_state:
                    self._increment_stat("corruption_recoveries")
                    return backup_state
                
                raise StateCorruptionError(f"Loaded state validation failed: {validation_result.issues}")
            
            self._increment_stat("states_loaded")
            return state
            
        except json.JSONDecodeError as e:
            # Try to load from backup
            backup_state = self._try_load_from_backup(ticket_id)
            if backup_state:
                self._increment_stat("corruption_recoveries")
                return backup_state
            
            raise StateCorruptionError(f"Failed to parse state file: {str(e)}") from e
        
        except Exception as e:
            raise StateManagerError(f"Failed to load state: {str(e)}") from e
    
    def save_state(self, state: TicketState, agent_type: Optional[AgentType] = None) -> None:
        """
        Save a ticket state to storage with validation.
        
        Args:
            state: The ticket state to save
            agent_type: The agent that processed the state (for validation)
            
        Raises:
            StateValidationError: If state validation fails
            StateManagerError: If saving fails
        """
        # Validate state before saving
        if agent_type:
            validation_result = self.validator.validate_state(state, agent_type)
            if not validation_result.is_valid:
                self._increment_stat("validation_failures")
                raise StateValidationError(f"State validation failed: {validation_result.issues}")
        
        # Update last modified timestamp
        state.last_updated = datetime.now()
        
        # Get thread-safe lock for this ticket
        with self._get_ticket_lock(state.ticket_id):
            try:
                # Create backup before saving
                if self.enable_backups:
                    self._create_backup(state.ticket_id)
                
                # Save state atomically
                self._save_state_atomic(state)
                
                self._increment_stat("states_saved")
                
            except Exception as e:
                raise StateManagerError(f"Failed to save state: {str(e)}") from e
    
    def update_agent_status(self, 
                          ticket_id: str,
                          agent_type: AgentType,
                          status: ProcessingStatus,
                          validation_result: Optional[ValidationResult] = None,
                          metadata: Optional[Dict[str, Any]] = None,
                          error_message: Optional[str] = None) -> TicketState:
        """
        Update the processing status for a specific agent and save the state.
        
        Args:
            ticket_id: The ticket ID
            agent_type: The agent type being updated
            status: The new processing status
            validation_result: Optional validation result
            metadata: Optional metadata to store
            error_message: Optional error message if status is FAILED
            
        Returns:
            TicketState: The updated state
            
        Raises:
            StateNotFoundError: If the ticket state doesn't exist
            StateManagerError: If updating fails
        """
        try:
            # Load current state
            state = self.load_state(ticket_id)
            
            # Update agent status
            state.update_agent_status(
                agent_type=agent_type,
                status=status,
                validation_result=validation_result,
                metadata=metadata,
                error_message=error_message
            )
            
            # Save updated state
            self.save_state(state, agent_type)
            
            return state
            
        except Exception as e:
            raise StateManagerError(f"Failed to update agent status: {str(e)}") from e
    
    def list_tickets(self, 
                    status_filter: Optional[ProcessingStatus] = None,
                    limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        List tickets with optional filtering.
        
        Args:
            status_filter: Optional status to filter by
            limit: Optional limit on number of results
            
        Returns:
            List of ticket summaries
        """
        tickets = []
        
        try:
            state_files = list(self.storage_path.glob("ticket_*.json"))
            state_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)  # Most recent first
            
            for state_file in state_files:
                if limit and len(tickets) >= limit:
                    break
                
                try:
                    with open(state_file, 'r', encoding='utf-8') as f:
                        state_data = json.load(f)
                    
                    # Apply status filter if specified
                    if status_filter and state_data.get('overall_status') != status_filter.value:
                        continue
                    
                    # Create summary
                    summary = {
                        "ticket_id": state_data.get('ticket_id'),
                        "created_at": state_data.get('created_at'),
                        "overall_status": state_data.get('overall_status'),
                        "classification": state_data.get('classification', {}),
                        "customer_id": state_data.get('customer_id'),
                        "priority": state_data.get('priority'),
                        "last_updated": state_data.get('last_updated')
                    }
                    tickets.append(summary)
                    
                except Exception:
                    # Skip corrupted files
                    continue
            
            return tickets
            
        except Exception as e:
            raise StateManagerError(f"Failed to list tickets: {str(e)}") from e
    
    def delete_state(self, ticket_id: str, delete_backups: bool = False) -> bool:
        """
        Delete a ticket state from storage.
        
        Args:
            ticket_id: The ticket ID to delete
            delete_backups: Whether to also delete backup files
            
        Returns:
            bool: True if deleted, False if not found
        """
        with self._get_ticket_lock(ticket_id):
            state_file = self._get_state_file_path(ticket_id)
            
            if not state_file.exists():
                return False
            
            try:
                # Delete main state file
                state_file.unlink()
                
                # Delete backups if requested
                if delete_backups:
                    backup_pattern = f"{ticket_id}_backup_*.json"
                    for backup_file in self.storage_path.glob(backup_pattern):
                        backup_file.unlink()
                
                return True
                
            except Exception as e:
                raise StateManagerError(f"Failed to delete state: {str(e)}") from e
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get state manager statistics"""
        total_files = len(list(self.storage_path.glob("ticket_*.json")))
        backup_files = len(list(self.storage_path.glob("*_backup_*.json")))
        
        return {
            **self.stats,
            "total_state_files": total_files,
            "total_backup_files": backup_files,
            "storage_path": str(self.storage_path)
        }
    
    def cleanup_old_backups(self, days_old: int = 7) -> int:
        """
        Clean up old backup files.
        
        Args:
            days_old: Delete backups older than this many days
            
        Returns:
            int: Number of files deleted
        """
        cutoff_time = datetime.now().timestamp() - (days_old * 24 * 60 * 60)
        deleted_count = 0
        
        try:
            backup_files = list(self.storage_path.glob("*_backup_*.json"))
            
            for backup_file in backup_files:
                if backup_file.stat().st_mtime < cutoff_time:
                    backup_file.unlink()
                    deleted_count += 1
            
            return deleted_count
            
        except Exception as e:
            raise StateManagerError(f"Failed to cleanup backups: {str(e)}") from e
    
    # Private helper methods
    
    def _get_state_file_path(self, ticket_id: str) -> Path:
        """Get the file path for a ticket state"""
        return self.storage_path / f"{ticket_id}.json"
    
    def _get_backup_file_path(self, ticket_id: str, timestamp: str) -> Path:
        """Get the file path for a backup"""
        return self.storage_path / f"{ticket_id}_backup_{timestamp}.json"
    
    def _get_ticket_lock(self, ticket_id: str) -> Lock:
        """Get or create a thread lock for a specific ticket"""
        with self._global_lock:
            if ticket_id not in self._locks:
                self._locks[ticket_id] = Lock()
            return self._locks[ticket_id]
    
    def _save_state_atomic(self, state: TicketState) -> None:
        """Save state atomically to prevent corruption"""
        state_file = self._get_state_file_path(state.ticket_id)
        
        # Convert state to JSON-serializable dict
        state_dict = state.dict()
        
        # Custom JSON encoder for datetime objects
        def json_serial(obj):
            """JSON serializer for objects not serializable by default json code"""
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Type {type(obj)} not serializable")
        
        # Write to temporary file first
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.json',
            dir=self.storage_path,
            delete=False,
            encoding='utf-8'
        ) as temp_file:
            json.dump(state_dict, temp_file, ensure_ascii=False, indent=2, default=json_serial)
            temp_file.flush()
            os.fsync(temp_file.fileno())  # Force write to disk
            temp_path = temp_file.name
        
        try:
            # Atomic move to final location
            shutil.move(temp_path, state_file)
        except Exception:
            # Clean up temp file if move fails
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise
    
    def _create_backup(self, ticket_id: str) -> None:
        """Create a backup of the current state file"""
        state_file = self._get_state_file_path(ticket_id)
        
        if not state_file.exists():
            return
        
        # Create backup with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_file = self._get_backup_file_path(ticket_id, timestamp)
        
        try:
            shutil.copy2(state_file, backup_file)
            
            # Clean up old backups if needed
            self._cleanup_ticket_backups(ticket_id)
            
        except Exception:
            # Don't fail the main operation if backup fails
            pass
    
    def _cleanup_ticket_backups(self, ticket_id: str) -> None:
        """Clean up old backups for a specific ticket"""
        backup_pattern = f"{ticket_id}_backup_*.json"
        backup_files = list(self.storage_path.glob(backup_pattern))
        
        if len(backup_files) > self.max_backups:
            # Sort by modification time (oldest first)
            backup_files.sort(key=lambda x: x.stat().st_mtime)
            
            # Delete oldest files
            files_to_delete = backup_files[:-self.max_backups]
            for backup_file in files_to_delete:
                try:
                    backup_file.unlink()
                except Exception:
                    pass
    
    def _try_load_from_backup(self, ticket_id: str) -> Optional[TicketState]:
        """Try to load state from the most recent backup"""
        backup_pattern = f"{ticket_id}_backup_*.json"
        backup_files = list(self.storage_path.glob(backup_pattern))
        
        if not backup_files:
            return None
        
        # Sort by modification time (most recent first)
        backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        for backup_file in backup_files:
            try:
                with open(backup_file, 'r', encoding='utf-8') as f:
                    state_data = json.load(f)
                
                # Convert datetime strings
                self._convert_datetime_strings(state_data)
                
                state = TicketState(**state_data)
                
                # Validate backup state
                validation_result = self.validator.validate_state(state, AgentType.ORCHESTRATOR)
                if validation_result.is_valid:
                    return state
                
            except Exception:
                continue
        
        return None
    
    def _convert_datetime_strings(self, data: Dict[str, Any]) -> None:
        """Convert ISO datetime strings back to datetime objects"""
        datetime_fields = [
            'created_at', 'last_updated'
        ]
        
        # Convert top-level datetime fields
        for field in datetime_fields:
            if field in data and isinstance(data[field], str):
                try:
                    data[field] = datetime.fromisoformat(data[field])
                except (ValueError, TypeError):
                    pass
        
        # Convert datetime fields in agent processing info
        if 'agent_processing' in data:
            for agent_info in data['agent_processing'].values():
                if isinstance(agent_info, dict):
                    for dt_field in ['started_at', 'completed_at']:
                        if dt_field in agent_info and isinstance(agent_info[dt_field], str):
                            try:
                                agent_info[dt_field] = datetime.fromisoformat(agent_info[dt_field])
                            except (ValueError, TypeError):
                                agent_info[dt_field] = None
    
    def _increment_stat(self, stat_name: str) -> None:
        """Thread-safe increment of statistics counter"""
        with self._stats_lock:
            self.stats[stat_name] += 1
    
    def cleanup_old_locks(self, inactive_hours: int = 24) -> int:
        """
        Clean up locks for tickets that haven't been accessed recently.
        
        Args:
            inactive_hours: Number of hours after which to consider a lock inactive
            
        Returns:
            Number of locks cleaned up
        """
        from datetime import datetime, timedelta
        
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(hours=inactive_hours)
        locks_cleaned = 0
        
        with self._global_lock:
            locks_to_remove = []
            
            for ticket_id, lock in self._locks.items():
                # Check if lock is not currently in use
                if not lock.locked():
                    # Check if state file exists and its modification time
                    state_file = self._get_state_file_path(ticket_id)
                    if not state_file.exists():
                        # No state file, safe to remove lock
                        locks_to_remove.append(ticket_id)
                    else:
                        # Check file modification time
                        modification_time = datetime.fromtimestamp(state_file.stat().st_mtime)
                        if modification_time < cutoff_time:
                            locks_to_remove.append(ticket_id)
            
            for ticket_id in locks_to_remove:
                del self._locks[ticket_id]
                locks_cleaned += 1
        
        return locks_cleaned
    
    def get_lock_stats(self) -> Dict[str, Any]:
        """Get statistics about current locks"""
        with self._global_lock:
            active_locks = sum(1 for lock in self._locks.values() if lock.locked())
            return {
                "total_locks": len(self._locks),
                "active_locks": active_locks,
                "inactive_locks": len(self._locks) - active_locks
            }
