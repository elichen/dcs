#!/usr/bin/env python3
"""
Session Registry - Process-wide tracking of active Fetch robot sessions.

This module provides a singleton registry that allows CLI tools to get direct
access to running MuJoCo environments without IPC overhead.
"""

import os
import json
import tempfile
import multiprocessing
from pathlib import Path
from typing import Dict, Any, Optional
import threading


class SessionRegistry:
    """Singleton registry for tracking active sessions and their environments."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.session_dir = Path(tempfile.gettempdir()) / "fetch-sessions"
        self.session_dir.mkdir(exist_ok=True)
        
        # Registry file for cross-process communication
        self.registry_file = self.session_dir / "registry.json"
        
        # Load existing registry
        self._load_registry()
    
    def _load_registry(self):
        """Load registry from file."""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r') as f:
                    data = json.load(f)
                    # Only load metadata, not the actual environment objects
                    for session_id, info in data.items():
                        self.sessions[session_id] = {
                            'metadata': info,
                            'env': None,  # Will be set when environment is registered
                            'pid': info.get('pid'),
                            'process_handle': None
                        }
            except Exception:
                # If registry is corrupted, start fresh
                self.sessions = {}
    
    def _save_registry(self):
        """Save registry metadata to file."""
        try:
            metadata = {}
            for session_id, info in self.sessions.items():
                metadata[session_id] = {
                    'session_id': session_id,
                    'pid': info.get('pid'),
                    'status': 'running' if info.get('env') else 'starting'
                }
            
            # Atomic write
            temp_file = self.registry_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            temp_file.replace(self.registry_file)
        except Exception as e:
            print(f"Warning: Could not save registry: {e}")
    
    def register_session(self, session_id: str, env=None, pid: int = None) -> bool:
        """
        Register a new session.
        
        Args:
            session_id: Unique session identifier
            env: MuJoCo environment object (optional, can be set later)
            pid: Process ID of the session
            
        Returns:
            True if successfully registered
        """
        if pid is None:
            pid = os.getpid()
            
        self.sessions[session_id] = {
            'env': env,
            'pid': pid,
            'metadata': {
                'session_id': session_id,
                'pid': pid,
                'status': 'running' if env else 'starting'
            }
        }
        
        self._save_registry()
        return True
    
    def set_environment(self, session_id: str, env, model, data) -> bool:
        """
        Set the environment object for an existing session.
        
        Args:
            session_id: Session ID
            env: MuJoCo environment
            model: MuJoCo model
            data: MuJoCo data
            
        Returns:
            True if successfully set
        """
        if session_id not in self.sessions:
            return False
            
        self.sessions[session_id]['env'] = env
        self.sessions[session_id]['model'] = model
        self.sessions[session_id]['data'] = data
        self.sessions[session_id]['metadata']['status'] = 'running'
        
        self._save_registry()
        return True
    
    def get_environment(self, session_id: str) -> tuple:
        """
        Get environment objects for a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            (env, model, data) or (None, None, None) if not found
        """
        if session_id not in self.sessions:
            # Try to reload registry in case it was updated by another process
            self._load_registry()
            
        session_info = self.sessions.get(session_id)
        if not session_info:
            return None, None, None
            
        env = session_info.get('env')
        model = session_info.get('model')
        data = session_info.get('data')
        
        return env, model, data
    
    def unregister_session(self, session_id: str) -> bool:
        """
        Unregister a session.
        
        Args:
            session_id: Session ID to remove
            
        Returns:
            True if successfully removed
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            self._save_registry()
            return True
        return False
    
    def list_sessions(self) -> Dict[str, Dict[str, Any]]:
        """
        List all registered sessions.
        
        Returns:
            Dictionary of session_id -> session_info
        """
        # Refresh from file first
        self._load_registry()
        
        # Check if processes are still alive
        active_sessions = {}
        for session_id, info in self.sessions.items():
            pid = info.get('pid')
            if pid:
                try:
                    os.kill(pid, 0)  # Check if process exists
                    active_sessions[session_id] = info
                except ProcessLookupError:
                    # Process is dead, will be cleaned up
                    pass
        
        # Update registry with only active sessions
        self.sessions = active_sessions
        self._save_registry()
        
        return {sid: info['metadata'] for sid, info in active_sessions.items()}
    
    def is_session_running(self, session_id: str) -> bool:
        """
        Check if a session is currently running.
        
        Args:
            session_id: Session ID to check
            
        Returns:
            True if session is active
        """
        return session_id in self.list_sessions()
    
    def cleanup_dead_sessions(self):
        """Remove registry entries for dead sessions."""
        self.list_sessions()  # This will clean up dead sessions as a side effect


# Global registry instance
_registry = None

def get_registry() -> SessionRegistry:
    """Get the global session registry instance."""
    global _registry
    if _registry is None:
        _registry = SessionRegistry()
    return _registry


def register_session(session_id: str, env=None, pid: int = None) -> bool:
    """Convenience function to register a session."""
    return get_registry().register_session(session_id, env, pid)


def get_environment(session_id: str) -> tuple:
    """Convenience function to get environment for a session."""
    return get_registry().get_environment(session_id)


def unregister_session(session_id: str) -> bool:
    """Convenience function to unregister a session."""
    return get_registry().unregister_session(session_id)