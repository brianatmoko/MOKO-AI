from moko_config import settings
from moko_agents.core_node import CoreNode
import time
import traceback
import threading
from enum import Enum
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

class ErrorSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class FallbackMode(Enum):
    L0_DIRECT = "L0"
    L1_LLM_ONLY = "L1"
    L2_STANDARD = "L2"
    L3_BASIC = "L3"

class RecoveryStrategy(Enum):
    RETRY = "retry"
    FALLBACK = "fallback"
    SKIP = "skip"
    CIRCUIT_BREAKER = "circuit_breaker"

@dataclass
class ErrorInfo:
    error_type: str
    message: str
    severity: ErrorSeverity
    timestamp: datetime
    component: str
    stack_trace: Optional[str] = None
    recovery_attempted: bool = False
    recovery_success: bool = False
    retry_count: int = 0

class ErrorHandlingEngine:
    def __init__(self, core_node: CoreNode, max_retries: int = 3):
        self.core_node = core_node
        self.max_retries = max_retries
        self.error_history: List[ErrorInfo] = []
        self.health_status: Dict[str, bool] = {}
        self.recovery_strategies: Dict[str, Callable] = {}
        self.fallback_responses: Dict[FallbackMode, Callable] = {}
        self.circuit_breaker_state: Dict[str, Dict] = {}
        
        # Component health monitoring
        self.component_health: Dict[str, Dict] = {
            "llm_engine": {"healthy": True, "last_check": datetime.now(), "failure_count": 0},
            "omni_search": {"healthy": True, "last_check": datetime.now(), "failure_count": 0},
            "cognitive_router": {"healthy": True, "last_check": datetime.now(), "failure_count": 0},
            "memory_system": {"healthy": True, "last_check": datetime.now(), "failure_count": 0},
        }
        
        # Error recovery setup
        self._setup_error_recovery()
        self._setup_fallback_responses()
        
        print("  ✅ [ErrorHandlingEngine] System initialized with recovery mechanisms")
    
    def _setup_error_recovery(self):
        """Setup error recovery strategies for different components"""
        self.recovery_strategies = {
            "llm_engine": self._recover_llm_failure,
            "omni_search": self._recover_search_failure,
            "cognitive_router": self._recover_router_failure,
            "memory_system": self._recover_memory_failure,
        }
    
    def _setup_fallback_responses(self):
        """Setup fallback responses for different severity levels"""
        
        def l0_direct_response(question: str, **kwargs) -> str:
            return question  # Return original question for direct response
        
        def l1_llm_only_response(question: str, **kwargs) -> str:
            return f"[LLM Only] {question}"
        
        def l2_standard_response(question: str, **kwargs) -> str:
            return f"[Standard Mode] I understand your question about: {question[:50]}..."
        
        def l3_basic_response(question: str, **kwargs) -> str:
            return "[Basic Mode] I can help with basic information."
        
        self.fallback_responses = {
            FallbackMode.L0_DIRECT: l0_direct_response,
            FallbackMode.L1_LLM_ONLY: l1_llm_only_response,
            FallbackMode.L2_STANDARD: l2_standard_response,
            FallbackMode.L3_BASIC: l3_basic_response,
        }
    
    def monitor_system_health(self) -> Dict[str, Any]:
        """
        Comprehensive system health monitoring
        
        Returns:
            Dictionary with health status and metrics
        """
        health_report = {
            "timestamp": datetime.now().isoformat(),
            "overall_health": "healthy",
            "components": {},
            "errors": [],
            "performance": {}
        }
        
        for component, status in self.component_health.items():
            is_healthy = self._check_component_health(component)
            status["healthy"] = is_healthy
            status["last_check"] = datetime.now()
            
            health_report["components"][component] = {
                "healthy": is_healthy,
                "last_check": status["last_check"].isoformat(),
                "failure_count": status["failure_count"]
            }
            
            if not is_healthy:
                health_report["overall_health"] = "degraded"
        
        # Check for critical errors
        critical_errors = [e for e in self.error_history if e.severity == ErrorSeverity.CRITICAL]
        if critical_errors:
            health_report["overall_health"] = "critical"
        
        return health_report
    
    def _check_component_health(self, component: str) -> bool:
        """Check specific component health based on its type"""
        status = self.component_health.get(component, {})
        
        if component == "llm_engine":
            return self._check_llm_health()
        elif component == "omni_search":
            return self._check_search_health()
        elif component == "cognitive_router":
            return self._check_router_health()
        elif component == "memory_system":
            return self._check_memory_health()
        
        return status.get("healthy", True)
    
    def _check_llm_health(self) -> bool:
        """Check LLM engine health"""
        try:
            if hasattr(self.core_node, 'engine') and self.core_node.engine:
                return self.core_node.engine.is_available()
            return False
        except Exception:
            return False
    
    def _check_search_health(self) -> bool:
        """Check Omni search health"""
        try:
            return self.core_node.disk_mgr is not None
        except Exception:
            return False
    
    def _check_router_health(self) -> bool:
        """Check cognitive router health"""
        try:
            return hasattr(self.core_node, 'router') and self.core_node.router is not None
        except Exception:
            return False
    
    def _check_memory_health(self) -> bool:
        """Check memory system health"""
        try:
            return self.core_node.disk_mgr is not None
        except Exception:
            return False
    
    def handle_error(self, error_type: str, message: str, 
                    severity: ErrorSeverity = ErrorSeverity.ERROR,
                    component: str = "unknown",
                    exception: Optional[Exception] = None) -> Dict[str, Any]:
        """
        Handle system errors with recovery and logging
        
        Args:
            error_type: Type of error
            message: Error description
            severity: Error severity level
            component: Component where error occurred
            exception: Original exception object
            
        Returns:
            Dictionary with error handling results
        ""
        error_info = ErrorInfo(
            error_type=error_type,
            message=message,
            severity=severity,
            timestamp=datetime.now(),
            component=component,
            stack_trace=traceback.format_exc() if exception else None
        )
        
        # Update component health
        if component in self.component_health:
            self.component_health[component]["failure_count"] += 1
            if severity == ErrorSeverity.CRITICAL:
                self.component_health[component]["healthy"] = False
        
        # Log error
        self._log_error(error_info)
        
        # Attempt recovery
        recovery_result = self._attempt_recovery(error_info)
        
        # Record error history
        self.error_history.append(error_info)
        
        # Keep only last 1000 errors to prevent memory buildup
        if len(self.error_history) > 1000:
            self.error_history = self.error_history[-1000:]
        
        return {
            "error_info": {
                "type": error_info.error_type,
                "message": error_info.message,
                "severity": error_info.severity.value,
                "timestamp": error_info.timestamp.isoformat(),
                "component": error_info.component
            },
            "recovery_attempted": error_info.recovery_attempted,
            "recovery_success": error_info.recovery_success,
            "fallback_mode": recovery_result.get("fallback_mode") if recovery_result else None
        }
    
    def _log_error(self, error_info: ErrorInfo):
        """Log error based on severity"""
        if error_info.severity == ErrorSeverity.CRITICAL:
            logging.critical(f"[{error_info.component}] CRITICAL: {error_info.message}")
        elif error_info.severity == ErrorSeverity.ERROR:
            logging.error(f"[{error_info.component}] ERROR: {error_info.message}")
        elif error_info.severity == ErrorSeverity.WARNING:
            logging.warning(f"[{error_info.component}] WARNING: {error_info.message}")
        else:
            logging.info(f"[{error_info.component}] {error_info.message}")
    
    def _attempt_recovery(self, error_info: ErrorInfo) -> Optional[Dict]:
        """
        Attempt error recovery based on error type and component
        
        Returns:
            Recovery result or None if no recovery attempted
        """
        # Check circuit breaker state
        if self._is_circuit_breaker_tripped(error_info.component):
            logging.warning(f"[{error_info.component}] Circuit breaker tripped - skipping recovery")
            return {"fallback_mode": FallbackMode.L3_BASIC.value}
        
        # Attempt component-specific recovery
        recovery_func = self.recovery_strategies.get(error_info.component)
        if recovery_func:
            try:
                error_info.recovery_attempted = True
                result = recovery_func(error_info)
                error_info.recovery_success = True
                
                # Reset circuit breaker on successful recovery
                self._reset_circuit_breaker(error_info.component)
                
                if result:
                    return result
            except Exception as recovery_error:
                logging.error(f"[{error_info.component}] Recovery failed: {recovery_error}")
                error_info.recovery_success = False
        
        # Determine fallback mode based on severity
        fallback_mode = self._determine_fallback_mode(error_info)
        
        # Apply circuit breaker for repeated failures
        if self._should_trip_circuit_breaker(error_info.component):
            self._trip_circuit_breaker(error_info.component)
        
        return {"fallback_mode": fallback_mode.value}
    
    def _recover_llm_failure(self, error_info: ErrorInfo) -> Optional[Dict]:
        """Handle LLM engine failures"""
        if error_info.retry_count < self.max_retries:
            error_info.retry_count += 1
            time.sleep(0.5 ** error_info.retry_count)  # Exponential backoff
            
            if self._check_llm_health():
                return {"fallback_mode": FallbackMode.L0_DIRECT.value}
        
        return {"fallback_mode": FallbackMode.L1_LLM_ONLY.value}
    
    def _recover_search_failure(self, error_info: ErrorInfo) -> Optional[Dict]:
        """Handle search index failures"""
        try:
            # Try to rebuild or rebuild search index
            if hasattr(self.core_node, 'disk_mgr'):
                # Attempt to reinitialize search
                return {"fallback_mode": FallbackMode.L1_LLM_ONLY.value}
        except Exception:
            pass
        
        return {"fallback_mode": FallbackMode.L2_STANDARD.value}
    
    def _recover_router_failure(self, error_info: ErrorInfo) -> Optional[Dict]:
        """Handle router failures"""
        try:
            if hasattr(self.core_node, 'router'):
                # Reinitialize router
                return {"fallback_mode": FallbackMode.L1_LLM_ONLY.value}
        except Exception:
            pass
        
        return {"fallback_mode": FallbackMode.L2_STANDARD.value}
    
    def _recover_memory_failure(self, error_info: ErrorInfo) -> Optional[Dict]:
        """Handle memory system failures"""
        try:
            if hasattr(self.core_node, 'disk_mgr'):
                # Reinitialize memory system
                return {"fallback_mode": FallbackMode.L2_STANDARD.value}
        except Exception:
            pass
        
        return {"fallback_mode": FallbackMode.L3_BASIC.value}
    
    def apply_fallback_response(self, fallback_mode: FallbackMode, 
                              question: str, **kwargs) -> str:
        """
        Apply appropriate fallback response based on error conditions
        
        Args:
            fallback_mode: Fallback mode to apply
            question: Original user question
            **kwargs: Additional parameters for fallback function
            
        Returns:
            Fallback response
        """
        fallback_func = self.fallback_responses.get(fallback_mode)
        if fallback_func:
            try:
                return fallback_func(question, **kwargs)
            except Exception as e:
                logging.error(f"Fallback response failed: {e}")
                # Ultimate fallback
                return f"[System Response] {question}"
        
        return f"[Fallback] {question}"
    
    def _determine_fallback_mode(self, error_info: ErrorInfo) -> FallbackMode:
        """
        Determine appropriate fallback mode based on error type and severity
        
        Returns:
            Appropriate fallback mode
        """
        if error_info.severity == ErrorSeverity.CRITICAL:
            return FallbackMode.L3_BASIC
        
        retry_count = error_info.retry_count
        if retry_count <= 1:
            return FallbackMode.L2_STANDARD
        elif retry_count <= 3:
            return FallbackMode.L1_LLM_ONLY
        else:
            return FallbackMode.L0_DIRECT
    
    def _is_circuit_breaker_tripped(self, component: str) -> bool:
        """Check if circuit breaker is tripped for component"""
        circuit_state = self.circuit_breaker_state.get(component, {})
        return circuit_state.get("tripped", False)
    
    def _trip_circuit_breaker(self, component: str):
        """Trip circuit breaker for component (prevent repeated recovery attempts)"""
        self.circuit_breaker_state[component] = {
            "tripped": True,
            "trip_time": datetime.now(),
            "retry_after": datetime.now() + timedelta(seconds=30)
        }
    
    def _reset_circuit_breaker(self, component: str):
        """Reset circuit breaker for component"""
        if component in self.circuit_breaker_state:
            self.circuit_breaker_state[component]["tripped"] = False
    
    def _should_trip_circuit_breaker(self, component: str) -> bool:
        """
        Determine if circuit breaker should be tripped
        
        Trip when:
        - Multiple consecutive failures
        - Pattern of recovery not working
        """
        circuit_state = self.circuit_breaker_state.get(component, {})
        if circuit_state.get("tripped", False):
            return False  # Already tripped
        
        # Check number of consecutive failures
        failures = self.component_health.get(component, {}).get("failure_count", 0)
        return failures >= 5  # Trip after 5 failures
    
    def get_error_summary(self, hours_back: int = 24) -> Dict[str, Any]:
        """
        Generate comprehensive error summary for given time period
        
        Args:
            hours_back: Number of hours of error history to analyze
            
        Returns:
            Dictionary with error summary and statistics
        """
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        period_errors = [e for e in self.error_history 
                        if e.timestamp >= cutoff_time]
        
        # Group errors by type and component
        error_by_type = defaultdict(int)
        error_by_component = defaultdict(int)
        error_by_severity = defaultdict(int)
        
        for error in period_errors:
            error_by_type[error.error_type] += 1
            error_by_component[error.component] += 1
            error_by_severity[error.severity.value] += 1
        
        # Calculate recovery statistics
        attempted_recoveries = sum(1 for e in period_errors if e.recovery_attempted)
        successful_recoveries = sum(1 for e in period_errors if e.recovery_success)
        
        return {
            "period_hours": hours_back,
            "total_errors": len(period_errors),
            "error_by_type": dict(error_by_type),
            "error_by_component": dict(error_by_component),
            "error_by_severity": dict(error_by_severity),
            "recovery_attempts": attempted_recoveries,
            "successful_recoveries": successful_recoveries,
            "recovery_rate": (successful_recoveries / attempted_recoveries 
                            if attempted_recoveries > 0 else 0)
        }
    
    def reset_component_health(self, component: str = None):
        """
        Reset health status for specified component or all components
        
        Args:
            component: Component to reset. If None, reset all components
        """
        if component:
            if component in self.component_health:
                self.component_health[component] = {
                    "healthy": True,
                    "last_check": datetime.now(),
                    "failure_count": 0
                }
        else:
            for comp_name in self.component_health:
                self.component_health[comp_name] = {
                    "healthy": True,
                    "last_check": datetime.now(),
                    "failure_count": 0
                }
        
        # Clear circuit breaker states
        self.circuit_breaker_state.clear()
        
        print(f"  ✅ [ErrorHandlingEngine] Health reset {'for ' + component + ' component' if component else 'for all components'}")

# Global error handling service instance
service_instance = None

def get_error_handling_engine(core_node: CoreNode = None) -> ErrorHandlingEngine:
    """
    Get or create global error handling engine instance
    
    Args:
        core_node: CoreNode instance (required for initialization)
        
    Returns:
        ErrorHandlingEngine instance
    """
    global service_instance
    
    if service_instance is None and core_node is not None:
        service_instance = ErrorHandlingEngine(core_node)
    
    return service_instance

# Convenience functions for error handling
def handle_system_error(error_type: str, message: str,
                      severity: ErrorSeverity = ErrorSeverity.ERROR,
                      component: str = "system"):
    """Global error handling function"""
    engine = get_error_handling_engine()
    if engine:
        return engine.handle_error(error_type, message, severity, component)
    return {"error": "Error handling engine not initialized"}