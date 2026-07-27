from moko_config import settings
from moko_agents.core_node import CoreNode
import time
import threading
from enum import Enum
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import numpy as np
from abc import ABC, abstractmethod

class PerformanceMode(Enum):
    CONSERVATION = "conservation"
    BALANCED = "balanced"
    PERFORMANCE = "performance"

class ResourcePressure(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class PerformanceMetrics:
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    gpu_usage: float
    vram_usage: float
    active_threads: int
    queue_depth: int
    response_time_ms: float
    throughput_tokens_per_sec: float
    error_rate: float

@dataclass
class ResourceState:
    cpu_capacity: float
    memory_capacity: float
    gpu_capacity: float
    vram_capacity: float
    active_memory_footprint: float
    estimated_available_time: float

class PerformanceOptimizer:
    def __init__(self, core_node: CoreNode):
        self.core_node = core_node
        self.performance_metrics: List[PerformanceMetrics] = []
        self.resource_state = ResourceState(
            cpu_capacity=100.0,
            memory_capacity=16384,
            gpu_capacity=100.0,
            vram_capacity=4096,
            active_memory_footprint=0.0,
            estimated_available_time=120.0
        )
        self.performance_mode = PerformanceMode.BALANCED
        self.current_pressure = ResourcePressure.LOW
        self.optimization_history: List[Dict[str, Any]] = []
        
        self._setup_performance_monitors()
        self._setup_optimization_strategies()
        
        print("  ✅ [PerformanceOptimizer] System initialized with adaptive optimization")
    
    def _setup_performance_monitors(self):
        """Setup performance monitoring systems"""
        self._monitor_metrics_thread = threading.Thread(
            target=self._background_performance_monitor,
            daemon=True
        )
        self._monitor_metrics_thread.start()
    
    def _setup_optimization_strategies(self):
        """Setup optimization strategies for different scenarios"""
        self.optimization_strategies = {
            PerformanceMode.CONSERVATION: {
                "methods": [
                    self._optimize_memory_usage,
                    self._reduce_computation_complexity,
                    self._enable_cached_responses,
                ],
                "priority": 3
            },
            PerformanceMode.BALANCED: {
                "methods": [
                    self._optimize_response_routing,
                    self._adaptive_thread_management,
                    self._medium_quality_caching,
                ],
                "priority": 2
            },
            PerformanceMode.PERFORMANCE: {
                "methods": [
                    self._maximize_throughput,
                    self._aggressive_caching,
                    self._enable_parallel_processing,
                ],
                "priority": 1
            }
        }
    
    def _background_performance_monitor(self):
        """Background thread for continuous performance monitoring"""
        while True:
            try:
                # Collect current system metrics
                metrics = self._collect_system_metrics()
                self.performance_metrics.append(metrics)
                
                # Update resource pressure assessment
                self._assess_resource_pressure(metrics)
                
                # Apply performance optimization based on pressure
                self._apply_performance_optimization()
                
                # Keep only recent metrics (last 10 minutes)
                cutoff_time = datetime.now() - timedelta(minutes=10)
                self.performance_metrics = [
                    m for m in self.performance_metrics 
                    if m.timestamp >= cutoff_time
                ]
                
                # Control monitoring frequency based on system load
                sleep_time = self._calculate_monitor_interval(metrics)
                time.sleep(sleep_time)
                
            except Exception as e:
                logging.error(f"Performance monitor error: {e}")
                time.sleep(1)
    
    def _collect_system_metrics(self) -> PerformanceMetrics:
        """Collect current system performance metrics"""
        # Mock metrics collection - in production, this would use system monitoring APIs
        import random
        
        return PerformanceMetrics(
            timestamp=datetime.now(),
            cpu_usage=random.uniform(20.0, 80.0),
            memory_usage=random.uniform(2048.0, 8192.0),
            gpu_usage=random.uniform(0.0, 100.0),
            vram_usage=random.uniform(0.0, 3072.0),
            active_threads=random.randint(1, 8),
            queue_depth=random.randint(0, 5),
            response_time_ms=random.uniform(50.0, 500.0),
            throughput_tokens_per_sec=random.uniform(50.0, 200.0),
            error_rate=random.uniform(0.0, 5.0)
        )
    
    def _assess_resource_pressure(self, metrics: PerformanceMetrics):
        """Assess current resource pressure based on metrics"""
        # Calculate pressure based on multiple factors
        pressure_score = 0.0
        
        # Memory pressure
        memory_pressure = metrics.memory_usage / self.resource_state.memory_capacity
        pressure_score += memory_pressure * 0.3
        
        # VRAM pressure
        vram_pressure = metrics.vram_usage / self.resource_state.vram_capacity
        pressure_score += vram_pressure * 0.25
        
        # Response time pressure
        response_time_pressure = min(1.0, metrics.response_time_ms / 100.0)
        pressure_score += response_time_pressure * 0.2
        
        # Error rate pressure
        error_pressure = min(1.0, metrics.error_rate / 10.0)
        pressure_score += error_pressure * 0.15
        
        # Queue depth pressure
        queue_pressure = min(1.0, metrics.queue_depth / 5.0)
        pressure_score += queue_pressure * 0.1
        
        # Determine resource pressure level
        if pressure_score < 0.3:
            self.current_pressure = ResourcePressure.LOW
        elif pressure_score < 0.6:
            self.current_pressure = ResourcePressure.MEDIUM
        elif pressure_score < 0.9:
            self.current_pressure = ResourcePressure.HIGH
        else:
            self.current_pressure = ResourcePressure.CRITICAL
    
    def _calculate_monitor_interval(self, metrics: PerformanceMetrics) -> float:
        """
        Calculate optimal monitoring interval based on system load
        
        Returns:
            Monitoring interval in seconds
        """
        base_interval = 5.0  # Base monitoring interval
        
        # Reduce interval under high pressure
        if self.current_pressure == ResourcePressure.CRITICAL:
            return 1.0
        elif self.current_pressure == ResourcePressure.HIGH:
            return 1.5
        elif self.current_pressure == ResourcePressure.MEDIUM:
            return 2.0
        else:
            return base_interval
    
    def _apply_performance_optimization(self):
        """Apply performance optimization based on current pressure"""
        if not self.performance_metrics:
            return
        
        # Get latest metrics
        latest_metrics = self.performance_metrics[-1]
        
        # Determine appropriate performance mode
        if self.current_pressure == ResourcePressure.CRITICAL:
            self.performance_mode = PerformanceMode.CONSERVATION
        elif self.current_pressure == ResourcePressure.HIGH:
            self.performance_mode = PerformanceMode.BALANCED
        elif self.current_pressure == ResourcePressure.MEDIUM:
            self.performance_mode = PerformanceMode.BALANCED
        else:
            self.performance_mode = PerformanceMode.PERFORMANCE
        
        # Apply optimization strategies for current mode
        strategy = self.optimization_strategies.get(self.performance_mode)
        if strategy:
            for method in strategy["methods"]:
                try:
                    result = method(latest_metrics)
                    if result:
                        self.optimization_history.append({
                            "timestamp": datetime.now(),
                            "strategy": self.performance_mode.value,
                            "method": method.__name__,
                            "result": result
                        })
                except Exception as e:
                    logging.error(f"Optimization method {method.__name__} failed: {e}")
    
    def _optimize_memory_usage(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Optimize memory usage based on current pressure"""
        optimizations = {}
        
        if metrics.memory_usage > self.resource_state.memory_capacity * 0.8:
            # Aggressive memory optimization
            optimizations["action"] = "aggressive_memory_cleanup"
            optimizations["estimated_savings_mb"] = metrics.memory_usage * 0.15
            self.resource_state.active_memory_footprint *= 0.85
        elif metrics.memory_usage > self.resource_state.memory_capacity * 0.6:
            # Moderate memory optimization
            optimizations["action"] = "moderate_cache_eviction"
            optimizations["estimated_savings_mb"] = metrics.memory_usage * 0.08
            self.resource_state.active_memory_footprint *= 0.92
        else:
            optimizations["action"] = "standard_maintenance"
            optimizations["estimated_savings_mb"] = metrics.memory_usage * 0.03
        
        return optimizations
    
    def _reduce_computation_complexity(self, metrics: PerformanceMetrics) -> Dict[str, Any]:n        # Mock implementation
        return {"action": "reduce_model_complexity", "estimated_savings_per_query_ms": 10.0}
    
    def _enable_cached_responses(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        return {"action": "increase_cache_hit_rate", "estimated_savings_per_query_ms": 5.0}
    
    def _optimize_response_routing(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        return {"action": "optimize_query_routing", "estimated_savings_per_query_ms": 8.0}
    
    def _adaptive_thread_management(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        return {"action": "adaptive_thread_pool", "estimated_savings_per_query_ms": 3.0}
    
    def _medium_quality_caching(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        return {"action": "medium_quality_cache", "estimated_savings_per_query_ms": 4.0}
    
    def _maximize_throughput(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        return {"action": "maximize_throughput", "estimated_savings_per_query_ms": 2.0}
    
    def _aggressive_caching(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        return {"action": "aggressive_caching", "estimated_savings_per_query_ms": 6.0}
    
    def _enable_parallel_processing(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        return {"action": "enable_parallel_processing", "estimated_savings_per_query_ms": 7.0}
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        if not self.performance_metrics:
            return {"status": "no_data", "message": "No performance metrics collected yet"}
        
        latest_metrics = self.performance_metrics[-1]
        
        # Calculate performance statistics
        avg_response_time = sum(m.response_time_ms for m in self.performance_metrics) / len(self.performance_metrics)
        avg_throughput = sum(m.throughput_tokens_per_sec for m in self.performance_metrics) / len(self.performance_metrics)
        avg_error_rate = sum(m.error_rate for m in self.performance_metrics) / len(self.performance_metrics)
        avg_cpu_usage = sum(m.cpu_usage for m in self.performance_metrics) / len(self.performance_metrics)
        avg_memory_usage = sum(m.memory_usage for m in self.performance_metrics) / len(self.performance_metrics)
        
        # Calculate pressure statistics
        pressure_counts = {p.value: 0 for p in ResourcePressure}
        for m in self.performance_metrics:
            pass  # Note: _assess_resource_pressure already called
        
        return {
            "overall_performance": {
                "average_response_time_ms": avg_response_time,
                "average_throughput_tokens_per_sec": avg_throughput,
                "average_error_rate": avg_error_rate,
                "average_cpu_usage": avg_cpu_usage,
                "average_memory_usage_mb": avg_memory_usage,
                "performance_grade": self._calculate_performance_grade(latest_metrics)
            },
            "current_state": {
                "performance_mode": self.performance_mode.value,
                "resource_pressure": self.current_pressure.value,
                "memory_footprint_mb": self.resource_state.active_memory_footprint,
                "available_time_min": self.resource_state.estimated_available_time
            },
            "optimization_history": self.optimization_history[-10:],  # Last 10 optimizations
            "resource_utilization": {
                "cpu": avg_cpu_usage / 100.0,
                "memory": avg_memory_usage / self.resource_state.memory_capacity,
                "active_time_percentage": (len([m for m in self.performance_metrics if m.response_time_ms < 100]) / len(self.performance_metrics)) * 100
            }
        }
    
    def _calculate_performance_grade(self, metrics: PerformanceMetrics) -> str:
        """Calculate overall performance grade based on metrics"""
        score = 0.0
        
        # Response time component
        response_score = max(0, 100 - metrics.response_time_ms) / 100
        score += response_score * 0.25
        
        # Throughput component
        throughput_score = min(100, metrics.throughput_tokens_per_sec / 10)
        score += throughput_score * 0.25
        
        # Error rate component
        error_score = max(0, 100 - metrics.error_rate * 20)
        score += error_score * 0.2
        
        # Resource efficiency component
        resource_score = 100 - (metrics.cpu_usage + metrics.memory_usage / self.resource_state.memory_capacity * 100)
        score += max(0, resource_score) * 0.15
        
        # Quality component
        quality_score = 100 - metrics.error_rate * 10
        score += max(0, quality_score) * 0.15
        
        if score >= 90:
            return "A+"
        elif score >= 85:
            return "A"
        elif score >= 80:
            return "B+"
        elif score >= 75:
            return "B"
        elif score >= 70:
            return "C+"
        elif score >= 60:
            return "C"
        elif score >= 50:
            return "D"
        else:
            return "F"
    
    def adjust_performance_mode(self, new_mode: PerformanceMode):
        """Manually set performance mode"""
        old_mode = self.performance_mode
        self.performance_mode = new_mode
        
        # Log performance mode change
        self.optimization_history.append({
            "timestamp": datetime.now(),
            "action": "performance_mode_change",
            "old_mode": old_mode.value,
            "new_mode": new_mode.value,
            "reason": "manual_adjustment"
        })
        
        print(f"  ✅ [PerformanceOptimizer] Performance mode changed from {old_mode.value} to {new_mode.value}")
    
    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Get recommendations for performance optimization"""
        recommendations = []
        
        if self.current_pressure == ResourcePressure.LOW:
            recommendations.extend([
                {"priority": "high", "recommendation": "Consider increasing caching to improve response times", "potential_improvement": "15-20%"},
                {"priority": "medium", "recommendation": "Monitor for gradual performance degradation", "potential_improvement": "minor"},
                {"priority": "low", "recommendation": "Perform baseline performance benchmarking", "potential_improvement": "5-10%"}
            ])
        
        elif self.current_pressure == ResourcePressure.MEDIUM:
            recommendations.extend([
                {"priority": "high", "recommendation": "Implement intelligent caching for repetitive queries", "potential_improvement": "25-30%"},
                {"priority": "medium", "recommendation": "Optimize thread pool configuration", "potential_improvement": "15-20%"},
                {"priority": "high", "recommendation": "Implement query prioritization", "potential_improvement": "20-25%"}
            ])
        
        elif self.current_pressure == ResourcePressure.HIGH:
            recommendations.extend([
                {"priority": "critical", "recommendation": "Implement emergency caching and response optimization", "potential_improvement": "35-40%"},
                {"priority": "high", "recommendation": "Use simpler models for complex queries", "potential_improvement": "30-35%"},
                {"priority": "critical", "recommendation": "Emergency resource cleanup and optimization", "potential_improvement": "40-45%"}
            ])
        
        else:  # CRITICAL
            recommendations.extend([
                {"priority": "critical", "recommendation": "Emergency response mode activated - reducing all non-essential operations", "potential_improvement": "50-60%"},
                {"priority": "critical", "recommendation": "Immediate resource preservation and cleanup", "potential_improvement": "45-50%"},
                {"priority": "high", "recommendation": "Implement fallback strategies for critical operations", "potential_improvement": "35-40%"}
            ])
        
        return recommendations

# Global performance optimizer service instance
service_instance = None

def get_performance_optimizer(core_node: CoreNode = None) -> PerformanceOptimizer:
    """
    Get or create global performance optimizer instance
    
    Args:
        core_node: CoreNode instance (required for initialization)
        
    Returns:
        PerformanceOptimizer instance
    """
    global service_instance
    
    if service_instance is None and core_node is not None:
        service_instance = PerformanceOptimizer(core_node)
    
    return service_instance

# Convenience functions for performance management
def get_system_performance():
    """Get current system performance metrics"""
    optimizer = get_performance_optimizer()
    if optimizer:
        return optimizer.get_performance_report()
    return {"error": "Performance optimizer not initialized"}

def adjust_performance_mode(mode: PerformanceMode):
    """Adjust system performance mode"""
    optimizer = get_performance_optimizer()
    if optimizer:
        optimizer.adjust_performance_mode(mode)
        return {"status": "success", "mode": mode.value}
    return {"error": "Performance optimizer not initialized"}

def get_optimization_recommendations():
    """Get performance optimization recommendations"""
    optimizer = get_performance_optimizer()
    if optimizer:
        return optimizer.get_optimization_recommendations()
    return {"error": "Performance optimizer not initialized"}