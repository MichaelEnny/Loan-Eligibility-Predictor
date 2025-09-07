"""
Training Progress Monitor
Provides real-time monitoring, logging, and visualization of model training progress
with performance tracking, resource utilization, and early stopping capabilities.
"""

import time
import logging
import json
from typing import Dict, Any, Optional, List, Callable, Union
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
import threading
from collections import deque

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TrainingStep:
    """Individual training step data."""
    step: int
    timestamp: datetime
    metrics: Dict[str, float]
    loss: Optional[float] = None
    learning_rate: Optional[float] = None
    duration_ms: Optional[float] = None


@dataclass
class TrainingSession:
    """Complete training session data."""
    session_id: str
    model_name: str
    model_type: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "running"
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    final_metrics: Dict[str, float] = field(default_factory=dict)
    steps: List[TrainingStep] = field(default_factory=list)
    total_duration_seconds: Optional[float] = None
    early_stopped: bool = False
    early_stop_reason: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'session_id': self.session_id,
            'model_name': self.model_name,
            'model_type': self.model_type,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'status': self.status,
            'hyperparameters': self.hyperparameters,
            'final_metrics': self.final_metrics,
            'total_duration_seconds': self.total_duration_seconds,
            'early_stopped': self.early_stopped,
            'early_stop_reason': self.early_stop_reason,
            'num_steps': len(self.steps)
        }


class MetricsBuffer:
    """Thread-safe metrics buffer for real-time monitoring."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()
    
    def add(self, step_data: TrainingStep):
        """Add training step data."""
        with self.lock:
            self.buffer.append(step_data)
    
    def get_recent(self, n: int = None) -> List[TrainingStep]:
        """Get recent training steps."""
        with self.lock:
            if n is None:
                return list(self.buffer)
            return list(self.buffer)[-n:] if len(self.buffer) >= n else list(self.buffer)
    
    def get_metrics_series(self, metric_name: str) -> List[float]:
        """Get time series for a specific metric."""
        with self.lock:
            return [step.metrics.get(metric_name, 0) for step in self.buffer 
                   if metric_name in step.metrics]
    
    def clear(self):
        """Clear the buffer."""
        with self.lock:
            self.buffer.clear()


class EarlyStopping:
    """Early stopping implementation with multiple criteria."""
    
    def __init__(self,
                 monitor: str = 'val_loss',
                 patience: int = 10,
                 min_delta: float = 0.001,
                 mode: str = 'min',
                 restore_best_weights: bool = True):
        """
        Initialize early stopping.
        
        Args:
            monitor: Metric to monitor
            patience: Number of epochs with no improvement to wait
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max' for metric optimization
            restore_best_weights: Whether to restore best weights
        """
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.wait = 0
        self.stopped_step = 0
        self.best_score = None
        self.best_weights = None
        
        if mode == 'min':
            self.monitor_op = np.less
            self.min_delta = -min_delta
        else:
            self.monitor_op = np.greater
            self.min_delta = min_delta
    
    def should_stop(self, metrics: Dict[str, float], weights=None) -> bool:
        """Check if training should stop."""
        if self.monitor not in metrics:
            return False
        
        current = metrics[self.monitor]
        
        if self.best_score is None:
            self.best_score = current
            if weights is not None and self.restore_best_weights:
                self.best_weights = weights
            return False
        
        if self.monitor_op(current - self.min_delta, self.best_score):
            self.best_score = current
            self.wait = 0
            if weights is not None and self.restore_best_weights:
                self.best_weights = weights
        else:
            self.wait += 1
            
        if self.wait >= self.patience:
            self.stopped_step = self.wait
            return True
        
        return False
    
    def get_best_weights(self):
        """Get the best weights if available."""
        return self.best_weights


class TrainingMonitor:
    """
    Comprehensive training progress monitor with real-time tracking.
    
    Provides real-time metrics monitoring, early stopping, progress visualization,
    resource tracking, and comprehensive logging for ML model training.
    """
    
    def __init__(self,
                 session_name: Optional[str] = None,
                 log_dir: str = "training_logs",
                 enable_early_stopping: bool = True,
                 early_stopping_config: Optional[Dict[str, Any]] = None,
                 save_frequency: int = 100,
                 buffer_size: int = 1000):
        """
        Initialize training monitor.
        
        Args:
            session_name: Training session name
            log_dir: Directory for training logs
            enable_early_stopping: Enable early stopping
            early_stopping_config: Early stopping configuration
            save_frequency: How often to save progress (in steps)
            buffer_size: Size of metrics buffer
        """
        self.session_name = session_name or f"training_{int(time.time())}"
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.enable_early_stopping = enable_early_stopping
        self.save_frequency = save_frequency
        
        # Create metrics buffer
        self.metrics_buffer = MetricsBuffer(max_size=buffer_size)
        
        # Create early stopping if enabled
        self.early_stopping = None
        if enable_early_stopping:
            es_config = early_stopping_config or {}
            self.early_stopping = EarlyStopping(**es_config)
        
        # Training session tracking
        self.current_session: Optional[TrainingSession] = None
        self.session_history: List[TrainingSession] = []
        
        # Callbacks
        self.callbacks: List[Callable] = []
        
        logger.info(f"Training monitor initialized: {self.session_name}")
    
    def start_session(self, 
                     model_name: str,
                     model_type: str,
                     hyperparameters: Dict[str, Any] = None) -> str:
        """
        Start a new training session.
        
        Args:
            model_name: Name of the model being trained
            model_type: Type of the model
            hyperparameters: Model hyperparameters
            
        Returns:
            Session ID
        """
        session_id = f"{self.session_name}_{int(time.time())}"
        
        self.current_session = TrainingSession(
            session_id=session_id,
            model_name=model_name,
            model_type=model_type,
            start_time=datetime.now(),
            hyperparameters=hyperparameters or {}
        )
        
        # Clear metrics buffer for new session
        self.metrics_buffer.clear()
        
        # Reset early stopping
        if self.early_stopping:
            self.early_stopping.wait = 0
            self.early_stopping.best_score = None
            self.early_stopping.best_weights = None
        
        logger.info(f"Started training session: {session_id}")
        return session_id
    
    def log_step(self, 
                step: int,
                metrics: Dict[str, float],
                loss: Optional[float] = None,
                learning_rate: Optional[float] = None,
                duration_ms: Optional[float] = None) -> bool:
        """
        Log a training step.
        
        Args:
            step: Step number
            metrics: Metrics dictionary
            loss: Training loss
            learning_rate: Current learning rate
            duration_ms: Step duration in milliseconds
            
        Returns:
            Whether training should continue (False if early stopping triggered)
        """
        if self.current_session is None:
            logger.warning("No active session. Call start_session() first.")
            return True
        
        # Create step data
        step_data = TrainingStep(
            step=step,
            timestamp=datetime.now(),
            metrics=metrics,
            loss=loss,
            learning_rate=learning_rate,
            duration_ms=duration_ms
        )
        
        # Add to buffer
        self.metrics_buffer.add(step_data)
        self.current_session.steps.append(step_data)
        
        # Check early stopping
        should_continue = True
        if self.early_stopping and self.enable_early_stopping:
            if self.early_stopping.should_stop(metrics):
                self.current_session.early_stopped = True
                self.current_session.early_stop_reason = f"Early stopping at step {step}"
                should_continue = False
                logger.info(f"Early stopping triggered at step {step}")
        
        # Save progress periodically
        if step % self.save_frequency == 0:
            self._save_session_progress()
        
        # Execute callbacks
        for callback in self.callbacks:
            try:
                callback(step_data)
            except Exception as e:
                logger.warning(f"Callback error: {e}")
        
        return should_continue
    
    def end_session(self, final_metrics: Dict[str, float] = None, status: str = "completed"):
        """
        End the current training session.
        
        Args:
            final_metrics: Final training metrics
            status: Session completion status
        """
        if self.current_session is None:
            logger.warning("No active session to end.")
            return
        
        # Update session
        self.current_session.end_time = datetime.now()
        self.current_session.status = status
        self.current_session.final_metrics = final_metrics or {}
        
        # Calculate total duration
        self.current_session.total_duration_seconds = (
            self.current_session.end_time - self.current_session.start_time
        ).total_seconds()
        
        # Add to history
        self.session_history.append(self.current_session)
        
        # Save final session data
        self._save_session_data(self.current_session)
        
        logger.info(f"Training session ended: {self.current_session.session_id}")
        logger.info(f"Total duration: {self.current_session.total_duration_seconds:.2f}s")
        
        # Clear current session
        self.current_session = None
    
    def get_current_metrics(self, window: int = 10) -> Dict[str, float]:
        """Get average metrics from recent steps."""
        recent_steps = self.metrics_buffer.get_recent(window)
        
        if not recent_steps:
            return {}
        
        # Calculate average metrics
        all_metrics = {}
        for step in recent_steps:
            for metric, value in step.metrics.items():
                if metric not in all_metrics:
                    all_metrics[metric] = []
                all_metrics[metric].append(value)
        
        # Calculate averages
        avg_metrics = {
            metric: np.mean(values) 
            for metric, values in all_metrics.items()
        }
        
        return avg_metrics
    
    def get_metric_history(self, metric_name: str) -> List[float]:
        """Get history for a specific metric."""
        return self.metrics_buffer.get_metrics_series(metric_name)
    
    def plot_training_curves(self, 
                            metrics: List[str] = None,
                            save_path: Optional[str] = None,
                            show_early_stop: bool = True):
        """Plot training curves for specified metrics."""
        try:
            import matplotlib.pyplot as plt
            
            if self.current_session is None and not self.session_history:
                logger.warning("No training data available for plotting")
                return
            
            # Use current session or latest from history
            session = self.current_session or self.session_history[-1]
            
            if not session.steps:
                logger.warning("No training steps available for plotting")
                return
            
            # Default metrics to plot
            if metrics is None:
                # Get all available metrics
                available_metrics = set()
                for step in session.steps:
                    available_metrics.update(step.metrics.keys())
                metrics = list(available_metrics)[:4]  # Limit to first 4
            
            # Create subplots
            n_metrics = len(metrics)
            if n_metrics == 0:
                logger.warning("No metrics to plot")
                return
            
            fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
            if n_metrics == 1:
                axes = [axes]
            
            # Plot each metric
            for i, metric in enumerate(metrics):
                values = []
                steps = []
                
                for step_data in session.steps:
                    if metric in step_data.metrics:
                        steps.append(step_data.step)
                        values.append(step_data.metrics[metric])
                
                if values:
                    axes[i].plot(steps, values, label=metric)
                    axes[i].set_title(f'Training {metric}')
                    axes[i].set_xlabel('Step')
                    axes[i].set_ylabel(metric)
                    axes[i].grid(True, alpha=0.3)
                    
                    # Mark early stopping point if applicable
                    if show_early_stop and session.early_stopped:
                        stop_step = session.steps[-1].step
                        axes[i].axvline(stop_step, color='red', linestyle='--', 
                                      label='Early Stop', alpha=0.7)
                    
                    axes[i].legend()
                else:
                    axes[i].text(0.5, 0.5, f'No data for {metric}', 
                               ha='center', va='center', transform=axes[i].transAxes)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Training curves saved to {save_path}")
            else:
                plt.show()
                
        except ImportError:
            logger.warning("matplotlib not available for plotting training curves")
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current or latest session."""
        session = self.current_session or (self.session_history[-1] if self.session_history else None)
        
        if session is None:
            return {}
        
        # Calculate step statistics
        step_durations = [step.duration_ms for step in session.steps if step.duration_ms]
        
        summary = {
            'session_id': session.session_id,
            'model_name': session.model_name,
            'model_type': session.model_type,
            'status': session.status,
            'start_time': session.start_time.isoformat(),
            'total_steps': len(session.steps),
            'early_stopped': session.early_stopped,
            'total_duration_seconds': session.total_duration_seconds,
            'final_metrics': session.final_metrics,
        }
        
        # Add step statistics if available
        if step_durations:
            summary['avg_step_duration_ms'] = np.mean(step_durations)
            summary['total_step_time_ms'] = np.sum(step_durations)
        
        # Add recent metrics
        if session.steps:
            latest_step = session.steps[-1]
            summary['latest_metrics'] = latest_step.metrics
            summary['latest_step'] = latest_step.step
        
        return summary
    
    def add_callback(self, callback: Callable[[TrainingStep], None]):
        """Add training callback."""
        self.callbacks.append(callback)
    
    def remove_callback(self, callback: Callable):
        """Remove training callback."""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    def _save_session_progress(self):
        """Save current session progress."""
        if self.current_session is None:
            return
        
        progress_file = self.log_dir / f"{self.current_session.session_id}_progress.json"
        
        progress_data = {
            'session_info': self.current_session.to_dict(),
            'current_metrics': self.get_current_metrics(),
            'total_steps': len(self.current_session.steps),
            'last_updated': datetime.now().isoformat()
        }
        
        try:
            with open(progress_file, 'w') as f:
                json.dump(progress_data, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Could not save session progress: {e}")
    
    def _save_session_data(self, session: TrainingSession):
        """Save complete session data."""
        session_file = self.log_dir / f"{session.session_id}_complete.json"
        
        # Create complete session data
        session_data = session.to_dict()
        
        # Add step details (sample for large sessions)
        if len(session.steps) > 1000:
            # Sample every 10th step for large sessions
            sampled_steps = session.steps[::10]
            logger.info(f"Sampling {len(sampled_steps)} steps from {len(session.steps)} total")
        else:
            sampled_steps = session.steps
        
        session_data['step_details'] = [
            {
                'step': step.step,
                'timestamp': step.timestamp.isoformat(),
                'metrics': step.metrics,
                'loss': step.loss,
                'learning_rate': step.learning_rate,
                'duration_ms': step.duration_ms
            }
            for step in sampled_steps
        ]
        
        try:
            with open(session_file, 'w') as f:
                json.dump(session_data, f, indent=2, default=str)
            logger.info(f"Session data saved to {session_file}")
        except Exception as e:
            logger.error(f"Could not save session data: {e}")
    
    def load_session_history(self) -> List[Dict[str, Any]]:
        """Load all session history from logs."""
        session_files = list(self.log_dir.glob("*_complete.json"))
        sessions = []
        
        for session_file in session_files:
            try:
                with open(session_file, 'r') as f:
                    session_data = json.load(f)
                sessions.append(session_data)
            except Exception as e:
                logger.warning(f"Could not load session {session_file}: {e}")
        
        # Sort by start time
        sessions.sort(key=lambda x: x.get('start_time', ''))
        
        return sessions
    
    def cleanup_old_logs(self, days_to_keep: int = 30):
        """Clean up old training logs."""
        cutoff_time = datetime.now().timestamp() - (days_to_keep * 24 * 3600)
        
        log_files = list(self.log_dir.glob("*.json"))
        cleaned_count = 0
        
        for log_file in log_files:
            if log_file.stat().st_mtime < cutoff_time:
                try:
                    log_file.unlink()
                    cleaned_count += 1
                except Exception as e:
                    logger.warning(f"Could not delete {log_file}: {e}")
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} old log files")


# Convenience classes for specific monitoring scenarios
class LoanEligibilityMonitor(TrainingMonitor):
    """Specialized monitor for loan eligibility models."""
    
    def __init__(self, **kwargs):
        # Default early stopping for loan models
        default_es_config = {
            'monitor': 'val_roc_auc',
            'patience': 20,
            'min_delta': 0.001,
            'mode': 'max',
            'restore_best_weights': True
        }
        
        kwargs['early_stopping_config'] = kwargs.get('early_stopping_config', default_es_config)
        super().__init__(**kwargs)
    
    def log_loan_metrics(self, 
                        step: int,
                        accuracy: float,
                        precision: float,
                        recall: float,
                        f1_score: float,
                        roc_auc: float,
                        **kwargs) -> bool:
        """Log loan-specific metrics."""
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'roc_auc': roc_auc
        }
        metrics.update(kwargs)
        
        return self.log_step(step, metrics)


# Factory function
def create_monitor(monitor_type: str = "standard", **kwargs) -> TrainingMonitor:
    """
    Create a training monitor of specified type.
    
    Args:
        monitor_type: Type of monitor ('standard', 'loan_eligibility')
        **kwargs: Additional monitor arguments
        
    Returns:
        Training monitor instance
    """
    if monitor_type == "loan_eligibility":
        return LoanEligibilityMonitor(**kwargs)
    else:
        return TrainingMonitor(**kwargs)