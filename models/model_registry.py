"""
Model Registry and Versioning System
Provides comprehensive model artifact management, versioning, deployment tracking,
and metadata storage for production ML systems.
"""

import os
import time
import json
import pickle
import hashlib
import shutil
import logging
from typing import Dict, Any, Optional, Union, List, Tuple
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """Model lifecycle status."""
    TRAINING = "training"
    VALIDATION = "validation"  
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"
    DEPRECATED = "deprecated"


class ModelStage(Enum):
    """Model deployment stages."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    SHADOW = "shadow"  # A/B testing or shadow deployment
    CHAMPION = "champion"  # Best performing production model
    CHALLENGER = "challenger"  # Model being tested against champion


@dataclass
class ModelMetadata:
    """Comprehensive model metadata structure."""
    model_id: str
    model_name: str
    model_type: str
    version: str
    created_at: datetime
    created_by: str
    status: ModelStatus
    stage: ModelStage
    
    # Training metadata
    training_data_hash: str
    training_data_shape: Tuple[int, int]
    feature_names: List[str]
    target_classes: List[Any]
    hyperparameters: Dict[str, Any]
    
    # Performance metrics
    validation_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    cross_validation_metrics: Dict[str, float]
    
    # Model characteristics
    model_size_mb: float
    inference_time_ms: float
    training_time_seconds: float
    
    # Deployment info
    deployment_target: Optional[str] = None
    deployment_date: Optional[datetime] = None
    deployment_config: Optional[Dict[str, Any]] = None
    
    # Business metrics
    business_impact: Optional[Dict[str, Any]] = None
    a_b_test_results: Optional[Dict[str, Any]] = None
    
    # Lifecycle info
    parent_model_id: Optional[str] = None
    champion_model_id: Optional[str] = None
    approval_status: Optional[str] = None
    approved_by: Optional[str] = None
    approval_date: Optional[datetime] = None
    
    # Tags and description
    tags: List[str] = None
    description: Optional[str] = None
    notes: Optional[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with proper serialization."""
        data = asdict(self)
        
        # Convert enums to strings
        data['status'] = self.status.value
        data['stage'] = self.stage.value
        
        # Convert datetime objects
        for field in ['created_at', 'deployment_date', 'approval_date']:
            if data[field]:
                data[field] = data[field].isoformat()
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """Create from dictionary with proper deserialization."""
        # Convert string enums back to enums
        if 'status' in data and isinstance(data['status'], str):
            data['status'] = ModelStatus(data['status'])
        if 'stage' in data and isinstance(data['stage'], str):
            data['stage'] = ModelStage(data['stage'])
        
        # Convert datetime strings back to datetime objects
        for field in ['created_at', 'deployment_date', 'approval_date']:
            if data.get(field) and isinstance(data[field], str):
                data[field] = datetime.fromisoformat(data[field])
        
        return cls(**data)


class ModelRegistry:
    """
    Production-grade model registry with versioning and lifecycle management.
    
    Provides centralized storage, versioning, metadata management, and 
    deployment tracking for all machine learning models.
    """
    
    def __init__(self, 
                 registry_path: Union[str, Path] = "model_registry",
                 enable_compression: bool = True,
                 enable_checksums: bool = True,
                 max_versions_per_model: int = 10):
        """
        Initialize model registry.
        
        Args:
            registry_path: Root path for model registry
            enable_compression: Enable model artifact compression
            enable_checksums: Enable model integrity checking
            max_versions_per_model: Maximum versions to keep per model
        """
        self.registry_path = Path(registry_path)
        self.enable_compression = enable_compression
        self.enable_checksums = enable_checksums
        self.max_versions_per_model = max_versions_per_model
        
        # Create registry structure
        self._create_registry_structure()
        
        # Load existing models
        self.models = self._load_model_index()
        
        logger.info(f"Model registry initialized at {self.registry_path}")
        logger.info(f"Found {len(self.models)} existing models")
    
    def _create_registry_structure(self):
        """Create registry directory structure."""
        directories = [
            self.registry_path,
            self.registry_path / "models",
            self.registry_path / "metadata",
            self.registry_path / "artifacts",
            self.registry_path / "deployments",
            self.registry_path / "experiments",
            self.registry_path / "logs"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _load_model_index(self) -> Dict[str, ModelMetadata]:
        """Load existing model index."""
        index_path = self.registry_path / "model_index.json"
        models = {}
        
        if index_path.exists():
            try:
                with open(index_path, 'r') as f:
                    index_data = json.load(f)
                
                for model_id, model_data in index_data.items():
                    models[model_id] = ModelMetadata.from_dict(model_data)
                    
            except Exception as e:
                logger.warning(f"Could not load model index: {e}")
        
        return models
    
    def _save_model_index(self):
        """Save model index to disk."""
        index_path = self.registry_path / "model_index.json"
        
        try:
            index_data = {
                model_id: metadata.to_dict() 
                for model_id, metadata in self.models.items()
            }
            
            with open(index_path, 'w') as f:
                json.dump(index_data, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Could not save model index: {e}")
    
    def _generate_model_id(self, model_name: str, version: str = None) -> str:
        """Generate unique model ID."""
        if version is None:
            # Auto-generate version based on timestamp
            version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        model_id = f"{model_name}_{version}"
        
        # Ensure uniqueness
        counter = 1
        original_id = model_id
        while model_id in self.models:
            model_id = f"{original_id}_{counter}"
            counter += 1
        
        return model_id
    
    def _calculate_file_hash(self, filepath: Path) -> str:
        """Calculate SHA256 hash of a file."""
        hash_sha256 = hashlib.sha256()
        
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        
        return hash_sha256.hexdigest()
    
    def _calculate_data_hash(self, X: np.ndarray, y: np.ndarray) -> str:
        """Calculate hash of training data."""
        # Combine X and y and calculate hash
        data_str = f"{X.tobytes().hex()}{y.tobytes().hex()}"
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]  # Short hash
    
    def register_model(self,
                      model_trainer,
                      model_name: str,
                      version: str = None,
                      stage: ModelStage = ModelStage.DEVELOPMENT,
                      description: str = None,
                      tags: List[str] = None,
                      created_by: str = "system",
                      training_data: Tuple[np.ndarray, np.ndarray] = None,
                      **kwargs) -> str:
        """
        Register a trained model in the registry.
        
        Args:
            model_trainer: Trained model trainer instance
            model_name: Name of the model
            version: Model version (auto-generated if None)
            stage: Deployment stage
            description: Model description
            tags: Model tags
            created_by: Creator identifier
            training_data: Training data tuple (X, y) for hash calculation
            **kwargs: Additional metadata
            
        Returns:
            Generated model ID
        """
        if not model_trainer.is_trained:
            raise ValueError("Model must be trained before registration")
        
        # Generate model ID
        model_id = self._generate_model_id(model_name, version)
        
        # Calculate training data hash
        training_data_hash = ""
        training_data_shape = (0, 0)
        if training_data is not None:
            X, y = training_data
            training_data_hash = self._calculate_data_hash(X, y)
            training_data_shape = X.shape
        
        # Create model metadata
        metadata = ModelMetadata(
            model_id=model_id,
            model_name=model_name,
            model_type=model_trainer.model_name,
            version=version or model_id.split('_', 1)[1],
            created_at=datetime.now(),
            created_by=created_by,
            status=ModelStatus.VALIDATION,
            stage=stage,
            training_data_hash=training_data_hash,
            training_data_shape=training_data_shape,
            feature_names=model_trainer.training_metadata.get('feature_names', []),
            target_classes=model_trainer.training_metadata.get('target_classes', []),
            hyperparameters=model_trainer.training_metadata.get('hyperparameters', {}),
            validation_metrics=model_trainer.metrics.validation_scores,
            test_metrics=model_trainer.metrics.test_scores,
            cross_validation_metrics=model_trainer.metrics.cross_val_scores,
            model_size_mb=model_trainer.metrics.model_size_mb,
            inference_time_ms=model_trainer.metrics.inference_time_ms,
            training_time_seconds=model_trainer.metrics.training_time,
            tags=tags or [],
            description=description
        )
        
        # Save model artifacts
        model_dir = self.registry_path / "models" / model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model trainer
        model_path = model_dir / "model.pkl"
        model_trainer.save_model(str(model_path))
        
        # Calculate model file hash if checksums enabled
        if self.enable_checksums:
            model_hash = self._calculate_file_hash(model_path)
            
            # Save checksum
            checksum_path = model_dir / "checksum.txt"
            with open(checksum_path, 'w') as f:
                f.write(model_hash)
        
        # Save detailed metadata
        metadata_path = self.registry_path / "metadata" / f"{model_id}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2, default=str)
        
        # Register in index
        self.models[model_id] = metadata
        self._save_model_index()
        
        # Log registration
        self._log_event("MODEL_REGISTERED", {
            'model_id': model_id,
            'model_name': model_name,
            'model_type': model_trainer.model_name,
            'created_by': created_by
        })
        
        logger.info(f"Model registered: {model_id}")
        
        # Cleanup old versions if needed
        self._cleanup_old_versions(model_name)
        
        return model_id
    
    def get_model(self, model_id: str):
        """
        Load a registered model.
        
        Args:
            model_id: Model ID to load
            
        Returns:
            Loaded model trainer instance
        """
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found in registry")
        
        metadata = self.models[model_id]
        model_path = self.registry_path / "models" / model_id / "model.pkl"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Verify checksum if enabled
        if self.enable_checksums:
            self._verify_model_integrity(model_id)
        
        # Import the appropriate trainer class
        if metadata.model_type == "RandomForest":
            from .random_forest_model import RandomForestTrainer
            model_trainer = RandomForestTrainer.load_model(str(model_path))
        elif metadata.model_type == "XGBoost":
            from .xgboost_model import XGBoostTrainer
            model_trainer = XGBoostTrainer.load_model(str(model_path))
        elif metadata.model_type == "NeuralNetwork":
            from .neural_network_model import NeuralNetworkTrainer
            model_trainer = NeuralNetworkTrainer.load_model(str(model_path))
        else:
            from .base_trainer import BaseModelTrainer
            model_trainer = BaseModelTrainer.load_model(str(model_path))
        
        logger.info(f"Model loaded: {model_id}")
        return model_trainer
    
    def _verify_model_integrity(self, model_id: str):
        """Verify model file integrity using checksums."""
        model_dir = self.registry_path / "models" / model_id
        model_path = model_dir / "model.pkl"
        checksum_path = model_dir / "checksum.txt"
        
        if not checksum_path.exists():
            logger.warning(f"No checksum found for model {model_id}")
            return
        
        # Load expected checksum
        with open(checksum_path, 'r') as f:
            expected_hash = f.read().strip()
        
        # Calculate actual checksum
        actual_hash = self._calculate_file_hash(model_path)
        
        if actual_hash != expected_hash:
            raise ValueError(f"Model integrity check failed for {model_id}")
    
    def update_model_stage(self, 
                          model_id: str, 
                          new_stage: ModelStage, 
                          deployment_config: Dict[str, Any] = None,
                          approved_by: str = None):
        """
        Update model deployment stage.
        
        Args:
            model_id: Model ID to update
            new_stage: New deployment stage
            deployment_config: Deployment configuration
            approved_by: Approver identifier
        """
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        old_stage = self.models[model_id].stage
        
        # Update metadata
        self.models[model_id].stage = new_stage
        
        if new_stage == ModelStage.PRODUCTION:
            self.models[model_id].deployment_date = datetime.now()
            self.models[model_id].deployment_config = deployment_config
            self.models[model_id].approved_by = approved_by
            self.models[model_id].approval_date = datetime.now()
        
        # Save updates
        self._save_model_index()
        
        # Log stage change
        self._log_event("STAGE_CHANGED", {
            'model_id': model_id,
            'old_stage': old_stage.value,
            'new_stage': new_stage.value,
            'approved_by': approved_by
        })
        
        logger.info(f"Model {model_id} stage changed from {old_stage.value} to {new_stage.value}")
    
    def list_models(self, 
                   model_name: str = None,
                   model_type: str = None,
                   stage: ModelStage = None,
                   status: ModelStatus = None) -> pd.DataFrame:
        """
        List registered models with filtering.
        
        Args:
            model_name: Filter by model name
            model_type: Filter by model type
            stage: Filter by deployment stage
            status: Filter by model status
            
        Returns:
            DataFrame with model information
        """
        models_data = []
        
        for model_id, metadata in self.models.items():
            # Apply filters
            if model_name and metadata.model_name != model_name:
                continue
            if model_type and metadata.model_type != model_type:
                continue
            if stage and metadata.stage != stage:
                continue
            if status and metadata.status != status:
                continue
            
            models_data.append({
                'model_id': model_id,
                'model_name': metadata.model_name,
                'model_type': metadata.model_type,
                'version': metadata.version,
                'stage': metadata.stage.value,
                'status': metadata.status.value,
                'created_at': metadata.created_at,
                'created_by': metadata.created_by,
                'validation_accuracy': metadata.validation_metrics.get('val_accuracy', 0),
                'validation_f1': metadata.validation_metrics.get('val_f1_score', 0),
                'validation_auc': metadata.validation_metrics.get('val_roc_auc', 0),
                'model_size_mb': metadata.model_size_mb,
                'inference_time_ms': metadata.inference_time_ms,
                'tags': ', '.join(metadata.tags)
            })
        
        df = pd.DataFrame(models_data)
        
        if not df.empty:
            df = df.sort_values(['model_name', 'created_at'], ascending=[True, False])
        
        return df
    
    def get_model_metadata(self, model_id: str) -> ModelMetadata:
        """Get detailed model metadata."""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        return self.models[model_id]
    
    def compare_models(self, model_ids: List[str]) -> pd.DataFrame:
        """Compare multiple models."""
        if not model_ids:
            return pd.DataFrame()
        
        comparison_data = []
        
        for model_id in model_ids:
            if model_id not in self.models:
                logger.warning(f"Model {model_id} not found, skipping")
                continue
            
            metadata = self.models[model_id]
            
            comparison_data.append({
                'model_id': model_id,
                'model_name': metadata.model_name,
                'model_type': metadata.model_type,
                'stage': metadata.stage.value,
                'validation_accuracy': metadata.validation_metrics.get('val_accuracy', 0),
                'validation_f1': metadata.validation_metrics.get('val_f1_score', 0),
                'validation_auc': metadata.validation_metrics.get('val_roc_auc', 0),
                'test_accuracy': metadata.test_metrics.get('test_accuracy', 0),
                'test_f1': metadata.test_metrics.get('test_f1_score', 0),
                'test_auc': metadata.test_metrics.get('test_roc_auc', 0),
                'model_size_mb': metadata.model_size_mb,
                'inference_time_ms': metadata.inference_time_ms,
                'training_time_s': metadata.training_time_seconds,
                'created_at': metadata.created_at
            })
        
        df = pd.DataFrame(comparison_data)
        return df.sort_values('validation_accuracy', ascending=False)
    
    def promote_to_production(self, 
                             model_id: str,
                             approved_by: str,
                             deployment_config: Dict[str, Any] = None,
                             champion_replacement: bool = True):
        """
        Promote model to production with approval workflow.
        
        Args:
            model_id: Model ID to promote
            approved_by: Approver identifier
            deployment_config: Production deployment configuration
            champion_replacement: Whether to replace current champion
        """
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        metadata = self.models[model_id]
        
        # Check if model is ready for production
        if metadata.stage not in [ModelStage.STAGING, ModelStage.CHALLENGER]:
            raise ValueError(f"Model must be in staging or challenger stage for production promotion")
        
        # If replacing champion, demote current champion
        if champion_replacement:
            current_champions = [
                mid for mid, meta in self.models.items() 
                if meta.stage == ModelStage.CHAMPION and meta.model_name == metadata.model_name
            ]
            
            for champion_id in current_champions:
                self.update_model_stage(champion_id, ModelStage.PRODUCTION)
                logger.info(f"Demoted champion {champion_id} to production")
        
        # Promote to champion
        self.update_model_stage(
            model_id, 
            ModelStage.CHAMPION, 
            deployment_config, 
            approved_by
        )
        
        logger.info(f"Model {model_id} promoted to champion by {approved_by}")
    
    def archive_model(self, model_id: str, reason: str = None):
        """Archive a model."""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        # Update status
        self.models[model_id].status = ModelStatus.ARCHIVED
        if reason:
            self.models[model_id].notes = f"Archived: {reason}"
        
        self._save_model_index()
        
        # Log archival
        self._log_event("MODEL_ARCHIVED", {
            'model_id': model_id,
            'reason': reason
        })
        
        logger.info(f"Model {model_id} archived: {reason}")
    
    def delete_model(self, model_id: str, force: bool = False):
        """
        Delete a model from registry.
        
        Args:
            model_id: Model ID to delete
            force: Force deletion of production models
        """
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        metadata = self.models[model_id]
        
        # Safety check for production models
        if metadata.stage in [ModelStage.PRODUCTION, ModelStage.CHAMPION] and not force:
            raise ValueError(f"Cannot delete production model {model_id} without force=True")
        
        # Remove model files
        model_dir = self.registry_path / "models" / model_id
        if model_dir.exists():
            shutil.rmtree(model_dir)
        
        # Remove metadata file
        metadata_path = self.registry_path / "metadata" / f"{model_id}.json"
        if metadata_path.exists():
            metadata_path.unlink()
        
        # Remove from index
        del self.models[model_id]
        self._save_model_index()
        
        # Log deletion
        self._log_event("MODEL_DELETED", {
            'model_id': model_id,
            'model_name': metadata.model_name,
            'force': force
        })
        
        logger.info(f"Model {model_id} deleted")
    
    def _cleanup_old_versions(self, model_name: str):
        """Clean up old versions of a model."""
        # Get all versions of the model
        model_versions = [
            (model_id, metadata.created_at) 
            for model_id, metadata in self.models.items()
            if metadata.model_name == model_name
        ]
        
        # Sort by creation date (newest first)
        model_versions.sort(key=lambda x: x[1], reverse=True)
        
        # Keep only the specified number of versions
        if len(model_versions) > self.max_versions_per_model:
            models_to_remove = model_versions[self.max_versions_per_model:]
            
            for model_id, _ in models_to_remove:
                metadata = self.models[model_id]
                
                # Don't delete production models
                if metadata.stage not in [ModelStage.PRODUCTION, ModelStage.CHAMPION]:
                    try:
                        self.delete_model(model_id)
                        logger.info(f"Cleaned up old version: {model_id}")
                    except Exception as e:
                        logger.warning(f"Could not clean up {model_id}: {e}")
    
    def _log_event(self, event_type: str, event_data: Dict[str, Any]):
        """Log registry events."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'event_data': event_data
        }
        
        # Write to log file
        log_path = self.registry_path / "logs" / f"registry_{datetime.now().strftime('%Y%m')}.log"
        
        with open(log_path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def get_registry_statistics(self) -> Dict[str, Any]:
        """Get registry statistics."""
        total_models = len(self.models)
        
        # Count by status
        status_counts = {}
        for status in ModelStatus:
            status_counts[status.value] = sum(1 for m in self.models.values() if m.status == status)
        
        # Count by stage
        stage_counts = {}
        for stage in ModelStage:
            stage_counts[stage.value] = sum(1 for m in self.models.values() if m.stage == stage)
        
        # Count by model type
        type_counts = {}
        for metadata in self.models.values():
            type_counts[metadata.model_type] = type_counts.get(metadata.model_type, 0) + 1
        
        # Calculate total size
        total_size_mb = sum(metadata.model_size_mb for metadata in self.models.values())
        
        return {
            'total_models': total_models,
            'status_distribution': status_counts,
            'stage_distribution': stage_counts,
            'type_distribution': type_counts,
            'total_size_mb': total_size_mb,
            'registry_path': str(self.registry_path)
        }