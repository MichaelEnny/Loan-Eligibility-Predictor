"""
Duplicate Record Detection and Handling
Implements sophisticated duplicate detection with configurable strategies
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any, Tuple
import hashlib
from difflib import SequenceMatcher
import logging

logger = logging.getLogger(__name__)

class DuplicateHandler:
    """
    Comprehensive duplicate detection and handling
    """
    
    def __init__(self, 
                 strategy: str = 'exact',
                 threshold: float = 0.9,
                 key_columns: List[str] = None,
                 keep: str = 'first',
                 similarity_columns: List[str] = None):
        """
        Initialize duplicate handler
        
        Args:
            strategy: Detection strategy ('exact', 'fuzzy', 'semantic', 'custom')
            threshold: Similarity threshold for fuzzy matching (0-1)
            key_columns: Columns to use for duplicate detection (if None, use all)
            keep: Which duplicate to keep ('first', 'last', 'best')
            similarity_columns: Columns for fuzzy similarity calculation
        """
        self.strategy = strategy
        self.threshold = threshold
        self.key_columns = key_columns
        self.keep = keep
        self.similarity_columns = similarity_columns or []
        
        # Weights for different column types in similarity calculation
        self.column_weights = {
            'numeric': 1.0,
            'categorical': 1.0,
            'text': 0.8,
            'date': 1.2
        }
    
    def detect_duplicates(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect duplicate records in the dataset
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dict with duplicate detection results
        """
        if self.strategy == 'exact':
            return self._detect_exact_duplicates(df)
        elif self.strategy == 'fuzzy':
            return self._detect_fuzzy_duplicates(df)
        elif self.strategy == 'semantic':
            return self._detect_semantic_duplicates(df)
        else:
            return self._detect_exact_duplicates(df)  # Default fallback
    
    def remove_duplicates(self, df: pd.DataFrame, 
                         duplicate_info: Dict[str, Any] = None) -> pd.DataFrame:
        """
        Remove duplicate records from the dataset
        
        Args:
            df: Input DataFrame
            duplicate_info: Duplicate detection results (if None, detect first)
            
        Returns:
            DataFrame with duplicates removed
        """
        if duplicate_info is None:
            duplicate_info = self.detect_duplicates(df)
        
        duplicate_mask = duplicate_info['duplicate_mask']
        duplicate_groups = duplicate_info.get('groups', [])
        
        if not duplicate_mask.any():
            return df.copy()
        
        result_df = df.copy()
        removal_log = {
            'original_count': len(df),
            'duplicate_count': duplicate_mask.sum(),
            'removal_strategy': self.keep,
            'groups_processed': len(duplicate_groups)
        }
        
        if self.strategy == 'exact':
            # Use pandas built-in duplicate removal for exact matches
            columns_to_check = self.key_columns or df.columns.tolist()
            result_df = df.drop_duplicates(subset=columns_to_check, keep=self.keep)
        
        else:
            # Handle fuzzy/semantic duplicates group by group
            indices_to_remove = set()
            
            for group in duplicate_groups:
                if len(group) <= 1:
                    continue
                
                if self.keep == 'first':
                    # Keep first occurrence, mark others for removal
                    indices_to_remove.update(group[1:])
                elif self.keep == 'last':
                    # Keep last occurrence, mark others for removal
                    indices_to_remove.update(group[:-1])
                elif self.keep == 'best':
                    # Keep the "best" record based on completeness
                    best_idx = self._select_best_record(df, group)
                    indices_to_remove.update(idx for idx in group if idx != best_idx)
            
            # Remove marked indices
            result_df = df.drop(index=list(indices_to_remove))
        
        removal_log['final_count'] = len(result_df)
        removal_log['removed_count'] = removal_log['original_count'] - removal_log['final_count']
        
        logger.info(f"Duplicate removal summary: {removal_log}")
        
        return result_df.reset_index(drop=True)
    
    def detect_and_remove(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Detect and remove duplicates in one step
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (cleaned DataFrame, duplicate detection results)
        """
        duplicate_info = self.detect_duplicates(df)
        cleaned_df = self.remove_duplicates(df, duplicate_info)
        return cleaned_df, duplicate_info
    
    def _detect_exact_duplicates(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect exact duplicate records"""
        columns_to_check = self.key_columns or df.columns.tolist()
        
        # Find duplicated rows
        duplicate_mask = df.duplicated(subset=columns_to_check, keep=False)
        
        # Group duplicates
        duplicate_groups = []
        if duplicate_mask.any():
            # Create hash for grouping
            df_subset = df[columns_to_check]
            df_subset['_hash'] = df_subset.apply(
                lambda row: hashlib.md5(str(tuple(row)).encode()).hexdigest(), 
                axis=1
            )
            
            # Group by hash
            hash_groups = df_subset.groupby('_hash').groups
            for hash_key, indices in hash_groups.items():
                if len(indices) > 1:
                    duplicate_groups.append(list(indices))
        
        return {
            'strategy': 'exact',
            'duplicate_mask': duplicate_mask,
            'duplicate_count': duplicate_mask.sum(),
            'duplicate_percentage': (duplicate_mask.sum() / len(df)) * 100,
            'groups': duplicate_groups,
            'columns_checked': columns_to_check
        }
    
    def _detect_fuzzy_duplicates(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect fuzzy duplicate records using similarity threshold"""
        columns_to_check = self.similarity_columns or self.key_columns or df.columns.tolist()
        
        duplicate_mask = np.zeros(len(df), dtype=bool)
        duplicate_groups = []
        processed_indices = set()
        
        for i in range(len(df)):
            if i in processed_indices:
                continue
            
            current_group = [i]
            
            for j in range(i + 1, len(df)):
                if j in processed_indices:
                    continue
                
                similarity = self._calculate_similarity(
                    df.iloc[i][columns_to_check], 
                    df.iloc[j][columns_to_check]
                )
                
                if similarity >= self.threshold:
                    current_group.append(j)
                    duplicate_mask[j] = True
            
            if len(current_group) > 1:
                duplicate_groups.append(current_group)
                duplicate_mask[i] = True
                processed_indices.update(current_group)
        
        return {
            'strategy': 'fuzzy',
            'threshold': self.threshold,
            'duplicate_mask': duplicate_mask,
            'duplicate_count': duplicate_mask.sum(),
            'duplicate_percentage': (duplicate_mask.sum() / len(df)) * 100,
            'groups': duplicate_groups,
            'columns_checked': columns_to_check
        }
    
    def _detect_semantic_duplicates(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect semantic duplicates using advanced similarity"""
        # For now, use enhanced fuzzy matching
        # In a real implementation, this could use embeddings or ML models
        return self._detect_fuzzy_duplicates(df)
    
    def _calculate_similarity(self, row1: pd.Series, row2: pd.Series) -> float:
        """
        Calculate similarity between two rows
        
        Args:
            row1: First row
            row2: Second row
            
        Returns:
            Similarity score (0-1)
        """
        similarities = []
        weights = []
        
        for column in row1.index:
            if pd.isna(row1[column]) and pd.isna(row2[column]):
                similarity = 1.0  # Both missing
            elif pd.isna(row1[column]) or pd.isna(row2[column]):
                similarity = 0.0  # One missing
            else:
                similarity = self._calculate_column_similarity(
                    row1[column], row2[column], column
                )
            
            similarities.append(similarity)
            weights.append(self._get_column_weight(row1[column]))
        
        # Weighted average similarity
        if sum(weights) == 0:
            return 0.0
        
        return sum(s * w for s, w in zip(similarities, weights)) / sum(weights)
    
    def _calculate_column_similarity(self, val1: Any, val2: Any, column: str) -> float:
        """Calculate similarity between two values in a column"""
        if val1 == val2:
            return 1.0
        
        # Numeric similarity
        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            if val1 == 0 and val2 == 0:
                return 1.0
            max_val = max(abs(val1), abs(val2))
            if max_val == 0:
                return 1.0
            return 1.0 - abs(val1 - val2) / max_val
        
        # String similarity
        if isinstance(val1, str) and isinstance(val2, str):
            return SequenceMatcher(None, val1.lower(), val2.lower()).ratio()
        
        # Different types - convert to string and compare
        str1, str2 = str(val1).lower(), str(val2).lower()
        return SequenceMatcher(None, str1, str2).ratio()
    
    def _get_column_weight(self, value: Any) -> float:
        """Get weight for a column based on its data type"""
        if isinstance(value, (int, float)):
            return self.column_weights.get('numeric', 1.0)
        elif isinstance(value, str):
            return self.column_weights.get('text', 0.8)
        elif pd.api.types.is_datetime64_any_dtype(type(value)):
            return self.column_weights.get('date', 1.2)
        else:
            return self.column_weights.get('categorical', 1.0)
    
    def _select_best_record(self, df: pd.DataFrame, indices: List[int]) -> int:
        """
        Select the best record from a group of duplicates
        
        Args:
            df: DataFrame
            indices: List of duplicate indices
            
        Returns:
            Index of the best record
        """
        if len(indices) == 1:
            return indices[0]
        
        # Score records by completeness and quality
        scores = []
        
        for idx in indices:
            row = df.iloc[idx]
            score = 0
            
            # Completeness score (non-null values)
            non_null_count = row.notna().sum()
            completeness_score = non_null_count / len(row)
            score += completeness_score * 0.5
            
            # Quality score (prefer more recent, higher values for certain fields)
            # This is domain-specific and can be customized
            if 'created_at' in df.columns or 'date' in df.columns:
                date_col = 'created_at' if 'created_at' in df.columns else 'date'
                if pd.notna(row[date_col]):
                    # Prefer more recent records
                    date_score = 0.3  # Base score for having a date
                    score += date_score
            
            # Prefer records with higher ID values (more recent in auto-increment systems)
            if 'id' in df.columns and pd.notna(row['id']):
                max_id = df['id'].max()
                if max_id > 0:
                    id_score = (row['id'] / max_id) * 0.2
                    score += id_score
            
            scores.append(score)
        
        # Return index with highest score
        best_position = np.argmax(scores)
        return indices[best_position]
    
    def get_duplicate_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get comprehensive summary of duplicates in the dataset
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with duplicate statistics
        """
        duplicate_info = self.detect_duplicates(df)
        
        summary_data = {
            'Total_Records': len(df),
            'Duplicate_Records': duplicate_info['duplicate_count'],
            'Duplicate_Percentage': duplicate_info['duplicate_percentage'],
            'Duplicate_Groups': len(duplicate_info['groups']),
            'Strategy_Used': duplicate_info['strategy'],
            'Columns_Checked': len(duplicate_info['columns_checked'])
        }
        
        if self.strategy == 'fuzzy':
            summary_data['Similarity_Threshold'] = self.threshold
        
        return pd.DataFrame([summary_data])
    
    def analyze_duplicate_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze patterns in duplicate records
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dict with duplicate pattern analysis
        """
        duplicate_info = self.detect_duplicates(df)
        
        if not duplicate_info['groups']:
            return {
                'no_duplicates': True,
                'total_records': len(df)
            }
        
        # Analyze group sizes
        group_sizes = [len(group) for group in duplicate_info['groups']]
        
        # Analyze duplicate fields
        field_duplicate_counts = {}
        columns_to_check = duplicate_info['columns_checked']
        
        for group in duplicate_info['groups']:
            if len(group) < 2:
                continue
            
            # Check which fields are actually duplicated within the group
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    row1 = df.iloc[group[i]]
                    row2 = df.iloc[group[j]]
                    
                    for col in columns_to_check:
                        if pd.notna(row1[col]) and pd.notna(row2[col]) and row1[col] == row2[col]:
                            field_duplicate_counts[col] = field_duplicate_counts.get(col, 0) + 1
        
        return {
            'total_records': len(df),
            'duplicate_groups': len(duplicate_info['groups']),
            'duplicate_records': duplicate_info['duplicate_count'],
            'group_sizes': {
                'min': min(group_sizes) if group_sizes else 0,
                'max': max(group_sizes) if group_sizes else 0,
                'mean': np.mean(group_sizes) if group_sizes else 0,
                'sizes_distribution': pd.Series(group_sizes).value_counts().to_dict() if group_sizes else {}
            },
            'field_duplicate_frequency': field_duplicate_counts,
            'most_duplicated_fields': sorted(field_duplicate_counts.items(), 
                                           key=lambda x: x[1], reverse=True)[:5]
        }
    
    def get_duplicate_examples(self, df: pd.DataFrame, n_examples: int = 5) -> pd.DataFrame:
        """
        Get examples of duplicate records for inspection
        
        Args:
            df: Input DataFrame
            n_examples: Number of duplicate groups to return
            
        Returns:
            DataFrame with example duplicates
        """
        duplicate_info = self.detect_duplicates(df)
        
        if not duplicate_info['groups']:
            return pd.DataFrame()
        
        example_groups = duplicate_info['groups'][:n_examples]
        example_indices = []
        
        for group in example_groups:
            example_indices.extend(group)
        
        examples_df = df.iloc[example_indices].copy()
        
        # Add group identifier
        group_id = 0
        for group in example_groups:
            for idx in group:
                examples_df.loc[examples_df.index[example_indices.index(idx)], 'duplicate_group'] = group_id
            group_id += 1
        
        return examples_df.sort_values('duplicate_group')