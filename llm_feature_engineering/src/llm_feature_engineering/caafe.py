import pandas as pd
import numpy as np
import logging
import re
import traceback
from typing import List, Dict, Any, Tuple, Optional
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from .llm_interface import LLMInterface
from .feature_selection import FeatureSelector

logger = logging.getLogger(__name__)

class CAAFESelector(FeatureSelector):
    """
    Context-Aware Automated Feature Engineering (CAAFE).
    Iteratively generates new features using an LLM and evaluates them.
    """
    
    def __init__(self, 
                 llm_interface: LLMInterface, 
                 dataset_description: str = "",
                 n_iterations: int = 10,
                 metric: str = 'roc_auc',
                 clf_params: Dict = None):
        super().__init__("caafe")
        self.llm = llm_interface
        self.dataset_description = dataset_description
        self.n_iterations = n_iterations
        self.metric = metric
        self.clf_params = clf_params or {'n_estimators': 100, 'random_state': 42}
        self.code_history: List[str] = []
        self.feature_history: List[Dict] = []
        
    def fit(self, X: pd.DataFrame, y: pd.Series, dataset_info: Dict = None, **kwargs):
        """
        Run the CAAFE iterative process.
        """
        self.original_features = X.columns.tolist()
        current_X = X.copy()
        current_score = self._evaluate_performance(current_X, y)
        logger.info(f"Initial {self.metric}: {current_score:.4f}")
        
        # Update dataset description if provided in dataset_info
        if dataset_info and 'description' in dataset_info:
            self.dataset_description = dataset_info['description']
            
        for i in range(self.n_iterations):
            logger.info(f"CAAFE Iteration {i+1}/{self.n_iterations}")
            
            # Generate prompt
            prompt = self._generate_prompt(current_X, y, i)
            
            # Get code from LLM
            try:
                response = self.llm.call_llm(prompt, max_tokens=1000, temperature=0.1)
                if response:
                    logger.info(f"LLM Response (first 500 chars): {response[:500]}")
                code_block = self._extract_code(response)
                if code_block:
                    logger.info(f"Extracted code block: {code_block[:200]}")
                else:
                    logger.warning("No code block extracted from response")
            except Exception as e:
                logger.error(f"LLM query failed: {e}")
                continue
                
            if not code_block:
                logger.warning("No code block found in LLM response.")
                continue
                
            # Execute code
            try:
                new_X, new_features = self._execute_code(current_X, code_block)
            except Exception as e:
                logger.warning(f"Code execution failed: {e}")
                # Ideally, we would feed the error back to the LLM in the next step
                continue
                
            # Evaluate
            new_score = self._evaluate_performance(new_X, y)
            logger.info(f"New {self.metric}: {new_score:.4f} (Previous: {current_score:.4f})")
            
            if new_score > current_score:
                logger.info("Performance improved. Keeping changes.")
                current_score = new_score
                current_X = new_X
                self.code_history.append(code_block)
                self.feature_history.append({
                    'iteration': i,
                    'code': code_block,
                    'score': new_score,
                    'added_features': new_features
                })
            else:
                logger.info("Performance did not improve. Discarding changes.")
                
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the learned transformations to new data.
        """
        current_X = X.copy()
        for code in self.code_history:
            try:
                current_X, _ = self._execute_code(current_X, code)
            except Exception as e:
                logger.error(f"Error applying transformation during transform: {e}")
        return current_X

    def select_features(self, X: pd.DataFrame, y: pd.Series, dataset_info: Dict = None, top_k: int = None) -> List[Dict]:
        """
        Wrapper for compatibility with the framework. 
        CAAFE generates features, it doesn't just select them.
        But we can return the list of generated features.
        """
        self.fit(X, y, dataset_info)
        
        # Return the features present in the final dataframe
        # Note: This might include original features if they weren't dropped
        final_X = self.transform(X)
        selected_features = []
        
        # Calculate importance for the final set of features
        clf = RandomForestClassifier(**self.clf_params)
        clf.fit(final_X, y)
        importances = clf.feature_importances_
        
        for name, importance in zip(final_X.columns, importances):
            selected_features.append({
                'name': name,
                'score': float(importance),
                'method': 'caafe',
                'generated': name not in self.original_features
            })
            
        selected_features.sort(key=lambda x: x['score'], reverse=True)
        if top_k:
            selected_features = selected_features[:top_k]
            
        return selected_features

    def _generate_prompt(self, df: pd.DataFrame, y: pd.Series, iteration: int) -> str:
        """
        Construct the prompt for the LLM based on the paper.
        """
        # Basic stats
        n_samples = len(df)
        columns_info = []
        for col in df.columns:
            dtype = df[col].dtype
            n_missing = df[col].isna().mean() * 100
            samples = df[col].dropna().sample(min(5, len(df))).tolist()
            columns_info.append(f"{col} ({dtype}): NaN-freq [{n_missing:.1f}%], Samples {samples}")
            
        columns_str = "\n".join(columns_info)
        
        prompt = f"""
The dataframe 'df' is loaded and in memory. Columns are also named attributes.
Description of the dataset in 'df':
"{self.dataset_description}"

Columns in 'df' (true feature dtypes listed here):
{columns_str}

This code was written by an expert data scientist working to improve predictions.
It is a snippet of code that adds new columns to the dataset.
Number of samples (rows) in training dataset: {n_samples}

This code generates additional columns that are useful for a downstream classification algorithm.
Additional columns add new semantic information, that is they use real world knowledge on the dataset.
They can e.g. be feature combinations, transformations, aggregations where the new column is a function of the existing columns.
The scale of columns and offset does not matter. Make sure all used columns exist.
Follow the above description of columns closely and consider the data types and meanings of classes.
This code also drops columns, if these may be redundant and hurt the predictive performance.

The classifier will be trained on the dataset with the generated columns and evaluated on a holdout set.
The evaluation metric is {self.metric}. The best performing code will be selected.
Added columns can be used in other code blocks, dropped columns are not available anymore.

Code formatting for each added column:
```python
# (Feature name and description)
# Usefulness: (Description why this adds useful real world knowledge)
# Input samples: (Three samples of the columns used)
# (Some pandas code using existing columns to add a new column for each row in df)
```

Code formatting for dropping columns:
```python
# Explanation why the column XX is dropped
df.drop(columns=['XX'], inplace=True)
```

Each code block generates exactly one useful column or drops unused columns.
Each code block ends with ``` and starts with ```python
Codeblock:
"""
        return prompt

    def _extract_code(self, response: str) -> Optional[str]:
        """
        Extract python code from markdown blocks.
        """
        match = re.search(r"```python\n(.*?)\n```", response, re.DOTALL)
        if match:
            return match.group(1)
        return None

    def _execute_code(self, df: pd.DataFrame, code: str) -> Tuple[pd.DataFrame, List[str]]:
        """
        Execute the generated code on a copy of the dataframe.
        Returns the new dataframe and a list of new feature names.
        """
        local_vars = {'df': df.copy(), 'pd': pd, 'np': np}
        
        # Safety check (very basic)
        forbidden = ['import os', 'import sys', 'open(', 'exec(', 'eval(']
        if any(f in code for f in forbidden):
            raise ValueError("Code contains forbidden operations.")
            
        exec(code, {}, local_vars)
        new_df = local_vars['df']
        
        # Identify new features
        new_features = [c for c in new_df.columns if c not in df.columns]
        
        return new_df, new_features

    def _evaluate_performance(self, X: pd.DataFrame, y: pd.Series) -> float:
        """
        Evaluate the current feature set using CV.
        """
        # Handle non-numeric data for the classifier
        # Simple encoding for evaluation purposes
        X_encoded = X.copy()
        for col in X_encoded.select_dtypes(include=['object', 'category']).columns:
            X_encoded[col] = pd.factorize(X_encoded[col])[0]
            
        X_encoded = X_encoded.fillna(0) # Simple imputation
        
        clf = RandomForestClassifier(**self.clf_params)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        
        try:
            scores = cross_val_score(clf, X_encoded, y, cv=cv, scoring=self.metric)
            return scores.mean()
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return 0.0
