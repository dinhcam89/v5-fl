"""
GA-Stacking implementation for federated learning.
Provides genetic algorithm-based ensemble stacking optimization.
Optimized for performance with credit card fraud detection datasets.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import logging
import random
import time
from typing import List, Dict, Any, Optional, Tuple, Union
from copy import deepcopy
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# Conditional imports for specialized models
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("GA-Stacking")

# Base Model Wrapper
class BaseModelWrapper(nn.Module):
    """Base wrapper for sklearn models to be used with PyTorch."""
    
    def __init__(self, model_type: str, model, input_dim: int, output_dim: int):
        super(BaseModelWrapper, self).__init__()
        self.model_type = model_type
        self.model = model
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.is_initialized = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using sklearn model's predict_proba."""
        # Convert to numpy if it's a torch tensor
        if isinstance(x, torch.Tensor):
            x_np = x.cpu().detach().numpy()
            device = x.device
        else:
            x_np = x
            device = torch.device("cpu")
            
        try:
            # Make sure model is initialized/fitted
            if not self.is_initialized:
                logger.warning(f"{self.model_type} model not initialized, returning zeros")
                zeros = np.zeros((x_np.shape[0], self.output_dim))
                return torch.tensor(zeros, dtype=torch.float32).to(device)
                
            # Check input shape
            if x_np.shape[1] != self.input_dim:
                logger.warning(f"Input dimension mismatch: got {x_np.shape[1]}, expected {self.input_dim}")
                # Try to handle dimension mismatch by adding zeros or taking subset
                if x_np.shape[1] < self.input_dim:
                    # Pad with zeros
                    padding = np.zeros((x_np.shape[0], self.input_dim - x_np.shape[1]))
                    x_np = np.hstack([x_np, padding])
                else:
                    # Take subset
                    x_np = x_np[:, :self.input_dim]
                
            # Get probabilities
            y_prob = self.model.predict_proba(x_np)
            
            # For binary classification, we only need the probability of class 1
            if y_prob.shape[1] == 2:
                y_prob = y_prob[:, 1:2]
                
            # Convert back to tensor
            return torch.tensor(y_prob, dtype=torch.float32).to(device)
            
        except Exception as e:
            logger.error(f"Error in {self.model_type} forward pass: {e}")
            # Return zeros in case of error
            zeros = np.zeros((x_np.shape[0], self.output_dim))
            return torch.tensor(zeros, dtype=torch.float32).to(device)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the underlying sklearn model."""
        try:
            self.model.fit(X, y)
            self.is_initialized = True
        except Exception as e:
            logger.error(f"Error fitting {self.model_type} model: {e}")
            
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        try:
            params = {
                "model_type": self.model_type,
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
            }
            # Add model params if available
            if hasattr(self.model, "get_params"):
                params["model_params"] = self.model.get_params()
            return params
        except Exception as e:
            logger.error(f"Error getting parameters: {e}")
            return {"model_type": self.model_type}
    
    def set_params(self, params: Dict[str, Any]) -> None:
        """Set model parameters."""
        try:
            if "model_params" in params and hasattr(self.model, "set_params"):
                self.model.set_params(**params["model_params"])
            if "input_dim" in params:
                self.input_dim = params["input_dim"]
            if "output_dim" in params:
                self.output_dim = params["output_dim"]
        except Exception as e:
            logger.error(f"Error setting parameters: {e}")

# Meta-learner wrapper for stacking
class MetaLearnerWrapper(BaseModelWrapper):
    """Meta-learner wrapper for stacking ensemble."""
    
    def __init__(self, input_dim: int, output_dim: int):
        # Use LogisticRegression as the meta-learner
        meta_model = LogisticRegression(solver='liblinear')
        super(MetaLearnerWrapper, self).__init__(
            model_type="meta_lr",
            model=meta_model,
            input_dim=input_dim,
            output_dim=output_dim
        )

# Create ensemble of base models
def create_base_models(input_dim: int, output_dim: int) -> Tuple[List[BaseModelWrapper], List[str]]:
    """
    Create ensemble of base models.
    
    Args:
        input_dim: Input dimension
        output_dim: Output dimension
        
    Returns:
        Tuple of (list of models, list of model names)
    """
    base_models = []
    model_names = []
    
    # LogisticRegression
    lr = LogisticRegression(solver='liblinear')
    base_models.append(BaseModelWrapper("lr", lr, input_dim, output_dim))
    model_names.append("lr")
    
    # SVC
    svc = SVC(probability=True)
    base_models.append(BaseModelWrapper("svc", svc, input_dim, output_dim))
    model_names.append("svc")
    
    # RandomForest
    rf = RandomForestClassifier(n_estimators=100)
    base_models.append(BaseModelWrapper("rf", rf, input_dim, output_dim))
    model_names.append("rf")
    
    # KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    base_models.append(BaseModelWrapper("knn", knn, input_dim, output_dim))
    model_names.append("knn")
    
    # XGBoost
    if XGBOOST_AVAILABLE:
        xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        base_models.append(BaseModelWrapper("xgb", xgb, input_dim, output_dim))
        model_names.append("xgb")
    
    # LightGBM
    if LIGHTGBM_AVAILABLE:
        lgbm = LGBMClassifier()
        base_models.append(BaseModelWrapper("lgbm", lgbm, input_dim, output_dim))
        model_names.append("lgbm")
    
    # CatBoost
    if CATBOOST_AVAILABLE:
        catboost = CatBoostClassifier(verbose=0)
        base_models.append(BaseModelWrapper("catboost", catboost, input_dim, output_dim))
        model_names.append("catboost")
    
    # Meta-learner
    meta_lr = MetaLearnerWrapper(len(base_models), output_dim)
    base_models.append(meta_lr)
    model_names.append("meta_lr")
    
    return base_models, model_names

# Ensemble Model
class EnsembleModel(nn.Module):
    """Ensemble model that combines predictions from multiple base models."""
    
    def __init__(
        self, 
        models: List[nn.Module], 
        weights: Optional[Union[List[float], np.ndarray, torch.Tensor]] = None,
        model_names: Optional[List[str]] = None,
        device: Optional[torch.device] = None
    ):
        super(EnsembleModel, self).__init__()
        self.models = models
        
        # Convert weights to tensor if needed
        if weights is None:
            # Equal weights
            weights = torch.ones(len(models)) / len(models)
        elif isinstance(weights, list):
            weights = torch.tensor(weights, dtype=torch.float32)
        elif isinstance(weights, np.ndarray):
            weights = torch.tensor(weights, dtype=torch.float32)
            
        # Normalize weights
        weights = weights / weights.sum()
        
        # Move to device
        if device is not None:
            self.weights = weights.to(device)
        else:
            self.weights = weights
            
        self.model_names = model_names if model_names is not None else [f"model_{i}" for i in range(len(models))]
        self.device = device
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass qua ensemble với xử lý đặc biệt cho meta-learner."""
        # Mảng lưu base model outputs
        base_outputs = []
        meta_learner = None
        meta_learner_idx = -1
        
        # Tìm meta-learner
        for i, (model, name) in enumerate(zip(self.models, self.model_names)):
            if name == "meta_lr":
                meta_learner = model
                meta_learner_idx = i
                break
        
        # Tiến hành base models inference
        for i, model in enumerate(self.models):
            if i != meta_learner_idx:  # Bỏ qua meta-learner ở bước này
                try:
                    with torch.no_grad():
                        model.eval()
                        output = model(x)
                        
                        # Đảm bảo output là tensor đúng kích thước
                        if len(output.shape) == 1:
                            output = output.unsqueeze(1)
                            
                        base_outputs.append(output)
                except Exception as e:
                    logger.error(f"Error in {self.model_names[i]} forward pass: {e}")
                    # Trả về zeros khi có lỗi
                    base_outputs.append(torch.zeros((x.size(0), 1), dtype=torch.float32, device=x.device))
        
        # Xử lý meta-learner nếu có
        if meta_learner is not None and meta_learner_idx >= 0:
            try:
                # Chuyển base outputs thành features cho meta-learner
                meta_features = torch.cat([o.detach() for o in base_outputs], dim=1)
                
                # Áp dụng meta-learner
                with torch.no_grad():
                    meta_learner.eval()
                    meta_output = meta_learner(meta_features)
                    
                    # Đảm bảo meta_output đúng kích thước
                    if len(meta_output.shape) == 1:
                        meta_output = meta_output.unsqueeze(1)
                    
                    # Thêm vào danh sách outputs
                    base_outputs.append(meta_output)
            except Exception as e:
                logger.error(f"Error in meta-learner forward pass: {e}")
                # Trả về zeros khi có lỗi
                base_outputs.append(torch.zeros((x.size(0), 1), dtype=torch.float32, device=x.device))
        
        # Kết hợp outputs theo weights
        combined_output = torch.zeros_like(base_outputs[0])
        
        for i, output in enumerate(base_outputs):
            # Xác định model index (khác nhau nếu có meta-learner)
            model_idx = i if i < meta_learner_idx or meta_learner_idx < 0 else i + 1
            
            # Áp dụng weight
            try:
                weight = self.weights[model_idx]
                if output.device != weight.device:
                    weight = weight.to(output.device)
                
                combined_output += output * weight
            except Exception as e:
                logger.error(f"Error combining outputs for model {model_idx}: {e}")
        
        return combined_output
    
    def get_weights(self) -> torch.Tensor:
        """Get ensemble weights."""
        return self.weights.clone()
    
    def set_weights(self, weights: torch.Tensor) -> None:
        """Set ensemble weights."""
        # Normalize weights
        weights = weights / weights.sum()
        self.weights = weights.to(self.weights.device)

# GA-Stacking Optimizer
class GAStacking:
    """
    Genetic Algorithm-based Stacking Ensemble Optimizer.
    
    This class implements a genetic algorithm to optimize the weights of an ensemble of models.
    """
    
    def __init__(
        self,
        base_models: List[nn.Module],
        model_names: List[str],
        population_size: int = 50,
        generations: int = 30,
        mutation_rate: float = 0.3,
        crossover_rate: float = 0.6,
        device: torch.device = None
    ):
        """
        Initialize GA-Stacking optimizer.
        
        Args:
            base_models: List of base models
            model_names: List of model names
            population_size: Size of population for GA
            generations: Number of generations for GA
            mutation_rate: Mutation rate for GA
            crossover_rate: Crossover rate for GA
            device: Device to use (cpu or cuda)
        """
        self.base_models = base_models
        self.model_names = model_names
        self.num_models = len(base_models)
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.device = device if device is not None else torch.device("cpu")
        
        # Store the best individual from GA
        self.best_weights = None
        self.best_fitness = 0.0
        
        # Store fitness history
        self.fitness_history = []
        
    def initialize_population(self) -> List[np.ndarray]:
        """
        Initialize population for GA.
        
        Returns:
            List of initial individuals (weight vectors)
        """
        population = []
        for _ in range(self.population_size):
            # Create random weights between 0 and 1
            weights = np.random.rand(self.num_models)
            # Normalize to sum to 1
            weights = weights / np.sum(weights)
            population.append(weights)
        return population
    
    def fitness_function(self, weights: np.ndarray, val_data: torch.utils.data.DataLoader) -> float:
        """
        Calculate fitness with strict separation of base and meta models.
        
        Args:
            weights: Array of weights for the ensemble
            val_data: Validation data loader
            
        Returns:
            Fitness score
        """
        correct = 0
        total = 0
        total_loss = 0.0
        
        # Thêm các biến để thu thập dự đoán cho AUC
        all_predictions = []
        all_targets = []
        
        # Create ensemble model with the given weights
        ensemble = EnsembleModel(
            models=self.base_models,
            weights=weights,
            model_names=self.model_names,
            device=self.device
        )
        
        # Evaluate on validation data
        ensemble.eval()
        criterion = nn.BCEWithLogitsLoss()
        
        with torch.no_grad():
            for data, target in val_data:
                data, target = data.to(self.device), target.to(self.device)
                
                # Get ensemble output
                output = ensemble(data)
                
                # Ensure output and target have the same shape
                if output.shape != target.shape:
                    if len(output.shape) == 1:
                        output = output.unsqueeze(1)
                    if len(target.shape) == 1:
                        target = target.unsqueeze(1)
                
                # Calculate loss
                loss = criterion(output, target)
                total_loss += loss.item() * len(data)
                
                # Apply sigmoid to get probabilities
                probs = torch.sigmoid(output)
                
                # Calculate accuracy
                pred = (probs >= 0.5).float()
                correct += (pred == target).sum().item()
                total += target.size(0)
                
                # Thu thập dự đoán và nhãn để tính AUC
                all_predictions.append(probs.cpu().detach().numpy())
                all_targets.append(target.cpu().detach().numpy())
        
        # Tính fitness cuối cùng
        if total == 0:
            return 0.0
        
        accuracy = correct / total
        avg_loss = total_loss / total if total > 0 else float('inf')
        
        # Tính đa dạng trọng số
        weight_diversity = np.std(weights)
        
        # Ghép tất cả dự đoán và nhãn
        if all_predictions and all_targets:
            try:
                all_preds = np.concatenate(all_predictions).flatten()
                all_targs = np.concatenate(all_targets).flatten()
                
                # Chỉ tính AUC nếu có cả hai lớp
                if len(np.unique(all_targs)) > 1:
                    auc = roc_auc_score(all_targs, all_preds)
                    
                    # Calculate F2-score (emphasize recall over precision)
                    y_pred_bin = (all_preds >= 0.5).astype(float)
                    
                    tp = np.sum((y_pred_bin == 1) & (all_targs == 1))
                    fp = np.sum((y_pred_bin == 1) & (all_targs == 0))
                    fn = np.sum((y_pred_bin == 0) & (all_targs == 1))
                    
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    
                    # Calculate F2-score
                    beta = 2
                    f2_score = (1 + beta**2) * precision * recall / ((beta**2 * precision) + recall) if (precision + recall) > 0 else 0
                    
                    # Combined fitness (higher is better)
                    combined_fitness = (0.4 * auc) + (0.3 * f2_score) + (0.2 * accuracy) + (0.1 * weight_diversity)
                    
                    # Add bonus for detecting fraud
                    if tp > 0:
                        combined_fitness += 0.05
                    
                    return combined_fitness
            except Exception as e:
                logger.warning(f"Error calculating AUC: {e}, falling back to accuracy")
        
        # Fallback: use accuracy and diversity
        combined_fitness = (0.8 * accuracy) + (0.2 * weight_diversity)
        return combined_fitness
    
    def selection(self, population: List[np.ndarray], fitnesses: List[float]) -> List[np.ndarray]:
        """
        Select individuals for next generation.
        
        Args:
            population: Current population
            fitnesses: Fitness scores for population
            
        Returns:
            Selected individuals
        """
        # Sort by fitness
        indices = np.argsort(fitnesses)[::-1]
        sorted_population = [population[i] for i in indices]
        
        # Keep elites (top 10%)
        num_elites = max(1, int(0.1 * len(population)))
        elites = sorted_population[:num_elites]
        
        # Tournament selection for the rest
        selected = elites.copy()
        while len(selected) < len(population):
            # Tournament of size 4
            competitors = random.sample(range(len(population)), k=4)
            winner_idx = max(competitors, key=lambda i: fitnesses[i])
            selected.append(population[winner_idx])
        
        return selected
    
    def crossover(self, population: List[np.ndarray]) -> List[np.ndarray]:
        """
        Apply crossover to population.
        
        Args:
            population: Current population
            
        Returns:
            New population after crossover
        """
        new_population = []
        i = 0
        
        # Keep elite individuals (top 10%)
        num_elites = max(1, int(0.1 * len(population)))
        elites = population[:num_elites]
        new_population.extend(elites)
        
        # Apply crossover to rest
        while i < len(population) - num_elites:
            if random.random() < self.crossover_rate and i + 1 < len(population) - num_elites:
                # Select two parents
                parent1 = population[num_elites + i]
                parent2 = population[num_elites + i + 1]
                
                # Perform arithmetic crossover
                alpha = random.random()
                child1 = alpha * parent1 + (1 - alpha) * parent2
                child2 = alpha * parent2 + (1 - alpha) * parent1
                
                # Normalize
                child1 = child1 / np.sum(child1)
                child2 = child2 / np.sum(child2)
                
                new_population.append(child1)
                new_population.append(child2)
                i += 2
            else:
                # No crossover, just copy
                new_population.append(population[num_elites + i])
                i += 1
        
        # Ensure population size is maintained
        if len(new_population) < len(population):
            # Add random individuals to maintain population size
            while len(new_population) < len(population):
                new_population.append(random.choice(population))
        
        return new_population
    
    def mutate(self, population: List[np.ndarray]) -> List[np.ndarray]:
        """
        Apply mutation to population.
        
        Args:
            population: Current population
            
        Returns:
            New population after mutation
        """
        new_population = []
        
        # Keep elite individuals (top 10%) without mutation
        num_elites = max(1, int(0.1 * len(population)))
        elites = population[:num_elites]
        new_population.extend(elites)
        
        # Apply mutation to rest
        for ind in population[num_elites:]:
            # Clone individual
            mutated = np.copy(ind)
            
            # Mutate with probability mutation_rate
            for j in range(len(mutated)):
                if random.random() < self.mutation_rate:
                    # Add Gaussian noise
                    mutated[j] += np.random.normal(0, 0.1)
                    
                    # Ensure non-negative
                    mutated[j] = max(0, mutated[j])
            
            # Normalize to sum to 1
            mutated = mutated / np.sum(mutated)
            
            # Add constraint: no single model should dominate too much (max 0.5 weight)
            max_weight = np.max(mutated)
            if max_weight > 0.5:
                # Reduce the weight of the dominant model
                max_idx = np.argmax(mutated)
                excess = max_weight - 0.5
                mutated[max_idx] = 0.5
                
                # Redistribute excess to other models
                remaining = np.ones_like(mutated)
                remaining[max_idx] = 0
                remaining_sum = np.sum(remaining)
                
                if remaining_sum > 0:
                    mutated += (excess * remaining / remaining_sum)
                
                # Renormalize
                mutated = mutated / np.sum(mutated)
            
            new_population.append(mutated)
        
        return new_population
    
    def optimize(
        self, 
        train_data: torch.utils.data.DataLoader, 
        val_data: torch.utils.data.DataLoader,
        verbose: bool = True
    ) -> np.ndarray:
        """
        Run GA-Stacking optimization.
        
        Args:
            train_data: Training data loader
            val_data: Validation data loader
            verbose: Whether to print progress
            
        Returns:
            Optimized weights for ensemble
        """
        # First train all base models on training data
        self._train_base_models(train_data)
        
        # Initialize population
        population = self.initialize_population()
        
        # Main GA loop
        for gen in range(self.generations):
            start_time = time.time()
            
            # Evaluate fitness for each individual
            fitnesses = []
            for ind in population:
                fitness = self.fitness_function(ind, val_data)
                fitnesses.append(fitness)
            
            # Store best individual
            best_idx = np.argmax(fitnesses)
            if fitnesses[best_idx] > self.best_fitness:
                self.best_fitness = fitnesses[best_idx]
                self.best_weights = population[best_idx].copy()
            
            # Store generation fitness
            generation_fitness = {
                "max": np.max(fitnesses),
                "mean": np.mean(fitnesses),
                "min": np.min(fitnesses),
                "std": np.std(fitnesses)
            }
            self.fitness_history.append(generation_fitness)
            
            # Selection
            selected = self.selection(population, fitnesses)
            
            # Crossover
            crossed = self.crossover(selected)
            
            # Mutation
            population = self.mutate(crossed)
            
            # Log progress
            end_time = time.time()
            if verbose:
                logger.info(f"Generation {gen+1}/{self.generations}: "
                          f"Best={generation_fitness['max']:.4f}, "
                          f"Mean={generation_fitness['mean']:.4f}, "
                          f"Time={end_time-start_time:.2f}s")
                
                # Log weights of best individual
                if gen == self.generations - 1 or gen % 5 == 0:
                    best_ind = population[np.argmax(fitnesses)]
                    weights_str = ", ".join([f"{name}: {weight:.4f}" 
                                          for name, weight in zip(self.model_names, best_ind)])
                    logger.info(f"Best weights: {weights_str}")
        
        return self.best_weights
    
    def _train_base_models(self, train_data: torch.utils.data.DataLoader) -> None:
        """
        Train all base models với xử lý đặc biệt cho meta-learner.
        """
        logger.info("Training base models...")
        
        # Thu thập dữ liệu huấn luyện
        X_train_all = []
        y_train_all = []
        
        for data, target in train_data:
            X_train_all.append(data.cpu().numpy())
            y_train_all.append(target.cpu().numpy())
        
        if not X_train_all or not y_train_all:
            logger.error("No training data available")
            return
        
        X_train = np.vstack(X_train_all)
        y_train = np.vstack(y_train_all).flatten()
        
        # Tách base models và meta-learner
        base_models = []
        meta_learner = None
        base_indices = []
        meta_index = -1
        
        for i, (model, name) in enumerate(zip(self.base_models, self.model_names)):
            if name == "meta_lr":
                meta_learner = model
                meta_index = i
            else:
                base_models.append(model)
                base_indices.append(i)
        
        # Huấn luyện base models đầu tiên
        for i, model in enumerate(base_models):
            try:
                idx = base_indices[i]
                name = self.model_names[idx]
                logger.info(f"Training {name} model...")
                
                if hasattr(model, "fit"):
                    model.fit(X_train, y_train)
                elif hasattr(model, "model") and hasattr(model.model, "fit"):
                    model.model.fit(X_train, y_train)
                    model.is_initialized = True
                    
                logger.info(f"Finished training {name} model")
            except Exception as e:
                logger.error(f"Error training {name} model: {e}")
        
        # Sau đó huấn luyện meta-learner trên dự đoán của base models
        if meta_learner is not None:
            try:
                logger.info("Training meta-learner...")
                
                # Tạo dự đoán từ các base models
                base_preds = []
                for model in base_models:
                    with torch.no_grad():
                        model.eval()
                        x_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
                        preds = model(x_tensor).cpu().numpy()
                        
                        # Đảm bảo dự đoán là vector 1 chiều
                        if len(preds.shape) > 1 and preds.shape[1] == 1:
                            preds = preds.flatten()
                        
                        base_preds.append(preds)
                
                # Ghép dự đoán theo cột
                meta_features = np.column_stack(base_preds)
                
                # QUAN TRỌNG: Cập nhật input_dim của meta_learner
                meta_learner.input_dim = len(base_models)
                
                # Huấn luyện meta-learner
                if hasattr(meta_learner, "fit"):
                    meta_learner.fit(meta_features, y_train)
                elif hasattr(meta_learner, "model") and hasattr(meta_learner.model, "fit"):
                    meta_learner.model.fit(meta_features, y_train)
                    meta_learner.is_initialized = True
                
                logger.info(f"Finished training meta-learner with input_dim={len(base_models)}")
            except Exception as e:
                logger.error(f"Error training meta-learner: {e}")
    
    def get_ensemble_model(self) -> EnsembleModel:
        """
        Get ensemble model with optimized weights.
        
        Returns:
            EnsembleModel with optimized weights
        """
        if self.best_weights is None:
            logger.warning("No optimized weights available, using equal weights")
            weights = np.ones(self.num_models) / self.num_models
        else:
            weights = self.best_weights
        
        return EnsembleModel(
            models=self.base_models,
            weights=weights,
            model_names=self.model_names,
            device=self.device
        )

# Helper functions
def generate_meta_features(
    X: np.ndarray,
    y: np.ndarray,
    model_dict: Dict[str, Any],
    n_splits: int = 5
) -> np.ndarray:
    """
    Generate meta-features for stacking.
    
    Args:
        X: Input features
        y: Target values
        model_dict: Dictionary of base models
        n_splits: Number of cross-validation splits
        
    Returns:
        Meta-features for stacking
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.base import clone
    
    y = np.asarray(y)
    n_samples = X.shape[0]
    n_models = len(model_dict)
    meta_X = np.zeros((n_samples, n_models))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    keys = list(model_dict.keys())
    
    for idx, name in enumerate(keys):
        model = model_dict[name]
        for train_idx, val_idx in skf.split(X, y):
            m = clone(model)
            m.fit(X[train_idx], y[train_idx])
            meta_X[val_idx, idx] = m.predict_proba(X[val_idx])[:, 1]
            
    return meta_X

def get_ensemble_state_dict(ensemble_model: EnsembleModel) -> Dict[str, Any]:
    """
    Get state dict for ensemble model.
    
    Args:
        ensemble_model: Ensemble model
        
    Returns:
        State dict with model states and weights
    """
    if ensemble_model is None:
        logger.warning("No ensemble model provided")
        return {}
    
    # Get weights
    weights = ensemble_model.get_weights().cpu().numpy().tolist()
    
    # Get model names
    model_names = ensemble_model.model_names
    
    # Get model states
    model_states = []
    for model in ensemble_model.models:
        if hasattr(model, "get_params"):
            model_states.append(model.get_params())
        else:
            model_states.append({"model_type": "unknown"})
    
    return {
        "weights": weights,
        "model_names": model_names,
        "model_states": model_states
    }

def load_ensemble_from_state_dict(
    state_dict: Dict[str, Any],
    device: torch.device = None
) -> EnsembleModel:
    """
    Load ensemble model from state dict.
    
    Args:
        state_dict: State dict from get_ensemble_state_dict
        device: Device to use
        
    Returns:
        Loaded ensemble model
    """
    if not state_dict:
        logger.warning("Empty state dict provided")
        return None
    
    # Get weights, model names, and states
    weights = state_dict.get("weights", [])
    model_names = state_dict.get("model_names", [])
    model_states = state_dict.get("model_states", [])
    
    if not weights or not model_names or not model_states:
        logger.warning("Missing data in state dict")
        return None
    
    # Create base models
    models = []
    for i, state in enumerate(model_states):
        model_type = state.get("model_type", "unknown")
        input_dim = state.get("input_dim", 10)
        output_dim = state.get("output_dim", 1)
        
        if model_type == "lr":
            model = BaseModelWrapper("lr", LogisticRegression(solver='liblinear'), input_dim, output_dim)
        elif model_type == "svc":
            model = BaseModelWrapper("svc", SVC(probability=True), input_dim, output_dim)
        elif model_type == "rf":
            model = BaseModelWrapper("rf", RandomForestClassifier(n_estimators=100), input_dim, output_dim)
        elif model_type == "knn":
            model = BaseModelWrapper("knn", KNeighborsClassifier(n_neighbors=5), input_dim, output_dim)
        elif model_type == "xgb" and XGBOOST_AVAILABLE:
            model = BaseModelWrapper("xgb", XGBClassifier(use_label_encoder=False, eval_metric='logloss'), input_dim, output_dim)
        elif model_type == "lgbm" and LIGHTGBM_AVAILABLE:
            model = BaseModelWrapper("lgbm", LGBMClassifier(), input_dim, output_dim)
        elif model_type == "catboost" and CATBOOST_AVAILABLE:
            model = BaseModelWrapper("catboost", CatBoostClassifier(verbose=0), input_dim, output_dim)
        elif model_type == "meta_lr":
            model = MetaLearnerWrapper(input_dim, output_dim)
        else:
            logger.warning(f"Unknown model type: {model_type}")
            model = MetaLearnerWrapper(input_dim, output_dim)
        
        # Set parameters
        if hasattr(model, "set_params"):
            model.set_params(state)
            
        models.append(model)
    
    # Create ensemble model
    ensemble = EnsembleModel(
        models=models,
        weights=weights,
        model_names=model_names,
        device=device
    )
    
    return ensemble