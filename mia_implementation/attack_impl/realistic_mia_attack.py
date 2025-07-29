"""
Realistic MIA Attack Implementation
- Target model: trains on REAL training data 
- Shadow model: trains on SYNTHETIC data
- Attack test: evaluates on REAL non-training data

This implements the proper threat model for membership inference attacks.
"""

'''
========================================================================================================
HYPERPARAMETER CONFIGURATION SECTION - MODIFY ALL PARAMETERS HERE
========================================================================================================
'''

# === EXPERIMENT CONFIGURATION ===
RANDOM_SEED = 42                    # Random seed for reproducibility
DATASET_NAME = "event"             # Dataset: "twitch" or "event"
CLASSIFICATION_TARGET = "gender"     # "mature" for Twitch, "gender" for Event

# === MODEL CONFIGURATION ===
TARGET_MODEL_TYPE = "GCN"          # GNN architecture: "GCN", "GAT", "SAGE"
SHADOW_MODEL_TYPE = "GCN"          # Can be different from target for transfer attacks
HIDDEN_DIM = 256                   # Hidden layer dimension (increased for more capacity)
NUM_GNN_LAYERS = 2                 # Number of GNN layers (reduced to avoid oversmoothing)
GNN_DROPOUT = 0.2                  # Dropout rate (reduced to allow more overfitting)

# === TRAINING HYPERPARAMETERS ===
# Target model training (make it overfit more to increase vulnerability)
TARGET_EPOCHS = 500                # Target model training epochs (increased)
TARGET_LR = 0.01                   # Target model learning rate (increased)
TARGET_WEIGHT_DECAY = 1e-5         # L2 regularization (reduced to allow overfitting)
TARGET_PATIENCE = 50               # Early stopping patience (increased)

# Shadow model training (match target behavior)
SHADOW_EPOCHS = 500                # Shadow model training epochs (increased)
SHADOW_LR = 0.01                   # Shadow model learning rate (match target)
SHADOW_WEIGHT_DECAY = 1e-5         # L2 regularization (match target)
SHADOW_PATIENCE = 50               # Early stopping patience (increased)

# === ATTACK MODEL HYPERPARAMETERS ===
ATTACK_MODEL_TYPE = "neural"       # "neural", "logistic", or "random_forest"

# Neural attack model
ATTACK_HIDDEN_DIM = 64             # Hidden dimension for neural attack model
ATTACK_DROPOUT = 0.3               # Dropout for attack model
ATTACK_EPOCHS = 100                # Attack model training epochs
ATTACK_LR = 0.01                   # Attack model learning rate
ATTACK_WEIGHT_DECAY = 1e-4         # L2 regularization for attack model
ATTACK_BATCH_SIZE = 128            # Batch size for attack training

# Classical ML attack models
RF_N_ESTIMATORS = 100              # Random Forest: number of trees
RF_MAX_DEPTH = 10                  # Random Forest: maximum depth
LOGISTIC_C = 1.0                   # Logistic Regression: regularization strength

# === DATA PROCESSING ===
VALIDATION_SPLIT = 0.2             # Fraction of data for validation
MAX_NODES_PER_GRAPH = 500          # Maximum nodes to sample per graph
MIN_NODES_PER_GRAPH = 50           # Minimum nodes to sample per graph

# === EVALUATION PARAMETERS ===
NUM_EVALUATION_RUNS = 1            # Number of independent runs for statistics
CONFIDENCE_THRESHOLD = 0.5         # Threshold for binary classification
EVAL_BATCH_SIZE = 256              # Batch size for evaluation

# === ADVANCED SETTINGS ===
USE_EARLY_STOPPING = True          # Enable early stopping
USE_BATCH_NORMALIZATION = True     # Enable batch normalization in GNN
USE_RESIDUAL_CONNECTIONS = False   # Enable residual connections (for deeper models)
NORMALIZE_POSTERIORS = True        # Normalize posterior vectors before attack
USE_TEMPERATURE_SCALING = False    # Apply temperature scaling to posteriors
TEMPERATURE = 1.0                  # Temperature for scaling (if enabled)

# === LOGGING AND OUTPUT ===
VERBOSE = True                     # Print detailed progress information
SAVE_MODELS = False                # Save trained models to disk
SAVE_RESULTS = True                # Save results to file
RESULTS_DIR = "results"            # Directory for saving results

'''
========================================================================================================
END HYPERPARAMETER CONFIGURATION
========================================================================================================
'''

import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
import numpy as np
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import time

# Add our data processing to path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir.parent / "data_processing"))

from mia_data_loader import get_realistic_attack_data

def setup_environment(seed=RANDOM_SEED):
    """Setup reproducible environment"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if VERBOSE:
        print(f"Using device: {device}")
        print(f"Random seed: {seed}")
    return device

class ImprovedGNNClassifier(torch.nn.Module):
    """Enhanced GNN classifier with better architecture"""
    
    def __init__(self, num_features, num_classes, model_type=TARGET_MODEL_TYPE, hidden_dim=HIDDEN_DIM, num_layers=NUM_GNN_LAYERS, dropout=GNN_DROPOUT):
        super().__init__()
        self.model_type = model_type
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Create layer lists
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        
        # Input layer
        if model_type == "GCN":
            self.convs.append(GCNConv(num_features, hidden_dim))
        elif model_type == "GAT":
            self.convs.append(GATConv(num_features, hidden_dim, heads=4, concat=True, dropout=dropout))
            hidden_dim = hidden_dim * 4  # Account for concatenated heads
        elif model_type == "SAGE":
            self.convs.append(SAGEConv(num_features, hidden_dim))
        else:
            raise ValueError(f"Model type {model_type} not supported")
        
        if USE_BATCH_NORMALIZATION:
            self.bns.append(torch.nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            if model_type == "GCN":
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            elif model_type == "GAT":
                self.convs.append(GATConv(hidden_dim, hidden_dim // 4, heads=4, concat=True, dropout=dropout))
            elif model_type == "SAGE":
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            
            if USE_BATCH_NORMALIZATION:
                self.bns.append(torch.nn.BatchNorm1d(hidden_dim))
        
        # Output layer
        if model_type == "GCN":
            self.convs.append(GCNConv(hidden_dim, num_classes))
        elif model_type == "GAT":
            self.convs.append(GATConv(hidden_dim, num_classes, heads=1, concat=False, dropout=dropout))
        elif model_type == "SAGE":
            self.convs.append(SAGEConv(hidden_dim, num_classes))
        
        self.dropout_layer = torch.nn.Dropout(dropout)
        
    def forward(self, x, edge_index):
        # Input and hidden layers
        for i in range(len(self.convs) - 1):
            x = self.convs[i](x, edge_index)
            if USE_BATCH_NORMALIZATION:
                x = self.bns[i](x)
            x = F.relu(x)
            x = self.dropout_layer(x)
        
        # Output layer
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)

class NeuralAttackModel(torch.nn.Module):
    """Neural network attack model for membership inference"""
    
    def __init__(self, input_dim, hidden_dim=ATTACK_HIDDEN_DIM, dropout=ATTACK_DROPOUT):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(), 
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim // 2, 1),
            torch.nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.layers(x)

# Keep old class for compatibility
class GNNClassifier(ImprovedGNNClassifier):
    def __init__(self, num_features, num_classes, model_type="GCN", hidden_dim=64):
        super().__init__(num_features, num_classes, model_type, hidden_dim, num_layers=2, dropout=0.5)

def train_model_on_subgraphs(model, subgraph_list, device, epochs=TARGET_EPOCHS, lr=TARGET_LR, patience=TARGET_PATIENCE, weight_decay=TARGET_WEIGHT_DECAY):
    """
    Enhanced training with better hyperparameters and early stopping
    
    Args:
        model: GNN model to train
        subgraph_list: List of torch_geometric.Data objects
        device: Training device
        epochs: Number of training epochs
        lr: Learning rate
        patience: Early stopping patience
    """
    if not subgraph_list:
        print("Warning: No subgraphs provided for training")
        return
        
    if VERBOSE:
        print(f"Training model on {len(subgraph_list)} subgraphs...")
    
    # Enhanced optimizer with weight decay
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    model.train()
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        total_loss = 0
        
        # Shuffle subgraphs for better training
        shuffled_subgraphs = subgraph_list.copy()
        random.shuffle(shuffled_subgraphs)
        
        # Iterate through each subgraph individually
        for subgraph in shuffled_subgraphs:
            # Move to device
            subgraph = subgraph.to(device)
            
            # Training step
            optimizer.zero_grad()
            out = model(subgraph.x, subgraph.edge_index)
            loss = F.nll_loss(out, subgraph.y)
            loss.backward()
            
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(subgraph_list)
        scheduler.step(avg_loss)
        
        # Early stopping check
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if epoch % 10 == 0:
            print(f"  Epoch {epoch}, Avg Loss: {avg_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch} (best loss: {best_loss:.4f})")
            break

def train_neural_attack_model(model, train_features, train_labels, device):
    """Train neural attack model with proper PyTorch training loop"""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=ATTACK_LR, weight_decay=ATTACK_WEIGHT_DECAY)
    criterion = torch.nn.BCELoss()
    
    # Convert data to tensors
    X_train = torch.FloatTensor(train_features).to(device)
    y_train = torch.FloatTensor(train_labels).to(device)
    
    # Create data loader for batch training
    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=ATTACK_BATCH_SIZE, shuffle=True)
    
    best_loss = float('inf')
    patience_counter = 0
    patience = 10
    
    if VERBOSE:
        print(f"    Training neural attack model for {ATTACK_EPOCHS} epochs...")
    
    for epoch in range(ATTACK_EPOCHS):
        total_loss = 0
        for batch_features, batch_labels in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_features).squeeze()
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            if VERBOSE:
                print(f"    Early stopping at epoch {epoch}")
            break
        
        if epoch % 20 == 0 and VERBOSE:
            print(f"    Epoch {epoch}: Loss = {avg_loss:.4f}")
    
    model.eval()
    return model

def extract_enhanced_attack_features(model, subgraph_list, device):
    """
    Extract enhanced features for membership inference attacks
    
    Returns:
        numpy array with posteriors, entropy, confidence, and other features
    """
    if not subgraph_list:
        return np.array([])
        
    model.eval()
    all_features = []
    
    with torch.no_grad():
        for subgraph in subgraph_list:
            subgraph = subgraph.to(device)
            
            # Get model output (log probabilities)
            log_probs = model(subgraph.x, subgraph.edge_index)
            probs = torch.exp(log_probs)  # Convert to probabilities
            
            # Standard posteriors
            posteriors = probs.cpu().numpy()
            
            # Additional attack features
            # 1. Prediction entropy (uncertainty measure)
            entropy = -(probs * log_probs).sum(dim=1).cpu().numpy()
            
            # 2. Maximum confidence
            max_confidence = probs.max(dim=1)[0].cpu().numpy()
            
            # 3. Prediction margin (difference between top two predictions)
            sorted_probs, _ = torch.sort(probs, dim=1, descending=True)
            margin = (sorted_probs[:, 0] - sorted_probs[:, 1]).cpu().numpy()
            
            # 4. L2 norm of probabilities
            prob_norm = torch.norm(probs, p=2, dim=1).cpu().numpy()
            
            # Combine all features
            enhanced_features = np.column_stack([
                posteriors,          # Original posteriors
                entropy.reshape(-1, 1),      # Entropy
                max_confidence.reshape(-1, 1),  # Max confidence  
                margin.reshape(-1, 1),       # Prediction margin
                prob_norm.reshape(-1, 1)     # L2 norm
            ])
            
            all_features.append(enhanced_features)
    
    return np.vstack(all_features) if all_features else np.array([])

def extract_posteriors(model, subgraph_list, device):
    """
    Extract model posteriors from subgraphs (backward compatibility)
    
    Returns:
        numpy array of posterior probabilities
    """
    enhanced_features = extract_enhanced_attack_features(model, subgraph_list, device)
    if enhanced_features.size == 0:
        return np.array([])
    
    # Return only the posterior probabilities (first num_classes columns)
    num_classes = 2  # Both datasets are binary classification
    return enhanced_features[:, :num_classes]

def run_realistic_mia_attack(dataset_name=DATASET_NAME, model_type=TARGET_MODEL_TYPE):
    """
    Run realistic MIA attack with separate data sources
    """
    print(f"\\n{'='*80}")
    print(f"REALISTIC MIA ATTACK: {dataset_name.upper()} with {model_type}")
    print(f"{'='*80}")
    
    device = setup_environment()
    
    # Load realistic attack data
    print("Loading realistic attack data...")
    attack_data = get_realistic_attack_data(dataset_name)
    
    target_subgraphs = attack_data['target_train_subgraphs']
    shadow_subgraphs = attack_data['shadow_train_subgraphs'] 
    # shadow_subgraphs = attack_data['target_train_subgraphs'] 
    test_subgraphs = attack_data['attack_test_subgraphs']
    dataset_info = attack_data['dataset_info']
    
    print(f"Dataset: {dataset_info['name']}")
    print(f"Target subgraphs (REAL train): {len(target_subgraphs)}")
    print(f"Shadow subgraphs (SYNTHETIC): {len(shadow_subgraphs)}")
    print(f"Test subgraphs (REAL non-train): {len(test_subgraphs)}")
    print(f"Classification target: {dataset_info['classification_target']}")
    
    # Check data availability
    if len(target_subgraphs) == 0:
        print("⚠️  Warning: No real training data available (pandas version issue)")
        print("🔄 Falling back to synthetic-only training for demonstration...")
        target_subgraphs = shadow_subgraphs[:len(shadow_subgraphs)//2]  # Use half synthetic for target
        shadow_subgraphs = shadow_subgraphs[len(shadow_subgraphs)//2:]  # Use half for shadow
    
    if len(test_subgraphs) == 0:
        print("⚠️  Warning: No real test data available (pandas version issue)")
        print("🔄 Using synthetic data for test evaluation...")
        test_subgraphs = shadow_subgraphs[-50:]  # Use some synthetic for testing
    
    # Create models
    num_features = dataset_info['num_features']
    num_classes = dataset_info['num_classes']
    
    print(f"\\nCreating improved models: {num_features} features → {num_classes} classes")
    target_model = ImprovedGNNClassifier(num_features, num_classes, TARGET_MODEL_TYPE, hidden_dim=HIDDEN_DIM, num_layers=NUM_GNN_LAYERS, dropout=GNN_DROPOUT).to(device)
    shadow_model = ImprovedGNNClassifier(num_features, num_classes, SHADOW_MODEL_TYPE, hidden_dim=HIDDEN_DIM, num_layers=NUM_GNN_LAYERS, dropout=GNN_DROPOUT).to(device)
    
    # Train target model on REAL data
    print(f"\\n1. Training Target Model on REAL data ({len(target_subgraphs)} subgraphs)...")
    train_model_on_subgraphs(target_model, target_subgraphs, device, epochs=TARGET_EPOCHS, lr=TARGET_LR, patience=TARGET_PATIENCE, weight_decay=TARGET_WEIGHT_DECAY)
    
    # Train shadow model on SYNTHETIC data  
    print(f"\\n2. Training Shadow Model on SYNTHETIC data ({len(shadow_subgraphs)} subgraphs)...")
    train_model_on_subgraphs(shadow_model, shadow_subgraphs, device, epochs=SHADOW_EPOCHS, lr=SHADOW_LR, patience=SHADOW_PATIENCE, weight_decay=SHADOW_WEIGHT_DECAY)
    
    # Extract posteriors for attack
    print(f"\\n3. Extracting enhanced attack features...")
    
    # Shadow model features (for training attack model)
    print("  - Shadow model on synthetic training data (members)...")
    shadow_train_features = extract_enhanced_attack_features(shadow_model, shadow_subgraphs[:len(shadow_subgraphs)//2], device)
    
    print("  - Shadow model on synthetic test data (non-members)...")  
    shadow_test_features = extract_enhanced_attack_features(shadow_model, shadow_subgraphs[len(shadow_subgraphs)//2:], device)
    
    # Target model features (for attack evaluation)
    print("  - Target model on real training data (members)...")
    target_train_features = extract_enhanced_attack_features(target_model, target_subgraphs, device)
    
    print("  - Target model on real test data (non-members)...")
    target_test_features = extract_enhanced_attack_features(target_model, test_subgraphs, device)
    
    # Check if we have enough data
    if len(shadow_train_features) == 0 or len(shadow_test_features) == 0:
        print("❌ Error: No features extracted for shadow model")
        return None
        
    if len(target_train_features) == 0 or len(target_test_features) == 0:
        print("❌ Error: No features extracted for target model")
        return None
    
    print(f"  ✓ Shadow train features: {shadow_train_features.shape}")
    print(f"  ✓ Shadow test features: {shadow_test_features.shape}")
    print(f"  ✓ Target train features: {target_train_features.shape}")
    print(f"  ✓ Target test features: {target_test_features.shape}")
    
    # Train enhanced attack model
    print(f"\\n4. Training enhanced attack model...")
    
    # Prepare attack training data
    attack_train_features = np.vstack([shadow_train_features, shadow_test_features])
    attack_train_labels = np.hstack([
        np.ones(len(shadow_train_features)),   # Members = 1
        np.zeros(len(shadow_test_features))    # Non-members = 0
    ])
    
    # Try multiple attack models and select the best
    attack_models = {}
    
    if ATTACK_MODEL_TYPE == "neural" or ATTACK_MODEL_TYPE == "all":
        neural_model = NeuralAttackModel(attack_train_features.shape[1]).to(device)
        attack_models['neural'] = neural_model
    
    if ATTACK_MODEL_TYPE == "logistic" or ATTACK_MODEL_TYPE == "all":
        attack_models['logistic'] = LogisticRegression(random_state=RANDOM_SEED, max_iter=1000, C=LOGISTIC_C)
    
    if ATTACK_MODEL_TYPE == "random_forest" or ATTACK_MODEL_TYPE == "all":
        attack_models['random_forest'] = RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS, 
            random_state=RANDOM_SEED, 
            max_depth=RF_MAX_DEPTH
        )
    
    # If no specific model type or "all", use both classical methods by default
    if not attack_models:
        attack_models = {
            'logistic': LogisticRegression(random_state=RANDOM_SEED, max_iter=1000, C=LOGISTIC_C),
            'random_forest': RandomForestClassifier(
                n_estimators=RF_N_ESTIMATORS, 
                random_state=RANDOM_SEED, 
                max_depth=RF_MAX_DEPTH
            )
        }
    
    best_model = None
    best_score = 0
    best_name = ""
    
    for name, model in attack_models.items():
        if name == 'neural':
            # Special handling for neural attack model
            train_neural_attack_model(model, attack_train_features, attack_train_labels, device)
            # Evaluate neural model performance
            model.eval()
            with torch.no_grad():
                X_val = torch.FloatTensor(attack_train_features).to(device)
                pred_probs = model(X_val).cpu().numpy().flatten()
                pred_labels = (pred_probs > 0.5).astype(int)
                cv_score = accuracy_score(attack_train_labels, pred_labels)
        else:
            # Classical ML models
            model.fit(attack_train_features, attack_train_labels)
            # Use cross-validation score on training data to select best model
            from sklearn.model_selection import cross_val_score
            cv_score = cross_val_score(model, attack_train_features, attack_train_labels, cv=5).mean()
        
        if VERBOSE:
            print(f"  - {name}: CV Score = {cv_score:.3f}")
        
        if cv_score > best_score:
            best_score = cv_score
            best_model = model
            best_name = name
    
    print(f"  ✓ Best attack model: {best_name} (CV Score: {best_score:.3f})")
    print(f"  ✓ Attack model trained on {len(attack_train_features)} samples with {attack_train_features.shape[1]} features")
    
    # Evaluate attack on target model
    print(f"\\n5. Evaluating attack on target model...")
    
    # Prepare target evaluation data
    target_features = np.vstack([target_train_features, target_test_features])
    target_true_labels = np.hstack([
        np.ones(len(target_train_features)),   # Training data = members
        np.zeros(len(target_test_features))    # Test data = non-members
    ])
    
    # Attack predictions
    if best_name == 'neural':
        # Neural model predictions
        best_model.eval()
        with torch.no_grad():
            X_target = torch.FloatTensor(target_features).to(device)
            attack_probabilities = best_model(X_target).cpu().numpy().flatten()
            attack_predictions = (attack_probabilities > 0.5).astype(int)
    else:
        # Classical ML model predictions
        attack_predictions = best_model.predict(target_features)
        attack_probabilities = best_model.predict_proba(target_features)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(target_true_labels, attack_predictions)
    precision = precision_score(target_true_labels, attack_predictions)
    recall = recall_score(target_true_labels, attack_predictions)
    f1 = f1_score(target_true_labels, attack_predictions)
    auc = roc_auc_score(target_true_labels, attack_probabilities)
    
    # Calculate advantage over random guessing
    advantage = accuracy - 0.5
    
    # Print results
    print(f"\\n{'='*80}")
    print(f"REALISTIC MIA ATTACK RESULTS")
    print(f"{'='*80}")
    print(f"Dataset: {dataset_name.upper()}")
    print(f"Model: {model_type}")
    print(f"Target: {dataset_info['classification_target']}")
    print(f"{'─'*50}")
    print(f"Attack Accuracy:    {accuracy:.3f}")
    print(f"Attack Precision:   {precision:.3f}")
    print(f"Attack Recall:      {recall:.3f}")
    print(f"Attack F1-Score:    {f1:.3f}")
    print(f"Attack AUC:         {auc:.3f}")
    print(f"Advantage:          {advantage:.3f}")
    print(f"{'─'*50}")
    
    # Interpret results
    if accuracy > 0.7:
        print("🚨 HIGH PRIVACY RISK: Strong membership inference attack")
    elif accuracy > 0.6:
        print("⚠️  MODERATE PRIVACY RISK: Successful membership inference")
    elif accuracy > 0.55:
        print("🔍 LOW PRIVACY RISK: Weak but detectable membership inference")
    else:
        print("✅ MINIMAL PRIVACY RISK: Attack not significantly better than random")
    
    if auc > 0.8:
        print("🚨 EXCELLENT ATTACK QUALITY: Very strong discriminative power")
    elif auc > 0.7:
        print("⚠️  GOOD ATTACK QUALITY: Strong discriminative power") 
    elif auc > 0.6:
        print("🔍 MODERATE ATTACK QUALITY: Some discriminative power")
    else:
        print("✅ POOR ATTACK QUALITY: Limited discriminative power")
    
    print(f"{'='*80}")
    
    return {
        'dataset': dataset_name,
        'model': model_type,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'advantage': advantage,
        'num_target_train': len(target_train_features),
        'num_target_test': len(target_test_features),
        'num_shadow_train': len(shadow_train_features),
        'num_shadow_test': len(shadow_test_features),
        'attack_model_type': best_name
    }

if __name__ == "__main__":
    print("🎯 REALISTIC MEMBERSHIP INFERENCE ATTACK")
    print("="*80)
    print("Threat Model:")
    print("- Target model: Trained on REAL data (victim's model)")
    print("- Shadow model: Trained on SYNTHETIC data (attacker's approximation)")
    print("- Attack test: Evaluated on REAL non-training data")
    print("="*80)
    
    # Test with configuration from hyperparameters or run multiple configs
    if NUM_EVALUATION_RUNS == 1:
        # Single run with hyperparameter configuration
        configs = [(DATASET_NAME, TARGET_MODEL_TYPE)]
    else:
        # Multiple configurations for comprehensive evaluation
        configs = [
            ("twitch", "GCN"),
            ("twitch", "GAT"), 
            ("twitch", "SAGE"),
            # ("event", "GCN"),  # Can enable later
        ]
    
    results = []
    
    for dataset_name, model_type in configs:
        for run_id in range(NUM_EVALUATION_RUNS):
            if VERBOSE and NUM_EVALUATION_RUNS > 1:
                print(f"\n--- Run {run_id + 1}/{NUM_EVALUATION_RUNS} ---")
            try:
                result = run_realistic_mia_attack(dataset_name, model_type)
                if result:
                    results.append(result)
                    if VERBOSE:
                        print(f"\\n✅ Completed: {dataset_name} + {model_type}")
            except Exception as e:
                if VERBOSE:
                    print(f"\\n❌ Failed: {dataset_name} + {model_type}")
                    print(f"Error: {e}")
                    import traceback
                    traceback.print_exc()
    
    # Final summary
    if results:
        print(f"\\n{'='*80}")
        print("FINAL SUMMARY")
        print(f"{'='*80}")
        
        for result in results:
            print(f"{result['dataset'].title()} + {result['model']}:")
            print(f"  Accuracy: {result['accuracy']:.3f} (Advantage: {result['advantage']:.3f})")
            print(f"  AUC: {result['auc']:.3f}")
            print(f"  Data sizes: Target({result['num_target_train']}/{result['num_target_test']}), "
                  f"Shadow({result['num_shadow_train']}/{result['num_shadow_test']})")
            
            status = "🚨 HIGH RISK" if result['accuracy'] > 0.7 else \
                     "⚠️  MODERATE RISK" if result['accuracy'] > 0.6 else \
                     "🔍 LOW RISK" if result['accuracy'] > 0.55 else "✅ MINIMAL RISK"
            print(f"  Status: {status}")
            print()
    
    print("🎯 Realistic MIA Attack Complete!")