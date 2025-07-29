"""
Minimal wrapper script to run MIA attacks using our datasets with rebMIGraph/TSTS.py

This script provides the minimal interface to rebMIGraph while using our custom data sources.
It modifies the necessary parameters and data loading without changing the core rebMIGraph code.
"""

import sys
import os
from pathlib import Path
import torch
import numpy as np
import random

# Add paths for imports
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))
sys.path.append(str(current_dir / "rebMIGraph"))

from mia_data_loader import setup_data_for_rebmi

def setup_environment():
    """Setup random seeds and device"""
    # Set random seeds for reproducibility
    random_seed = 42
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    return device

def modify_rebmi_parameters(dataset_name):
    """
    Set parameters that rebMIGraph expects
    Returns dictionary of parameters to inject into rebMIGraph
    """
    if dataset_name == "twitch":
        # Parameters for Twitch dataset
        params = {
            'data_type': 'Custom_Twitch',
            'model_type': 'GCN',  # Start with GCN
            'num_train_Train_per_class': 50,  # Reduced for our smaller datasets
            'num_train_Shadow_per_class': 50,
            'num_test_Target': 100,
            'num_test_Shadow': 100,
        }
    elif dataset_name == "event":
        # Parameters for Event dataset  
        params = {
            'data_type': 'Custom_Event',
            'model_type': 'GCN',  # Start with GCN
            'num_train_Train_per_class': 50,
            'num_train_Shadow_per_class': 50,
            'num_test_Target': 100,
            'num_test_Shadow': 100,
        }
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return params

def inject_custom_data_into_rebmi():
    """
    This function will modify the global namespace that rebMIGraph uses
    We'll inject our custom dataset and parameters
    """
    pass  # Will be implemented when we run rebMIGraph

def run_attack(dataset_name, model_type="GCN"):
    """
    Run the MIA attack for the specified dataset and model
    """
    print(f"\n{'='*60}")
    print(f"Running MIA Attack on {dataset_name} with {model_type}")
    print(f"{'='*60}")
    
    # Setup environment
    device = setup_environment()
    
    # Load our custom data
    print("Loading custom datasets...")
    mock_dataset, data_dict = setup_data_for_rebmi(dataset_name)
    
    print(f"Dataset loaded successfully:")
    print(f"  - Classes: {mock_dataset.num_classes}")
    print(f"  - Features: {mock_dataset.num_node_features}")
    print(f"  - Nodes: {mock_dataset.data.x.shape[0]}")
    print(f"  - Edges: {mock_dataset.data.edge_index.shape[1]}")
    
    # Get parameters for rebMIGraph
    params = modify_rebmi_parameters(dataset_name)
    params['model_type'] = model_type
    
    print(f"Using parameters: {params}")
    
    # TODO: Integrate with rebMIGraph/TSTS.py
    # For now, we'll create a minimal version that demonstrates the data flow
    
    print("\n--- Data Flow Verification ---")
    print(f"Target data available: {len(data_dict['target_data'])} subgraphs")
    print(f"Shadow data available: {len(data_dict['shadow_data'])} subgraphs") 
    print(f"Test data available: {len(data_dict['test_data'])} subgraphs")
    
    # Verify data consistency
    if data_dict['shadow_data']:
        shadow_sample = data_dict['shadow_data'][0]
        print(f"Shadow sample: {shadow_sample.x.shape[0]} nodes, {shadow_sample.x.shape[1]} features")
        print(f"Shadow labels: {torch.unique(shadow_sample.y).tolist()}")
    
    if data_dict['target_data']:
        target_sample = data_dict['target_data'][0]
        print(f"Target sample: {target_sample.x.shape[0]} nodes, {target_sample.x.shape[1]} features")
        print(f"Target labels: {torch.unique(target_sample.y).tolist()}")
    
    # Return data for potential manual integration
    return {
        'dataset': mock_dataset,
        'data_dict': data_dict,
        'params': params,
        'device': device
    }

def create_rebmi_integration_script(dataset_name):
    """
    Create a script that can be used to integrate with rebMIGraph/TSTS.py
    This script will contain the minimal modifications needed
    """
    script_content = f'''
# Integration script for {dataset_name} dataset with rebMIGraph/TSTS.py
# Add this code to the beginning of TSTS.py or run this script first

import sys
from pathlib import Path

# Add our data loader to path
sys.path.append(str(Path(__file__).parent.parent))
from mia_data_loader import setup_data_for_rebmi

# Setup our custom data
dataset_name = "{dataset_name}"
mock_dataset, data_dict = setup_data_for_rebmi(dataset_name)

# Override the dataset variable that rebMIGraph expects
dataset = mock_dataset
data = dataset[0]  # rebMIGraph expects dataset[0] to be the data

# Set custom parameters
data_type = "Custom_{dataset_name.title()}"
num_train_Train_per_class = 50
num_train_Shadow_per_class = 50
num_test_Target = 100
num_test_Shadow = 100

print(f"Custom dataset loaded: {{data_type}}")
print(f"Nodes: {{data.x.shape[0]}}, Features: {{data.x.shape[1]}}")
print(f"Classes: {{len(torch.unique(data.y))}}")

# The rest of rebMIGraph/TSTS.py can now run with our custom data
'''
    
    script_path = current_dir / f"rebmi_integration_{dataset_name}.py"
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    print(f"Created integration script: {script_path}")
    return script_path

if __name__ == "__main__":
    # Run attacks for both datasets
    datasets = ["twitch", "event"]
    models = ["GCN"]  # Start with GCN, can extend to GAT, SAGE, SGC
    
    results = {}
    
    for dataset_name in datasets:
        print(f"\\n{'#'*80}")
        print(f"TESTING DATASET: {dataset_name.upper()}")
        print(f"{'#'*80}")
        
        try:
            # Test data loading and setup
            result = run_attack(dataset_name, "GCN")
            results[dataset_name] = result
            
            # Create integration script
            integration_script = create_rebmi_integration_script(dataset_name)
            
            print(f"\\n✓ Successfully setup {dataset_name} for MIA attack")
            print(f"✓ Integration script created: {integration_script}")
            
        except Exception as e:
            print(f"✗ Error with {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    for dataset_name, result in results.items():
        if result:
            dataset = result['dataset']
            print(f"{dataset_name.title()}:")
            print(f"  - Classes: {dataset.num_classes}")
            print(f"  - Features: {dataset.num_node_features}")
            print(f"  - Total nodes: {dataset.data.x.shape[0]}")
            print(f"  - Total edges: {dataset.data.edge_index.shape[1]}")
    
    print("\\nNext steps:")
    print("1. Use the integration scripts to modify rebMIGraph/TSTS.py")
    print("2. Run the modified TSTS.py with our custom datasets")
    print("3. Analyze the attack results")