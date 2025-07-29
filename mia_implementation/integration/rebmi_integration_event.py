
# Integration script for event dataset with rebMIGraph/TSTS.py
# Add this code to the beginning of TSTS.py or run this script first

import sys
from pathlib import Path

# Add our data loader to path
sys.path.append(str(Path(__file__).parent.parent))
from data_processing import setup_data_for_rebmi

# Setup our custom data
dataset_name = "event"
mock_dataset, data_dict = setup_data_for_rebmi(dataset_name)

# Override the dataset variable that rebMIGraph expects
dataset = mock_dataset
data = dataset[0]  # rebMIGraph expects dataset[0] to be the data

# Set custom parameters
data_type = "Custom_Event"
num_train_Train_per_class = 50
num_train_Shadow_per_class = 50
num_test_Target = 100
num_test_Shadow = 100

print(f"Custom dataset loaded: {data_type}")
print(f"Nodes: {data.x.shape[0]}, Features: {data.x.shape[1]}")
print(f"Classes: {len(torch.unique(data.y))}")

# The rest of rebMIGraph/TSTS.py can now run with our custom data
