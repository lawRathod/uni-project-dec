import pickle
import pandas as pd
import networkx as nx
from pathlib import Path
import numpy as np

def analyze_data_conversion_requirements():
    """Analyze what's needed to convert our data to torch_geometric format"""
    
    print("="*80)
    print("DATA CONVERSION ANALYSIS")
    print("="*80)
    
    datasets_dir = Path("/Users/prateek/notes/uni-project-dec/datasets")
    
    # Analyze both datasets
    for dataset_name in ["twitch", "event"]:
        print(f"\n{dataset_name.upper()} DATASET ANALYSIS:")
        print("-" * 50)
        
        # Load synthetic data (we can access this)
        synth_file = datasets_dir / dataset_name / f"{dataset_name}_synth.pickle"
        with open(synth_file, "rb") as f:
            synth_data = pickle.load(f)
        
        # Analyze first subgraph
        df, graph = synth_data[0]
        
        print(f"Graph Structure:")
        print(f"  - Nodes: {graph.number_of_nodes()}")
        print(f"  - Edges: {graph.number_of_edges()}")
        print(f"  - Node IDs: {list(graph.nodes())[:10]}...")  # First 10
        print(f"  - Edge format: {list(graph.edges())[:5]}...")  # First 5
        
        print(f"\nDataFrame Structure:")
        print(f"  - Shape: {df.shape}")
        print(f"  - Index: {df.index.tolist()[:10]}...")  # First 10
        print(f"  - Columns: {df.columns.tolist()}")
        
        print(f"\nNode-Graph Alignment:")
        df_nodes = set(df.index)
        graph_nodes = set(graph.nodes())
        if df_nodes == graph_nodes:
            print("  ✓ DataFrame index matches graph nodes")
        else:
            print("  ✗ MISMATCH between DataFrame index and graph nodes")
            print(f"    DF nodes: {len(df_nodes)}, Graph nodes: {len(graph_nodes)}")
            print(f"    Missing in DF: {graph_nodes - df_nodes}")
            print(f"    Missing in Graph: {df_nodes - graph_nodes}")
        
        print(f"\nFeature Types for torch_geometric conversion:")
        for col in df.columns:
            dtype = df[col].dtype
            if dtype == 'bool':
                print(f"  - {col}: {dtype} → convert to int (0/1)")
            elif dtype in ['int64', 'float64']:
                print(f"  - {col}: {dtype} → use as-is")
            elif dtype == 'datetime64[ns]':
                print(f"  - {col}: {dtype} → convert to timestamp or extract features")
            elif dtype == 'category':
                print(f"  - {col}: {dtype} → one-hot encode or label encode")
            else:
                print(f"  - {col}: {dtype} → NEEDS INVESTIGATION")
        
        # Identify classification target
        if dataset_name == "twitch":
            target_col = "mature"
        else:  # event
            target_col = "gender"
        
        print(f"\nClassification Target: {target_col}")
        if target_col in df.columns:
            print(f"  - Type: {df[target_col].dtype}")
            print(f"  - Unique values: {df[target_col].unique()}")
            if df[target_col].dtype == 'bool':
                print(f"  - Distribution: {df[target_col].value_counts().to_dict()}")
            else:
                print(f"  - Distribution: {df[target_col].value_counts().to_dict()}")

def analyze_potential_compatibility_issues():
    """Identify potential issues between real and synthetic data"""
    
    print("\n" + "="*80)
    print("REAL vs SYNTHETIC DATA COMPATIBILITY ANALYSIS")
    print("="*80)
    
    datasets_dir = Path("/Users/prateek/notes/uni-project-dec/datasets")
    
    for dataset_name in ["twitch", "event"]:
        print(f"\n{dataset_name.upper()} COMPATIBILITY ANALYSIS:")
        print("-" * 50)
        
        # Load synthetic data
        synth_file = datasets_dir / dataset_name / f"{dataset_name}_synth.pickle"
        with open(synth_file, "rb") as f:
            synth_data = pickle.load(f)
        
        # Analyze synthetic data properties
        synth_df, synth_graph = synth_data[0]
        
        print("Synthetic Data Properties:")
        print(f"  - Number of subgraphs: {len(synth_data)}")
        print(f"  - Nodes per subgraph: {synth_graph.number_of_nodes()}")
        print(f"  - Edges per subgraph: {synth_graph.number_of_edges()}")
        print(f"  - Edge density: {synth_graph.number_of_edges() / (synth_graph.number_of_nodes() * (synth_graph.number_of_nodes() - 1) / 2):.4f}")
        
        # Check feature distributions
        print("\nFeature Statistics (Synthetic):")
        for col in synth_df.columns:
            if synth_df[col].dtype in ['int64', 'float64']:
                print(f"  - {col}: mean={synth_df[col].mean():.2f}, std={synth_df[col].std():.2f}")
            elif synth_df[col].dtype == 'bool':
                print(f"  - {col}: True={synth_df[col].sum()}/{len(synth_df)} ({synth_df[col].mean()*100:.1f}%)")
        
        print("\nPotential Issues to Address:")
        print("  1. Feature scale differences between real and synthetic data")
        print("  2. Different graph density/structure properties")
        print("  3. Classification target distribution shifts")
        print("  4. Node feature correlation patterns")
        print("  5. Graph size variations (real data may have different sizes)")

def analyze_implementation_requirements():
    """Analyze what modifications are needed"""
    
    print("\n" + "="*80)
    print("IMPLEMENTATION REQUIREMENTS ANALYSIS")
    print("="*80)
    
    print("\n1. DATA CONVERSION PIPELINE:")
    print("-" * 30)
    print("  Required Components:")
    print("    - Function to convert (DataFrame, Graph) → torch_geometric.Data")
    print("    - Feature preprocessing (normalization, encoding)")
    print("    - Label extraction for classification targets")
    print("    - Edge index conversion from NetworkX to torch format")
    print("    - Batch loading for multiple subgraphs")
    
    print("\n2. rebMIGraph MODIFICATIONS:")
    print("-" * 30)
    print("  Required Changes:")
    print("    - Replace dataset loading with our custom data loader")
    print("    - Modify training loop to handle our data format")
    print("    - Separate data paths for target (real) vs shadow (synthetic)")
    print("    - Update evaluation to use real non-training data")
    print("    - Adapt to our classification targets (mature/gender)")
    
    print("\n3. EXPERIMENTAL SETUP:")
    print("-" * 30)
    print("  Required Implementation:")
    print("    - Cross-validation or multiple random splits")
    print("    - Statistical significance testing")
    print("    - Baseline comparisons (random guessing, simple ML)")
    print("    - Ablation studies (different GNN architectures)")
    print("    - Privacy metric calculations")
    
    print("\n4. EVALUATION FRAMEWORK:")
    print("-" * 30)
    print("  Metrics to Track:")
    print("    - Attack Accuracy, Precision, Recall, F1-score")
    print("    - AUROC for membership inference")
    print("    - False Positive Rate at different thresholds")
    print("    - Advantage over random guessing")
    print("    - Model utility metrics (original task performance)")

def analyze_critical_risks():
    """Identify potential showstoppers"""
    
    print("\n" + "="*80)
    print("CRITICAL RISKS AND MITIGATION STRATEGIES")
    print("="*80)
    
    print("\nRISK 1: Feature Distribution Mismatch")
    print("  Problem: Synthetic features may have different distributions than real")
    print("  Impact: Attack model trained on synthetic won't transfer to real")
    print("  Mitigation: Feature normalization, distribution matching")
    
    print("\nRISK 2: Graph Structure Differences")
    print("  Problem: Synthetic graphs may have different structural properties")
    print("  Impact: GNN models may behave differently on synthetic vs real")
    print("  Mitigation: Analyze graph metrics, adjust DLGrapher parameters")
    
    print("\nRISK 3: Insufficient Synthetic Data")
    print("  Problem: 256 synthetic samples may not cover real data diversity")
    print("  Impact: Poor attack transferability")
    print("  Mitigation: Generate more synthetic data, data augmentation")
    
    print("\nRISK 4: Classification Task Mismatch")
    print("  Problem: Synthetic data labels may not reflect real patterns")
    print("  Impact: Models may not learn meaningful patterns")
    print("  Mitigation: Validate label consistency, alternative targets")
    
    print("\nRISK 5: Pandas Version Compatibility")
    print("  Problem: Cannot load real data due to version mismatch")
    print("  Impact: Cannot validate synthetic-real similarity")
    print("  Mitigation: Create data loading utilities, version management")

if __name__ == "__main__":
    analyze_data_conversion_requirements()
    analyze_potential_compatibility_issues()
    analyze_implementation_requirements()
    analyze_critical_risks()