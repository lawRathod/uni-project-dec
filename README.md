# Theoretical Adaptation

## Adapting dataset
  1. The `rebMIGraph` source already runs attacks on several datasets, using `torch_geometric.data.Data` objects from `torch_geometric.datasets`.
  2. To use synthetic data for attacks, the following steps would be required:
     - Format synthetic subgraphs as `torch_geometric.data.Data` objects, to make sure node features, edge indices, and labels are present.
     - Use synthetic subgraphs for both the target and shadow models. Both models will be trained and evaluated exclusively on synthetic data.
     - **Settings**:
       - **TSTS**: Use different sets of synthetic subgraphs for the target and shadow models. Each model is trained/tested on its respective synthetic subgraphs.
       - **TSTF**: Requires a single large synthetic graph for each model (not just subgraphs and not available yet). This may require generating a full synthetic graph for both target and shadow settings using available checkpoint??
   3. Overall generating complete full synthetic graph and not samples would make things highly efficient and little modification to the original source for the attack would be required.

## Attack outline (with synthetic data only)
1. **Dataset Loading & Preparation**
   - Load all synthetic subgraphs for both the target and shadow models.
   - Ensure all data is in `torch_geometric.data.Data` format.
   - Set parameters for number of training/testing samples per class for both target and shadow models.

2. **Inductive Split (Train/Test Splits)**
   - Handled by `get_inductive_spilt()` in `rebMIGraph/TSTF.py`
     - Randomly select nodes for:
       - **Target Train**: For training the target model.
       - **Shadow Train**: For training the shadow model (simulates attacker’s knowledge).
       - **Target Test**: For evaluating the target model (out-of-training nodes).
       - **Shadow Test**: For evaluating the shadow model.
     - Ensures splits are based on class and do not overlap.
     - Creates subgraphs for target + shadow training using these splits.
     - Might require modifications because our synthetic data isn't a big subgraph but 256 sample subgraphs.

3. **Model Definition** (No modification)
   - Defines GNN models (GCN, SAGE, SGC, GAT) for both target and shadow.

4. **Model Training** (No modification)
   - **Target Model**: Trained on its assigned synthetic subgraphs (target train split).
   - **Shadow Model**: Trained on its assigned synthetic subgraphs (shadow train split).

5. **Model Testing**
   - Both models are evaluated on their respective test splits (subgraphs for TSTS, full synthetic graph for TSTF).

6. **Posterior Extraction**
   - Save model outputs (posteriors) for:
     - **Shadow In-Train** (shadow model, shadow train nodes from synthetic subgraphs)
     - **Shadow Out-Train** (shadow model, shadow test nodes from synthetic subgraphs)
     - **Target In-Train** (target model, target train nodes from synthetic subgraphs)
     - **Target Out-Train** (target model, target test nodes from synthetic subgraphs)

7. **Attack Data Construction**
   - **Positive samples**: Shadow In-Train posteriors 
   - **Negative samples**: Shadow Out-Train posteriors 
   - **Target In/Out posteriors** are also available for evaluation.

8. **Attack Model Training**
   - Train a neural network attack model to distinguish between "member" and "non-member" using the shadow model’s posteriors (from synthetic data).
   - Use `train_test_split` to split the attack data into attack-train and attack-test sets.

9. **Attack Evaluation**
   - Evaluate the attack model on:
     - The attack test set (from shadow/synthetic data).
     - The target model’s in/out posteriors (from synthetic data) for membership inference.

10. **Metrics & Logging**
    - Compute accuracy, AUROC, precision, recall, and F1 scores to measure performance of the attack.
    - Log results and save to files.

### General notes
- Most attack pipeline remains same as original attack and implementation. 
- Need to ensure that the data is correctly added and being used as expected by the project description.

## Experimentation
- Will require generating more synthetic data by tweaking paraments in DLGrapher.
- Running attack on different data and observing evaluation metrics.