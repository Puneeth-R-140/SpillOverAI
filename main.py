import os
import torch
from src.graph_engine import prepare_pyg_data
from src.predictor import train_gnn

def main():
    print("==================================================")
    print("     SpilloverAI GNN Deep Learning Pipeline       ")
    print("==================================================")
    
    scale_data_path = os.path.join("data", "scale_ecosystem.json")
    if not os.path.exists(scale_data_path):
        raise FileNotFoundError(f"Scale dataset not found at: {scale_data_path}")
        
    # 1. Prepare PyTorch Geometric Data
    print(f"\n[1/3] Converting {scale_data_path} to PyG Graph Structure...")
    data = prepare_pyg_data(scale_data_path)
    
    print("\n--- GNN Graph Dimensions ---")
    print(f"  Graph Data Object: {data}")
    print(f"  Total Nodes:       {data.num_nodes}")
    print(f"  Total Edges:       {data.num_edges}")
    print(f"  Node Feature Dim:  {data.num_node_features} ([Norm Funding, Patents, Type_Uni, Type_Inc, Type_Startup])")
    print(f"  Train Startups:    {data.train_mask.sum().item()}")
    print(f"  Test Startups:     {data.test_mask.sum().item()}")
    
    # 2. Train the Graph Convolutional Network (GCN)
    print("\n[2/3] Initializing and training 2-layer GCN model...")
    epochs = 150
    model, losses, report_str, test_f1 = train_gnn(data, epochs=epochs)
    
    # 3. Plotting the Training Loss Curve
    print("\n[3/3] Plotting GCN training loss curve...")
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, epochs + 1), losses, label="Train Loss", color="royalblue", lw=2)
        plt.xlabel("Epoch")
        plt.ylabel("CrossEntropy Loss")
        plt.title("SpilloverAI GNN Training Loss Curve")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plot_path = "loss_curve.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"  -> Loss curve successfully plotted and saved to '{plot_path}'")
    except ImportError:
        print("  -> Matplotlib not installed. Skipping plot generation.")
        
    print("\n" + "=" * 80)
    print("                           GNN PERFORMANCE REPORT                                 ")
    print("=" * 80)
    print(f"Trained Model: 2-Layer GCNConv Model")
    print(f"Overall Test Set F1-Score (Weighted): {test_f1:.4f}\n")
    print("Classification Report on Masked Test Nodes:")
    print(report_str)
    
    print("[CONFIRMATION] GNN utilizing structural graph connectivity (edge_index) directly")
    print("               for node representation learning and classification.")
    print("==================================================")

if __name__ == "__main__":
    main()
