import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.metrics import classification_report, f1_score

class GCN(torch.nn.Module):
    """
    A 2-layer Graph Convolutional Network (GCN) for node classification.
    Interspersed with ReLU activation and Dropout to prevent overfitting.
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

def train_gnn(data, epochs: int = 150):
    """
    Runs a PyTorch training loop for the GCN model on the provided PyG Data.
    Masks out non-startup nodes using train_mask during loss computation.
    """
    torch.manual_seed(42)
    
    # 5 input features, 16 hidden channels, 3 target classes
    model = GCN(in_channels=data.num_node_features, hidden_channels=16, out_channels=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    losses = []
    
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        out = model(data.x, data.edge_index)
        
        # Loss computation using only startup nodes in train_mask
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if epoch % 15 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}/{epochs:03d} | Train Loss: {loss.item():.4f}")
            
    # Evaluation
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        preds = out.argmax(dim=-1)
        
        # Get labels and predictions for test set mask
        y_test = data.y[data.test_mask].cpu().numpy()
        y_pred = preds[data.test_mask].cpu().numpy()
        
        target_names = ["Low", "Medium", "High"]
        report_str = classification_report(y_test, y_pred, target_names=target_names, zero_division=0)
        test_f1 = f1_score(y_test, y_pred, average="weighted")
        
    return model, losses, report_str, test_f1
