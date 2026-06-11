import json
import torch
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
import numpy as np

def build_network(data_path: str):
    """
    Dummy/fallback for the old networkX graph engine loader.
    """
    import networkx as nx
    with open(data_path, "r") as f:
        data = json.load(f)
    G = nx.Graph()
    for node in data["nodes"]:
        G.add_node(node["id"], **node)
    for link in data["links"]:
        G.add_edge(link["source"], link["target"], weight=link.get("weight", 1.0))
    return G

def prepare_pyg_data(data_path: str) -> Data:
    """
    Parses ecosystem JSON and constructs a PyTorch Geometric Data object.
    
    Node Feature Matrix X:
      - Column 0: Normalized Funding (0 for institutions)
      - Column 1: Patent Count (0 for institutions)
      - Column 2-4: One-hot encoded Node Type [University, Incubator, Startup]
      
    Target Labels Y:
      - Low = 0, Medium = 1, High = 2
      - Non-startups = -1 (filtered out in training/eval)
    """
    with open(data_path, "r") as f:
        data = json.load(f)
        
    nodes = data["nodes"]
    links = data["links"]
    
    # 1. Map Node IDs to integer indices
    node_id_map = {node["id"]: idx for idx, node in enumerate(nodes)}
    num_nodes = len(nodes)
    
    # 2. Extract features
    # Get max/min funding and patents for startups for normalization
    startup_fundings = [n.get("funding", 0.0) for n in nodes if n.get("type") == "Startup"]
    max_funding = max(startup_fundings) if startup_fundings else 1.0
    min_funding = min(startup_fundings) if startup_fundings else 0.0
    fund_denom = (max_funding - min_funding) if (max_funding - min_funding) > 0 else 1.0
    
    x_list = []
    y_list = []
    startup_indices = []
    startup_labels = []
    
    label_map = {"Low": 0, "Medium": 1, "High": 2}
    
    for idx, node in enumerate(nodes):
        node_type = node.get("type")
        
        # Normalized funding
        raw_funding = node.get("funding", 0.0)
        norm_funding = (raw_funding - min_funding) / fund_denom if node_type == "Startup" else 0.0
        
        # Patents
        patents = float(node.get("patent_count", 0)) if node_type == "Startup" else 0.0
        
        # One-hot node type: [University, Incubator, Startup]
        if node_type == "University":
            type_onehot = [1.0, 0.0, 0.0]
            y_val = -1
        elif node_type == "Incubator":
            type_onehot = [0.0, 1.0, 0.0]
            y_val = -1
        else: # Startup
            type_onehot = [0.0, 0.0, 1.0]
            y_val = label_map.get(node.get("label", "Low"), 0)
            startup_indices.append(idx)
            startup_labels.append(y_val)
            
        x_list.append([norm_funding, patents] + type_onehot)
        y_list.append(y_val)
        
    X = torch.tensor(x_list, dtype=torch.float)
    Y = torch.tensor(y_list, dtype=torch.long)
    
    # 3. Construct edge_index (undirected)
    edge_sources = []
    edge_targets = []
    
    for link in links:
        src_idx = node_id_map[link["source"]]
        dst_idx = node_id_map[link["target"]]
        
        # Undirected: add both directions
        edge_sources.extend([src_idx, dst_idx])
        edge_targets.extend([dst_idx, src_idx])
        
    edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
    
    # 4. Perform Stratified Train-Test Split (80% Train, 20% Test) on Startups
    train_idx, test_idx = train_test_split(
        startup_indices, 
        test_size=0.20, 
        stratify=startup_labels, 
        random_state=42
    )
    
    # Create masks
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[train_idx] = True
    test_mask[test_idx] = True
    
    # 5. Build PyG Data Object
    pyg_data = Data(
        x=X,
        edge_index=edge_index,
        y=Y,
        train_mask=train_mask,
        test_mask=test_mask
    )
    
    return pyg_data
