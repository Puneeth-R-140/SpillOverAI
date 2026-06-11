# SpilloverAI // Regional Network Intelligence Dashboard

SpilloverAI is an advanced entrepreneurship analytics platform designed to analyze, map, and predict startup success within regional innovation networks. Using Graph Deep Learning (Graph Convolutional Networks) and Network Topology analysis, the platform models knowledge flow and technology transfer dynamics across Startups, Incubators, and Universities.

## 🚀 Key Features

*   **Graph Deep Learning (GNN)**: Implements a 2-layer Graph Convolutional Network (GCN) using PyTorch Geometric (PyG) that learns node representation embeddings directly from ecosystem topology without developer-biased manual features.
*   **Scale Data Generation**: Includes a synthetic data generator modeling complex institutional tiers and three distinct startup profiles (*IP-Rich Bootstrappers*, *VC-Heavies*, and *Average Market Players*).
*   **Dynamic Interactive Dashboard**: Built with Streamlit and Plotly, featuring a wide-screen glassmorphic design system, point-selection click callbacks, localized 1-hop ego-graph visualization, and live GNN kernel activation breakdowns.

---

## 🛠️ Repository Architecture

```text
SpilloverAI/
│
├── data/
│   └── mock_ecosystem.json      # Initial seed mock ecosystem metadata
│
├── src/
│   ├── __init__.py              # Package initialization
│   ├── generator.py             # Large-scale probabilistic graph generator
│   ├── graph_engine.py          # PyTorch Geometric Data converter & NetworkX utilities
│   └── predictor.py             # PyTorch GCN architecture & training loops
│
├── app.py                       # Glassmorphic Streamlit Dashboard UI
├── main.py                      # CLI Pipeline execution & training script
├── .gitignore                   # Version control exclusions
└── README.md                    # Project documentation
```

---

## 💻 Setup & Installation Instructions

Follow these steps to set up and run SpilloverAI locally:

### 1. Clone the Repository
```bash
git clone https://github.com/Puneeth-R-140/SpillOverAI.git
cd SpillOverAI
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows (PowerShell)
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install torch torch-geometric networkx pandas scikit-learn streamlit plotly matplotlib
```
*(Note: On Windows, PyTorch Geometric will run on pure PyTorch CPU fallbacks automatically if binary wheels are missing).*

---

## 🏃 Execution Instructions

### A. Run CLI Pipeline (GNN Training & Metrics)
To execute the data generator, build the network, train the GNN for 150 epochs, and display the classification report in the console:
```bash
python main.py
```

### B. Run Streamlit Interactive Dashboard
To launch the premium glassmorphic dashboard interface:
```bash
streamlit run app.py
```
After the server initializes, navigate to **`http://localhost:8501`** in your web browser.

---

## 📊 Analytical Insights & Methodology

### Message-Passing Formulation
The GCN operates by convolving representations across node neighborhoods:
$$\mathbf{x}_i^{(k)} = \mathbf{W}^{(k)} \sum_{j \in \mathcal{N}(i) \cup \{i\}} \frac{1}{\sqrt{\deg(i)\deg(j)}} \mathbf{x}_j^{(k-1)}$$
By passing features (Normalized Funding, Patents, Node Type) along the edges, the network learns to identify startups enjoying academic spillovers without manual scoring inputs.
