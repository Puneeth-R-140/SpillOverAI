import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import torch
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from src.graph_engine import prepare_pyg_data
from src.predictor import train_gnn, GCN

# Page config
st.set_page_config(
    page_title="SPILLOVER // Network Intelligence Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Custom Styling Block (Google Fonts + Glassmorphism + Accent Glows)
st.markdown("""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Space+Grotesk:wght@400;500;700&display=swap" rel="stylesheet">

    <style>
    /* Base theme overrides */
    .main {
        background: radial-gradient(circle at 50% 10%, #1e1b4b 0%, #0f172a 60%, #020617 100%);
        color: #f8fafc;
        font-family: 'Outfit', sans-serif;
    }
    
    /* Header & Title Styling */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Space Grotesk', sans-serif;
        color: #f1f5f9;
        letter-spacing: -0.02em;
    }
    
    /* Custom Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.9) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(12px);
    }
    
    /* Premium Glassmorphic Cards */
    .premium-card {
        background: rgba(30, 41, 59, 0.4);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.5);
        margin-bottom: 20px;
        transition: transform 0.2s ease, border 0.2s ease;
    }
    .premium-card:hover {
        border: 1px solid rgba(99, 102, 241, 0.4);
        transform: translateY(-2px);
    }
    
    /* Customized Metric Headers */
    .metric-title {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #94a3b8;
        margin-bottom: 6px;
        font-weight: 600;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        font-family: 'Space Grotesk', sans-serif;
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-badge {
        display: inline-block;
        padding: 4px 8px;
        border-radius: 6px;
        font-size: 0.7rem;
        font-weight: bold;
        background: rgba(99, 102, 241, 0.15);
        color: #818cf8;
        border: 1px solid rgba(99, 102, 241, 0.3);
        margin-top: 8px;
    }
    
    /* Sleek top navigation bar */
    .top-navbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 16px 24px;
        background: rgba(15, 23, 42, 0.6);
        backdrop-filter: blur(10px);
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        margin-bottom: 30px;
    }
    .nav-brand {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.4rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        background: linear-gradient(90deg, #6366f1, #38bdf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .status-pulse {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background-color: #10b981;
        box-shadow: 0 0 10px #10b981;
        animation: pulse-animation 2s infinite;
        display: inline-block;
    }
    @keyframes pulse-animation {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }
    .status-badge {
        font-size: 0.8rem;
        font-weight: bold;
        color: #38bdf8;
        background: rgba(56, 189, 248, 0.1);
        border: 1px solid rgba(56, 189, 248, 0.2);
        padding: 6px 12px;
        border-radius: 20px;
        letter-spacing: 0.05em;
    }
    
    /* Native streamlit tab overrides */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        border-radius: 8px !important;
        background-color: rgba(30, 41, 59, 0.3) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        color: #94a3b8 !important;
        font-family: 'Space Grotesk', sans-serif !important;
        padding: 0px 20px !important;
        font-weight: 500 !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(99,102,241,0.2) 0%, rgba(56,189,248,0.2) 100%) !important;
        border: 1px solid rgba(99, 102, 241, 0.4) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        box-shadow: 0 0 15px rgba(99, 102, 241, 0.15);
    }
    </style>
""", unsafe_allow_html=True)

DATA_PATH = os.path.join("data", "scale_ecosystem.json")

# 1. Load Data and Train GNN live
@st.cache_resource
def load_and_train_pipeline():
    if not os.path.exists(DATA_PATH):
        from src.generator import generate_ecosystem
        generate_ecosystem(DATA_PATH)
        
    pyg_data = prepare_pyg_data(DATA_PATH)
    model, losses, report_str, test_f1 = train_gnn(pyg_data, epochs=150)
    
    with open(DATA_PATH, "r") as f:
        raw_data = json.load(f)
        
    return pyg_data, model, losses, test_f1, raw_data

try:
    data, model, losses, f1_score, raw_json = load_and_train_pipeline()
except Exception as e:
    st.error(f"Error initializing pipeline: {e}")
    st.stop()

# Build NetworkX graph for topology visualization
@st.cache_data
def get_nx_layout(raw_data):
    G = nx.Graph()
    for node in raw_data["nodes"]:
        G.add_node(node["id"], **node)
    for link in raw_data["links"]:
        G.add_edge(link["source"], link["target"], weight=link.get("weight", 1.0))
        
    # Spring layout coordinates
    pos = nx.spring_layout(G, seed=42, k=0.15, iterations=50)
    return G, pos

G, pos = get_nx_layout(raw_json)

# Model predictions extraction
model.eval()
with torch.no_grad():
    out = model(data.x, data.edge_index)
    probs = torch.softmax(out, dim=-1).numpy()
    preds = out.argmax(dim=-1).numpy()

label_names = ["Low", "Medium", "High"]

# Prepare DataFrame of startups
startups_list = []
for idx, node in enumerate(raw_json["nodes"]):
    if node.get("type") == "Startup":
        startups_list.append({
            "entity_id": node["id"],
            "name": node.get("name", node["id"]),
            "profile": node.get("profile", "Average Market Player"),
            "funding": node.get("funding", 0.0),
            "patents": node.get("patent_count", 0),
            "label": node.get("label", "Low"),
            "predicted_label": label_names[preds[idx]],
            "prob_low": probs[idx][0],
            "prob_med": probs[idx][1],
            "prob_high": probs[idx][2],
            "node_idx": idx
        })

df_startups = pd.DataFrame(startups_list)

# 2. Sidebar controls
st.sidebar.markdown(
    '<div style="font-family:\'Space Grotesk\'; font-size:1.6rem; font-weight:700; color:#818cf8; margin-bottom:20px;">SPILLOVER.AI</div>',
    unsafe_allow_html=True
)

profiles = df_startups["profile"].unique().tolist()
selected_profiles = st.sidebar.multiselect(
    "Filter Startups by Profile",
    options=profiles,
    default=profiles
)

threshold = st.sidebar.slider(
    "GNN Classification Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    help="Modifies predictions to High-Success if probability exceeds this threshold."
)

# Apply threshold filter to predictions
def apply_threshold(row):
    if row["prob_high"] >= threshold:
        return "High"
    elif row["prob_low"] > row["prob_med"]:
        return "Low"
    else:
        return "Medium"

df_startups["adjusted_pred"] = df_startups.apply(apply_threshold, axis=1)

# Header navbar
st.markdown("""
    <div class="top-navbar">
        <div class="nav-brand">
            <span class="status-pulse"></span>
            SPILLOVER // NETWORK INTELLIGENCE
        </div>
        <div class="status-badge">GNN ENGINE ACTIVE</div>
    </div>
""", unsafe_allow_html=True)

# Top custom metric cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
        <div class="premium-card">
            <div class="metric-title">Ecosystem Nodes</div>
            <div class="metric-value">{len(raw_json["nodes"])}</div>
            <div class="metric-badge">Heterogeneous Nodes</div>
        </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
        <div class="premium-card">
            <div class="metric-title">Ecosystem Links</div>
            <div class="metric-value">{len(raw_json["links"])}</div>
            <div class="metric-badge">IPR & Tech Transfers</div>
        </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown(f"""
        <div class="premium-card">
            <div class="metric-title">Graph Density</div>
            <div class="metric-value">{nx.density(G):.5f}</div>
            <div class="metric-badge">Edge Cohesion Metric</div>
        </div>
    """, unsafe_allow_html=True)
with col4:
    st.markdown(f"""
        <div class="premium-card">
            <div class="metric-title">GNN F1-Score</div>
            <div class="metric-value">{f1_score:.4f}</div>
            <div class="metric-badge">Stratified Validation</div>
        </div>
    """, unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["Ecosystem Network Map", "GNN Predictive Analytics", "Model Metrics & IPR Proof"])

# Tab 1: Ecosystem Network Map
with tab1:
    st.subheader("Regional Knowledge Spillover Map")
    st.markdown("💡 *Click on any node in the map to select it and inspect its deep GNN properties and localized sub-graph.*")
    
    # Filter nodes for map highlight
    filtered_startups = set(df_startups[df_startups["profile"].isin(selected_profiles)]["entity_id"])
    
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1.2, color='rgba(99, 102, 241, 0.25)'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Node coordinates, types, PageRank, and Degree Centrality
    nodes_data = {
        "x": [], "y": [], "color": [], "size": [],
        "id": [], "name": [], "type": [], "deg_cent": [], "pr": []
    }
    
    # Calculate PageRank and Degree Centrality
    pagerank = nx.pagerank(G, weight="weight")
    deg_centrality = nx.degree_centrality(G)
    
    for node_id in G.nodes():
        node_attrs = G.nodes[node_id]
        node_type = node_attrs.get("type", "Startup")
        x, y = pos[node_id]
        
        nodes_data["x"].append(x)
        nodes_data["y"].append(y)
        nodes_data["id"].append(node_id)
        nodes_data["name"].append(node_attrs.get("name", node_id))
        nodes_data["type"].append(node_type)
        nodes_data["deg_cent"].append(deg_centrality.get(node_id, 0.0))
        
        pr_score = pagerank.get(node_id, 0.0)
        nodes_data["pr"].append(pr_score)
        
        # Color coding: Uni=Gold, Incubator=Emerald, Startups=Neon Blue
        if node_type == "University":
            nodes_data["color"].append("#f59e0b") # Gold
            nodes_data["size"].append(28 + pr_score * 300)
        elif node_type == "Incubator":
            nodes_data["color"].append("#10b981") # Emerald
            nodes_data["size"].append(20 + pr_score * 200)
        else: # Startup
            is_active = node_id in filtered_startups
            # Neon Blue if active, dark slate grey if filtered
            color = "#6366f1" if is_active else "rgba(71, 85, 105, 0.3)"
            size = 11 if is_active else 6
            nodes_data["color"].append(color)
            nodes_data["size"].append(size)
            
    customdata_nodes = list(zip(
        nodes_data["id"],
        nodes_data["name"],
        nodes_data["type"],
        nodes_data["deg_cent"],
        nodes_data["pr"]
    ))
    
    node_trace = go.Scatter(
        x=nodes_data["x"],
        y=nodes_data["y"],
        mode='markers',
        customdata=customdata_nodes,
        hovertemplate=(
            "<b>%{customdata[1]}</b><br>"
            "Type: %{customdata[2]}<br>"
            "Degree Centrality: %{customdata[3]:.4f}<br>"
            "PageRank Score: %{customdata[4]:.4f}<br>"
            "<extra></extra>"
        ),
        marker=dict(
            color=nodes_data["color"],
            size=nodes_data["size"],
            line=dict(width=1.5, color='#0f172a'),
            colorscale="Viridis"
        )
    )
    
    fig = go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=0, l=0, r=0, t=0),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    height=550
                 ))
                 
    # Interactive Event Chart
    map_event = st.plotly_chart(
        fig,
        use_container_width=True,
        on_select="rerun",
        selection_mode="points",
        key="network_map"
    )
    
    selected_node = None
    if map_event and "selection" in map_event:
        points = map_event["selection"].get("points", [])
        if points:
            selected_node = points[0].get("customdata", [None])[0]
            
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Dynamic Inspection panel
    if selected_node:
        col_d1, col_d2 = st.columns(2)
        
        with col_d1:
            node_attrs = G.nodes[selected_node]
            node_type = node_attrs.get("type", "Startup")
            
            st.markdown(f"""
                <div class="premium-card">
                    <h3 style="margin-top:0;">🔍 Inspection: {node_attrs.get('name', selected_node)}</h3>
                    <p style="color:#94a3b8; font-size:0.9rem;">Entity ID: <code>{selected_node}</code> | Type: <b>{node_type}</b></p>
                </div>
            """, unsafe_allow_html=True)
            
            if node_type == "Startup":
                startup_row = df_startups[df_startups["entity_id"] == selected_node]
                if not startup_row.empty:
                    s_data = startup_row.iloc[0]
                    
                    st.markdown("#### Operational Metrics")
                    cm1, cm2, cm3 = st.columns(3)
                    with cm1:
                        st.markdown(f'<div class="premium-card"><div class="metric-title">Funding</div><div class="metric-value" style="font-size:1.6rem;">${s_data["funding"]:.1f}M</div></div>', unsafe_allow_html=True)
                    with cm2:
                        st.markdown(f'<div class="premium-card"><div class="metric-title">Patents</div><div class="metric-value" style="font-size:1.6rem;">{int(s_data["patents"])}</div></div>', unsafe_allow_html=True)
                    with cm3:
                        st.markdown(f'<div class="premium-card"><div class="metric-title">GNN Prediction</div><div class="metric-value" style="font-size:1.6rem; color:#818cf8;">{s_data["adjusted_pred"]}</div></div>', unsafe_allow_html=True)
                    
                    st.markdown("#### Message-Passing Confidence Scores")
                    st.progress(float(s_data['prob_high']), text=f"High Success Probability: {s_data['prob_high']:.2%}")
                    st.progress(float(s_data['prob_med']), text=f"Medium Success Probability: {s_data['prob_med']:.2%}")
                    st.progress(float(s_data['prob_low']), text=f"Low Success Probability: {s_data['prob_low']:.2%}")
            else:
                st.markdown("#### Topological Centralities")
                cm1, cm2 = st.columns(2)
                with cm1:
                    st.markdown(f'<div class="premium-card"><div class="metric-title">PageRank Score</div><div class="metric-value" style="font-size:1.6rem;">{pagerank.get(selected_node, 0.0):.6f}</div></div>', unsafe_allow_html=True)
                with cm2:
                    st.markdown(f'<div class="premium-card"><div class="metric-title">Ecosystem Connections</div><div class="metric-value" style="font-size:1.6rem;">{G.degree(selected_node)}</div></div>', unsafe_allow_html=True)
                    
                if "tier" in node_attrs:
                    st.markdown(f'<div class="premium-card"><div class="metric-title">Institutional Tier</div><div class="metric-value" style="font-size:1.4rem; color:#f59e0b;">{node_attrs["tier"]}</div></div>', unsafe_allow_html=True)
                    
        with col_d2:
            st.markdown(f"### Localized 1-Hop Sub-Graph: **{selected_node}**")
            
            ego_G = nx.ego_graph(G, selected_node, radius=1)
            ego_pos = nx.spring_layout(ego_G, seed=42)
            
            sub_edge_x = []
            sub_edge_y = []
            for edge in ego_G.edges():
                x0, y0 = ego_pos[edge[0]]
                x1, y1 = ego_pos[edge[1]]
                sub_edge_x.extend([x0, x1, None])
                sub_edge_y.extend([y0, y1, None])
                
            sub_edge_trace = go.Scatter(
                x=sub_edge_x, y=sub_edge_y,
                line=dict(width=1.5, color='rgba(99, 102, 241, 0.3)'),
                hoverinfo='none',
                mode='lines'
            )
            
            sub_nodes = {"x": [], "y": [], "color": [], "size": [], "text": []}
            for node_id in ego_G.nodes():
                x, y = ego_pos[node_id]
                sub_nodes["x"].append(x)
                sub_nodes["y"].append(y)
                
                n_type = ego_G.nodes[node_id].get("type", "Startup")
                n_name = ego_G.nodes[node_id].get("name", node_id)
                sub_nodes["text"].append(f"{n_name} ({n_type})")
                
                if node_id == selected_node:
                    sub_nodes["color"].append("#ef4444") # Center highlighted in neon red
                    sub_nodes["size"].append(24)
                else:
                    if n_type == "University":
                        sub_nodes["color"].append("#f59e0b")
                        sub_nodes["size"].append(18)
                    elif n_type == "Incubator":
                        sub_nodes["color"].append("#10b981")
                        sub_nodes["size"].append(14)
                    else:
                        sub_nodes["color"].append("#3b82f6")
                        sub_nodes["size"].append(10)
                        
            sub_node_trace = go.Scatter(
                x=sub_nodes["x"], y=sub_nodes["y"],
                mode='markers+text',
                text=sub_nodes["text"],
                textposition="top center",
                hoverinfo='text',
                marker=dict(
                    color=sub_nodes["color"],
                    size=sub_nodes["size"],
                    line=dict(width=1.5, color='#0f172a')
                )
            )
            
            sub_fig = go.Figure(data=[sub_edge_trace, sub_node_trace],
                             layout=go.Layout(
                                showlegend=False,
                                margin=dict(b=0, l=0, r=0, t=0),
                                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                height=380
                             ))
            st.plotly_chart(sub_fig, use_container_width=True)
    else:
        st.markdown("""
            <div class="premium-card" style="text-align: center; border: 1px dashed rgba(255, 255, 255, 0.15);">
                <p style="color: #94a3b8; font-size: 1.1rem;">💡 <b>Interaction Tip:</b> Click on any node inside the regional network map above to inspect its properties and render its localized sub-graph.</p>
            </div>
        """, unsafe_allow_html=True)

# Tab 2: GNN Predictive Analytics
with tab2:
    st.subheader("GNN Node Predictions & Message-Passing Analytics")
    
    df_filtered = df_startups[df_startups["profile"].isin(selected_profiles)]
    col_t1, col_t2 = st.columns([3, 2])
    
    with col_t1:
        st.markdown("### GNN Predictions Table")
        df_display = df_filtered[["entity_id", "profile", "funding", "patents", "label", "adjusted_pred", "prob_high"]].copy()
        df_display.columns = ["Startup ID", "Profile", "Funding ($M)", "Patent Count", "Actual Label", "GNN Adjusted Pred", "Success Prob"]
        
        fig_table = go.Figure(data=[go.Table(
            header=dict(values=list(df_display.columns),
                        fill_color='#1e293b',
                        align='left',
                        font=dict(color='#f8fafc', size=12, family='Space Grotesk')),
            cells=dict(values=[df_display[col] for col in df_display.columns],
                       fill_color='rgba(15, 23, 42, 0.4)',
                       align='left',
                       font=dict(color='#e2e8f0', size=11))
        )])
        fig_table.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            height=420,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_table, use_container_width=True)
        
    with col_t2:
        st.markdown("### Message-Passing Probability Distribution")
        fig_bar = px.histogram(
            df_filtered,
            x="prob_high",
            color="adjusted_pred",
            labels={"prob_high": "GNN Success Probability", "count": "Number of Startups"},
            title="GNN Success Classification Confidence Spread",
            color_discrete_map={"High": "#6366f1", "Medium": "#38bdf8", "Low": "#f87171"},
            template="plotly_dark"
        )
        fig_bar.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=50, b=10, l=10, r=10)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

# Tab 3: Model Metrics & IPR Proof
with tab3:
    st.subheader("Model Validation & Intellectual Property Proof")
    col_m1, col_m2 = st.columns(2)
    
    with col_m1:
        st.markdown("### Interactive Training Loss History")
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(
            x=list(range(1, len(losses) + 1)),
            y=losses,
            mode='lines',
            name='Train Loss',
            line=dict(color='#818cf8', width=2.5)
        ))
        fig_loss.update_layout(
            xaxis_title="Epoch",
            yaxis_title="CrossEntropy Loss",
            template="plotly_dark",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=10, t=30, b=10)
        )
        st.plotly_chart(fig_loss, use_container_width=True)
        
    with col_m2:
        st.markdown("### Mathematical Proof of Network Spillover")
        st.markdown(r"""
        The Graph Convolutional Network (GNN) operates **without being fed any hand-crafted topological centralities**. 
        Instead, it aggregates feature representations dynamically from neighboring nodes via message-passing convolutions:
        
        $$\mathbf{x}_i^{(k)} = \mathbf{W}^{(k)} \sum_{j \in \mathcal{N}(i) \cup \{i\}} \frac{1}{\sqrt{\deg(i)\deg(j)}} \mathbf{x}_j^{(k-1)}$$
        
        By convolving neighborhood properties (such as tiered University knowledge proximity and Incubator linkages), the model discovers implicit clusters without developer-biased features.
        """)
        
        # Calculate weights of GCN to show feature activation
        with torch.no_grad():
            weights = model.conv1.lin.weight.abs().mean(dim=0).numpy()
        
        feat_names = ["Normalized Funding", "Patent Count", "Uni Node Type", "Incubator Node Type", "Startup Node Type"]
        df_weights = pd.DataFrame({
            "Graph Feature Attribute": feat_names,
            "GCN Conv Kernel Activation Strength": weights
        }).sort_values(by="GCN Conv Kernel Activation Strength", ascending=True)
        
        fig_weights = px.bar(
            df_weights,
            x="GCN Conv Kernel Activation Strength",
            y="Graph Feature Attribute",
            orientation='h',
            title="GCN Conv1 Layer Activation Magnitudes",
            color="GCN Conv Kernel Activation Strength",
            color_continuous_scale="Viridis",
            template="plotly_dark"
        )
        fig_weights.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig_weights, use_container_width=True)
