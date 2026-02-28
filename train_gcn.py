import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import kneighbors_graph
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import os

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# Task 1: Static Graph Construction
# ==========================================
def load_and_build_graph(csv_path: str) -> Data:
    df = pd.read_csv(csv_path)
    
    # Store Loan_ID for eventual export
    loan_ids = df['Loan_ID'].values if 'Loan_ID' in df.columns else np.arange(len(df))
    
    # Target Processing (Loan_Status)
    if 'Loan_Status' in df.columns:
        df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})
        y = torch.tensor(df['Loan_Status'].values, dtype=torch.long)
    else:
        y = None

    # Base continuous columns
    base_cont_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
    for col in base_cont_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
            
    # FEATURE ENGINEERING
    if 'ApplicantIncome' in df.columns and 'CoapplicantIncome' in df.columns:
        df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
        df['TotalIncome_log'] = np.log((df['TotalIncome'] + 1).astype(float))
    
    if 'LoanAmount' in df.columns:
        df['LoanAmount_log'] = np.log((df['LoanAmount'] + 1).astype(float))
        
    if 'LoanAmount' in df.columns and 'Loan_Amount_Term' in df.columns:
        # Calculate EMI. If term is 0, EMI is 0.
        df['EMI'] = np.where(df['Loan_Amount_Term'] == 0, 0, df['LoanAmount'] / df['Loan_Amount_Term'])
        df['EMI'] = df['EMI'].fillna(0)
        # Balance Income
        if 'TotalIncome' in df.columns:
            df['BalanceIncome'] = df['TotalIncome'] - (df['EMI'] * 1000)

    # Gather all available continuous features
    all_cont_cols = [
        'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
        'TotalIncome', 'TotalIncome_log', 'LoanAmount_log', 'EMI', 'BalanceIncome'
    ]
    cont_cols = [c for c in all_cont_cols if c in df.columns]
    
    # Extract continuous features & normalize
    scaler = StandardScaler()
    X_cont = scaler.fit_transform(df[cont_cols])
    
    # Build Edges (KNN) dynamically based on continuous features (k=5)
    A = kneighbors_graph(X_cont, n_neighbors=5, mode='distance', include_self=False)
    
    # Extract edge indices and distances via COO format to maintain explicit zeros
    A_coo = A.tocoo()
    edge_index = torch.tensor(np.vstack([A_coo.row, A_coo.col]), dtype=torch.long)
    distances = A_coo.data
    
    # Assign Edge Weights (Inverse of Euclidean Distance)
    edge_weights = 1.0 / (distances + 1e-8)
    edge_attr = torch.tensor(edge_weights, dtype=torch.float32)

    # PREPROCESSING CATEGORICAL FEATURES
    cat_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 
                'Credit_History', 'Property_Area']
    
    # Fill NAs in categorical columns with the mode
    for col in cat_cols:
        if col in df.columns:
            # handle case where mode() is empty
            mode_vals = df[col].mode()
            if len(mode_vals) > 0:
                df[col] = df[col].fillna(mode_vals[0])
            else:
                df[col] = df[col].fillna("Unknown")
            
    # Dummy encoding (One-Hot)
    existing_cats = [c for c in cat_cols if c in df.columns]
    df_cat = pd.get_dummies(df[existing_cats], drop_first=True)
    X_cat = df_cat.values
    
    # Node features: Concatenate normalized continuous and encoded categoricals
    X_features = np.hstack((X_cont, X_cat.astype(float)))
    x = torch.tensor(X_features, dtype=torch.float32)

    # Create torch_geometric Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data.loan_ids = loan_ids
    
    return data

# ==========================================
# Task 2: GCN Architecture
# ==========================================
class RiskGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(RiskGCN, self).__init__()
        # 2-layer GCNConv
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_weight=None):
        # Layer 1
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        # Layer 2
        x = self.conv2(x, edge_index, edge_weight)
        return x # Return logits

def train_gcn(data: Data):
    num_nodes = data.x.shape[0]
    
    # Train / Validation Node Mask (e.g., 80% Train, 20% Val)
    indices = np.arange(num_nodes)
    np.random.shuffle(indices)
    split_idx = int(num_nodes * 0.8)
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[indices[:split_idx]] = True
    val_mask[indices[split_idx:]] = True
    
    data.train_mask = train_mask
    data.val_mask = val_mask

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    
    model = RiskGCN(in_channels=data.x.shape[1], hidden_channels=16, out_channels=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    # Training Loop
    model.train()
    for epoch in range(1, 101):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr)
        # We compute loss only on training nodes
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            # Validation Step
            model.eval()
            with torch.no_grad():
                val_out = model(data.x, data.edge_index, data.edge_attr)
                val_loss = F.cross_entropy(val_out[data.val_mask], data.y[data.val_mask])
                pred = val_out[data.val_mask].argmax(dim=1)
                acc = (pred == data.y[data.val_mask]).sum().item() / data.val_mask.sum().item()
            print(f"Epoch {epoch:3d} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f} | Val Acc: {acc:.4f}")
            model.train()

    return model, data, device

# ==========================================
# Task 3: O(1) Lookup Export
# ==========================================
def export_risk_scores(model, data, device, output_file="gcn_scores.json"):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index, data.edge_attr)
        # Probabilities via softmax
        probs = F.softmax(out, dim=1)
        
        # We output Risk probability: 
        # Typically class 0 is 'N' (Denied), which represents "High Risk".
        # Therefore, probability of predicting class 0 is our risk probability.
        risk_probs = probs[:, 0].cpu().numpy().astype(float)
        
    # Combine with Loan_IDs
    # O(1) Lookup structure: Dictionary mapping expected by backend
    gcn_scores = {}
    for loan_id, rp in zip(data.loan_ids, risk_probs):
        gcn_scores[loan_id] = rp
        
    with open(output_file, 'w') as f:
        json.dump(gcn_scores, f, indent=4)
        
    print(f"\n[+] Successfully exported risk scores to: {output_file}")
    print(f"[+] Total entries embedded: {len(gcn_scores)}")

if __name__ == "__main__":
    csv_path = 'train_u6lujuX_CVtuZ9i.csv'
    if not os.path.exists(csv_path):
        print(f"Error: Could not find {csv_path} in the current directory.")
        exit(1)
        
    print(f"Building Data object from {csv_path}...")
    data = load_and_build_graph(csv_path)
    print(data)

    print("\nTraining GCN Model...")
    model, processed_data, device = train_gcn(data)
    
    print("\nComputing probabilities and exporting to JSON...")
    export_risk_scores(model, processed_data, device, output_file="gcn_scores.json")
