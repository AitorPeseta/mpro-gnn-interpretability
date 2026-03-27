"""
═══════════════════════════════════════════════════════════════════════
SCRIPT DE ENTRENAMIENTO CON MODELOS EDGE-AWARE
Versión Optimizada - Limpieza de RAM + Modelo Químico CGConv
═══════════════════════════════════════════════════════════════════════
"""

import os
import json
import gc

import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d, Dropout
from torch_geometric.loader import DataLoader

# Importamos CGConv (Especial para Cristalografía y Química) en lugar de NNConv
from torch_geometric.nn import (
    CGConv,
    GATv2Conv,
    GINEConv,
    TransformerConv,
    global_mean_pool
)

import optuna
from complex_dataset_loader import MProComplexDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATASET_MODE = 'complex'
DATASET_PATH = './data'
RESULTS_ROOT = f"results"
N_TRIALS = 50
EDGE_DIM = 13

# =====================================================================
# ARQUITECTURAS DE LAS REDES NEURONALES (EL CEREBRO)
# =====================================================================

class CGConv_Model(torch.nn.Module):
    """
    Modelo 1: CGConv (Crystal Graph Convolution).
    El rey de la química computacional. Lee las distancias y enlaces sin asfixiar la CPU.
    """
    def __init__(self, in_channels, hidden_channels, num_layers, dropout):
        super().__init__()
        
        # CGConv exige que los nodos tengan el mismo tamaño matemático desde el principio,
        # así que usamos esta capa "Linear" para adaptar el tamaño de entrada.
        self.node_proj = Linear(in_channels, hidden_channels)
        
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        
        for _ in range(num_layers):
            self.convs.append(CGConv(channels=hidden_channels, dim=EDGE_DIM, batch_norm=True))
            self.bns.append(BatchNorm1d(hidden_channels))
            
        self.lin1 = Linear(hidden_channels, hidden_channels // 2)
        self.lin2 = Linear(hidden_channels // 2, 1)
        self.dropout = dropout
    
    def forward(self, x, edge_index, edge_attr, batch):
        x = self.node_proj(x) # Ajustamos el tamaño inicial
        
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin2(x)

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, dropout, heads=4):
        super(GAT, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATv2Conv(in_channels=in_channels, out_channels=hidden_channels, heads=heads, concat=True, edge_dim=EDGE_DIM))
        for _ in range(num_layers - 1):
            self.convs.append(GATv2Conv(in_channels=hidden_channels * heads, out_channels=hidden_channels, heads=heads, concat=True, edge_dim=EDGE_DIM))
        self.lin = Linear(hidden_channels * heads, 1)
        self.dropout = dropout
    
    def forward(self, x, edge_index, edge_attr, batch):
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        return self.lin(x)

class GINE_Model(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, dropout):
        super(GINE_Model, self).__init__()
        self.node_encoder = Sequential(Linear(in_channels, hidden_channels), BatchNorm1d(hidden_channels), ReLU())
        self.edge_encoder = Sequential(Linear(EDGE_DIM, hidden_channels), ReLU())
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        for _ in range(num_layers):
            mlp = Sequential(Linear(hidden_channels, hidden_channels * 2), BatchNorm1d(hidden_channels * 2), ReLU(), Dropout(dropout), Linear(hidden_channels * 2, hidden_channels))
            self.convs.append(GINEConv(mlp, train_eps=True))
            self.bns.append(BatchNorm1d(hidden_channels))
        self.lin1 = Linear(hidden_channels, hidden_channels // 2)
        self.lin2 = Linear(hidden_channels // 2, 1)
        self.dropout = dropout
    
    def forward(self, x, edge_index, edge_attr, batch):
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return x

class GT_Model(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, dropout, heads=4):
        super(GT_Model, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.convs.append(TransformerConv(in_channels=in_channels, out_channels=hidden_channels, heads=heads, concat=True, beta=True, edge_dim=EDGE_DIM, dropout=dropout))
        self.bns.append(BatchNorm1d(hidden_channels * heads))
        for _ in range(num_layers - 1):
            self.convs.append(TransformerConv(in_channels=hidden_channels * heads, out_channels=hidden_channels, heads=heads, concat=True, beta=True, edge_dim=EDGE_DIM, dropout=dropout))
            self.bns.append(BatchNorm1d(hidden_channels * heads))
        self.lin1 = Linear(hidden_channels * heads, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)
        self.dropout = dropout
    
    def forward(self, x, edge_index, edge_attr, batch):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return x

def build_model(model_name, in_channels, params):
    h = params['hidden_channels']
    l = params['num_layers']
    d = params['dropout']
    hd = params.get('heads', 4)
    
    # ---------------- CAMBIO CLAVE AQUÍ ----------------
    if model_name == 'CGConv': return CGConv_Model(in_channels, h, l, d)
    elif model_name == 'GAT': return GAT(in_channels, h, l, d, hd)
    elif model_name == 'GINE': return GINE_Model(in_channels, h, l, d)
    elif model_name == 'GT': return GT_Model(in_channels, h, l, d, hd)
    else: raise ValueError(f"Modelo desconocido: {model_name}")

class EarlyStopping:
    def __init__(self, patience=20, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best = None
        self.stop = False
    
    def __call__(self, val_loss):
        if self.best is None:
            self.best = val_loss
            self.counter = 0
            return
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience: self.stop = True

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(DEVICE)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        loss = criterion(out.view(-1), data.y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    predictions, targets = [], []
    for data in loader:
        data = data.to(DEVICE)
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        loss = criterion(out.view(-1), data.y.view(-1))
        total_loss += loss.item() * data.num_graphs
        predictions.extend(out.view(-1).cpu().numpy())
        targets.extend(data.y.view(-1).cpu().numpy())
    loss_avg = total_loss / len(loader.dataset)
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)
    return loss_avg, rmse, mae, r2, predictions, targets

def plot_scatter(targets, preds, model_name, fold, save_dir, metrics):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(targets, preds, alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
    lo = min(min(targets), min(preds)) - 0.5
    hi = max(max(targets), max(preds)) + 0.5
    ax.plot([lo, hi], [lo, hi], 'r--', lw=2, label='Ideal')
    ax.set_xlabel('pIC50 Experimental', fontsize=13, fontweight='bold')
    ax.set_ylabel('pIC50 Predicho', fontsize=13, fontweight='bold')
    ax.set_title(f'{model_name} - Fold {fold}', fontsize=14, fontweight='bold')
    txt = f"R² = {metrics['r2']:.3f}\nRMSE = {metrics['rmse']:.3f}\nMAE = {metrics['mae']:.3f}"
    ax.text(0.05, 0.95, txt, transform=ax.transAxes, fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, f'scatter_fold_{fold}.png')
    plt.savefig(path, dpi=300)
    plt.close('all') 
    return path

def objective(trial, model_name):
    # Valores ajustados para CPU (Batch sizes pequeños, hidden_channels racionales)
    hidden_channels = trial.suggest_int("hidden_channels", 32, 128, step=32)
    num_layers = trial.suggest_int("num_layers", 4, 7)
    dropout = trial.suggest_float("dropout", 0.2, 0.5)
    lr = trial.suggest_float("lr", 5e-4, 5e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [4, 8, 16])
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-2, log=True)
    heads = trial.suggest_categorical("heads", [2, 4]) if model_name in ('GAT', 'GT') else 4
    
    fold_rmses = []
    for eval_fold in range(5):
        train_dataset = MProComplexDataset(root=DATASET_PATH, split='train', fold_idx=eval_fold, mode=DATASET_MODE)
        valid_dataset = MProComplexDataset(root=DATASET_PATH, split='valid', fold_idx=eval_fold, mode=DATASET_MODE)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        in_channels = train_dataset.num_features
        params = {'hidden_channels': hidden_channels, 'num_layers': num_layers, 'dropout': dropout, 'heads': heads}
        model = build_model(model_name, in_channels, params).to(DEVICE)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8)
        criterion = torch.nn.MSELoss()
        early_stopping = EarlyStopping(patience=15, min_delta=0.001)
        
        best_val_rmse = float('inf')
        for epoch in range(100):
            train_loss = train_epoch(model, train_loader, optimizer, criterion)
            val_loss, val_rmse, _, _, _, _ = evaluate(model, val_loader, criterion)
            scheduler.step(val_loss)
            if val_rmse < best_val_rmse: best_val_rmse = val_rmse
            early_stopping(val_loss)
            if early_stopping.stop: break
            trial.report(val_rmse, epoch + eval_fold * 100)
            if trial.should_prune(): raise optuna.exceptions.TrialPruned()
        fold_rmses.append(best_val_rmse)
    return np.mean(fold_rmses)

def run_cross_validation(model_name, best_params, save_dir):
    print(f"\n{'='*60}\nEvaluación 5-Fold CV: {model_name}\n{'='*60}")
    fold_results = {'rmse': [], 'mae': [], 'r2': []}
    global_best_rmse = float('inf') 
    
    for fold in range(5):
        print(f"\nFold {fold + 1}/5")
        train_dataset = MProComplexDataset(DATASET_PATH, 'train', fold, mode=DATASET_MODE)
        valid_dataset = MProComplexDataset(DATASET_PATH, 'valid', fold, mode=DATASET_MODE)
        train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
        val_loader = DataLoader(valid_dataset, batch_size=best_params['batch_size'], shuffle=False)
        
        in_channels = train_dataset.num_features
        model = build_model(model_name, in_channels, best_params).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=best_params['lr'], weight_decay=best_params.get('weight_decay', 1e-4))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        criterion = torch.nn.MSELoss()
        early_stopping = EarlyStopping(patience=20, min_delta=0.001)
        
        best_val_rmse = float('inf')
        best_model_state = None
        for epoch in range(150):
            train_loss = train_epoch(model, train_loader, optimizer, criterion)
            val_loss, val_rmse, _, _, _, _ = evaluate(model, val_loader, criterion)
            scheduler.step(val_loss)
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            early_stopping(val_loss)
            if early_stopping.stop:
                print(f"  Early stopping en epoch {epoch + 1}")
                break
                
        model.load_state_dict(best_model_state)
        _, val_rmse, val_mae, val_r2, preds, targets = evaluate(model, val_loader, criterion)
        fold_results['rmse'].append(val_rmse)
        fold_results['mae'].append(val_mae)
        fold_results['r2'].append(val_r2)
        print(f"  RMSE: {val_rmse:.4f} | MAE: {val_mae:.4f} | R²: {val_r2:.4f}")
        
        metrics = {'rmse': val_rmse, 'mae': val_mae, 'r2': val_r2}
        plot_path = plot_scatter(targets, preds, model_name, fold, save_dir, metrics)
        
        # Guarda el mejor modelo global
        if val_rmse < global_best_rmse:
            global_best_rmse = val_rmse
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model_overall.pth'))
            print(f" Nuevo mejor modelo global guardado: (Fold {fold+1} con RMSE: {val_rmse:.4f})")
            
    summary = {
        'rmse_mean': np.mean(fold_results['rmse']), 'rmse_std': np.std(fold_results['rmse']),
        'mae_mean': np.mean(fold_results['mae']), 'mae_std': np.std(fold_results['mae']),
        'r2_mean': np.mean(fold_results['r2']), 'r2_std': np.std(fold_results['r2']),
        'per_fold': fold_results
    }
    print(f"\nResultados Finales {model_name}:")
    print(f"  RMSE: {summary['rmse_mean']:.4f} ± {summary['rmse_std']:.4f}")
    print(f"  MAE:  {summary['mae_mean']:.4f} ± {summary['mae_std']:.4f}")
    print(f"  R²:   {summary['r2_mean']:.4f} ± {summary['r2_std']:.4f}")
    return summary

if __name__ == "__main__":
    os.makedirs(RESULTS_ROOT, exist_ok=True)
    
    # ---------------- CAMBIO CLAVE AQUÍ ----------------
    # Hemos sustituido NNConv por CGConv
    MODELS = ["CGConv", "GAT", "GINE", "GT"]
    all_results = {}
    print(f"\n{'='*60}\nENTRENAMIENTO CON MODELOS EDGE-AWARE\n{'='*60}\n")
    
    for model_name in MODELS:
        model_dir = os.path.join(RESULTS_ROOT, model_name)
        os.makedirs(model_dir, exist_ok=True)
        print(f"\n{'#'*60}\n# MODELO: {model_name}\n{'#'*60}")
        print(f"\nOptimizando hiperparámetros ({N_TRIALS} trials)...")
        
        study = optuna.create_study(
            study_name=f"mpro_edge_aware_{model_name}",
            storage=f"sqlite:///{model_dir}/optuna.db",
            direction="minimize",
            load_if_exists=True,
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=15)
        )
        study.optimize(lambda trial: objective(trial, model_name), n_trials=N_TRIALS, show_progress_bar=True)
        best_params = study.best_params
        print(f"\n  Mejores hiperparámetros: {best_params}\n Mejor RMSE (5 folds): {study.best_value:.4f}")
        
        with open(os.path.join(model_dir, 'best_params.json'), 'w') as f:
            json.dump(best_params, f, indent=2)
            
        print(f"\nValidación Cruzada 5-Fold...")
        cv_results = run_cross_validation(model_name, best_params, model_dir)
        all_results[model_name] = cv_results
        
        with open(os.path.join(model_dir, 'cv_results.json'), 'w') as f:
            json.dump(cv_results, f, indent=2)
            
        # =========================================================
        # LIMPIEZA MASIVA DE MEMORIA ANTES DEL SIGUIENTE MODELO
        # =========================================================
        print("Limpiando RAM y caché para el siguiente modelo...")
        del study  
        gc.collect() 
        if torch.cuda.is_available():
            torch.cuda.empty_cache() 
        # =========================================================
            
    print(f"\n{'='*60}\nRESUMEN FINAL [MODELOS EDGE-AWARE]\n{'='*60}")
    for model, results in all_results.items():
        print(f"\n{model}:")
        print(f"  RMSE: {results['rmse_mean']:.4f} ± {results['rmse_std']:.4f}")
        print(f"  MAE:  {results['mae_mean']:.4f} ± {results['mae_std']:.4f}")
        print(f"  R²:   {results['r2_mean']:.4f} ± {results['r2_std']:.4f}")
        
    with open(os.path.join(RESULTS_ROOT, 'final_summary.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResultados guardados en {RESULTS_ROOT}/\nEntrenamiento completado!")