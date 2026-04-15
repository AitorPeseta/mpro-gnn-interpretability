import os
import json
import torch
import numpy as np
from sklearn.metrics import roc_auc_score

from train_optuna import GAT

class GATExplanationEvaluator:
    def __init__(self, model_path='results/GAT/best_model_overall.pth', 
                 params_path='results/GAT/best_params.json', 
                 device='cpu'):
        """Carga el modelo y los parámetros para evaluar explicaciones."""
        self.device = device
        
        with open(params_path, 'r') as f:
            self.params = json.load(f)
        
        # Reconstruir Modelo (Asegurando in_channels=15)
        self.model = GAT(
            in_channels=15,
            hidden_channels=self.params['hidden_channels'],
            num_layers=self.params['num_layers'],
            dropout=self.params['dropout']
        ).to(self.device)
        
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def get_sparsity(self, edge_mask, threshold=0.5):
        edge_mask = edge_mask.view(-1)
        if edge_mask.size(0) == 0: return 0.0
        return 1.0 - ((edge_mask >= threshold).sum().item() / edge_mask.size(0))

    def get_fidelity(self, data, edge_mask, threshold=0.5):
        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)
        edge_attr = data.edge_attr.to(self.device) if data.edge_attr is not None else None
        batch = torch.zeros(x.size(0), dtype=torch.long, device=self.device)

        edge_mask = edge_mask.view(-1).to(self.device)
        important_mask = (edge_mask >= threshold)

        with torch.no_grad():
            y_orig = self.model(x, edge_index, edge_attr, batch).item()

            # Fidelity+
            ei_plus = edge_index[:, ~important_mask]
            ea_plus = edge_attr[~important_mask] if edge_attr is not None else None
            
            if ei_plus.size(1) == 0:
                ei_plus = torch.empty((2, 0), dtype=torch.long, device=self.device)
                ea_plus = torch.empty((0, 13), dtype=torch.float, device=self.device)
                
            y_plus = self.model(x, ei_plus, ea_plus, batch).item()
            fid_plus = y_orig - y_plus

            # Fidelity-
            ei_minus = edge_index[:, important_mask]
            ea_minus = edge_attr[important_mask] if edge_attr is not None else None
            
            if ei_minus.size(1) == 0:
                ei_minus = torch.empty((2, 0), dtype=torch.long, device=self.device)
                ea_minus = torch.empty((0, 13), dtype=torch.float, device=self.device)

            y_minus = self.model(x, ei_minus, ea_minus, batch).item()
            fid_minus = y_orig - y_minus

        return fid_plus, fid_minus

    def get_accuracy(self, edge_mask, ground_truth_mask):
        if ground_truth_mask is None:
            return None
        edge_mask_np = edge_mask.cpu().view(-1).numpy()
        gt_mask_np = ground_truth_mask.cpu().view(-1).numpy()
        
        if len(np.unique(gt_mask_np)) <= 1:
            return None
        try:
            return roc_auc_score(gt_mask_np, edge_mask_np)
        except ValueError:
            return None

    def get_stability(self, explainer_func, data, perturbation_std=0.01):
        if explainer_func is None:
            return None
        try:
            mask_orig = explainer_func(data.x, data.edge_index, data.edge_attr)
            
            data_perturbed = data.clone()
            data_perturbed.x += torch.randn_like(data.x) * perturbation_std
            
            mask_pert = explainer_func(data_perturbed.x, data_perturbed.edge_index, data_perturbed.edge_attr)

            m_orig_np = mask_orig.cpu().view(-1).numpy()
            m_pert_np = mask_pert.cpu().view(-1).numpy()
            
            if np.std(m_orig_np) == 0 or np.std(m_pert_np) == 0: return 0.0
            stability = np.corrcoef(m_orig_np, m_pert_np)[0, 1]
            if np.isnan(stability): return 0.0
            
            return float(stability)
        except Exception as e:
            return None

    def evaluate_explanation(self, data, edge_mask, explainer_func=None, gt_mask=None, threshold=0.5):
        f_plus, f_minus = self.get_fidelity(data, edge_mask, threshold)
        sparsity = self.get_sparsity(edge_mask, threshold)
        accuracy = self.get_accuracy(edge_mask, gt_mask)
        stability = self.get_stability(explainer_func, data)

        return {
            "Fidelity+ (prob)": round(f_plus, 4),
            "Fidelity- (prob)": round(f_minus, 4),
            "Sparsity": round(sparsity, 4),
            "Accuracy": round(accuracy, 4) if accuracy is not None else "N/A",
            "Stability": round(stability, 4) if stability is not None else "N/A"
        }