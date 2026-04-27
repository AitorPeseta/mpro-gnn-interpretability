import os
import json
import torch
import numpy as np
from sklearn.metrics import roc_auc_score

from train_optuna import GINE_Model

class GINEExplanationEvaluator:
    def __init__(self, model_path='results/GINE/best_model_overall.pth', 
                 params_path='results/GINE/best_params.json', 
                 device='cpu'):
        """Carga el modelo GINE y los parámetros para evaluar explicaciones."""
        self.device = device
        
        with open(params_path, 'r') as f:
            self.params = json.load(f)
        
        # GINE no usa heads — solo hidden_channels, num_layers, dropout
        self.model = GINE_Model(
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

            # Fidelity+: eliminar aristas importantes, ver cuánto cae la predicción
            ei_plus = edge_index[:, ~important_mask]
            ea_plus = edge_attr[~important_mask] if edge_attr is not None else None
            if ei_plus.size(1) == 0:
                ei_plus = torch.empty((2, 0), dtype=torch.long, device=self.device)
                ea_plus = torch.empty((0, 13), dtype=torch.float, device=self.device)
            y_plus = self.model(x, ei_plus, ea_plus, batch).item()
            fid_plus = y_orig - y_plus

            # Fidelity-: quedarse solo con aristas importantes
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
            if np.std(m_orig_np) == 0 or np.std(m_pert_np) == 0:
                return 0.0
            stability = np.corrcoef(m_orig_np, m_pert_np)[0, 1]
            if np.isnan(stability):
                return 0.0
            return float(stability)
        except Exception:
            return None

    def get_fidelity_curve(self, data, edge_mask, pdb_id, method_name, save_dir,
                           n_steps=20):
        """
        Curva de fidelitat progressiva.

        Ordena les arestes per importància (de major a menor) i les elimina
        progressivament en n_steps passos. En cada pas registra:
          - Fidelity+ acumulada: predicció sense les X% arestes més importants
          - Fidelity- acumulada: predicció amb NOMÉS les X% arestes més importants

        Un bon explicador hauria de mostrar:
          - Fidelity+ creixent: com més arestes importants eliminem, més cau la predicció
          - Fidelity- decreixent: com menys arestes importants conservem, menys captura

        Paràmetres
        ----------
        n_steps : int
            Nombre de punts a la corba (ex. 20 = avalua cada 5% d'arestes).

        Retorna
        -------
        dict amb les llistes de valors per a cada corba.
        """
        x          = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)
        edge_attr  = data.edge_attr.to(self.device) if data.edge_attr is not None else None
        batch      = torch.zeros(x.size(0), dtype=torch.long, device=self.device)
        edge_mask_t = edge_mask.view(-1).to(self.device)

        n_edges = edge_mask_t.size(0)

        with torch.no_grad():
            y_orig = self.model(x, edge_index, edge_attr, batch).item()

        # Ordenar arestes de major a menor importància
        sorted_idx = torch.argsort(edge_mask_t, descending=True)

        # Percentatges que avaluarem
        percentages = np.linspace(0, 1, n_steps + 1)   # 0%, 5%, ..., 100%

        fid_plus_curve  = []   # predicció eliminant les top-k% més importants
        fid_minus_curve = []   # predicció conservant NOMÉS les top-k% més importants
        x_labels        = []

        with torch.no_grad():
            for pct in percentages:
                k = int(pct * n_edges)
                x_labels.append(round(pct * 100, 1))

                # Índexs de les k arestes més importants
                top_k_idx = sorted_idx[:k] if k > 0 else torch.tensor([], dtype=torch.long)

                # Màscara booleana
                important_mask = torch.zeros(n_edges, dtype=torch.bool, device=self.device)
                if k > 0:
                    important_mask[top_k_idx] = True

                # Fidelity+: graf SENSE les arestes importants (eliminem les millors)
                ei_plus = edge_index[:, ~important_mask]
                ea_plus = edge_attr[~important_mask] if edge_attr is not None else None
                if ei_plus.size(1) == 0:
                    ei_plus = torch.empty((2, 0), dtype=torch.long, device=self.device)
                    ea_plus = torch.empty((0, 13), dtype=torch.float, device=self.device)
                y_plus = self.model(x, ei_plus, ea_plus, batch).item()
                fid_plus_curve.append(round(abs(y_orig - y_plus), 4))

                # Fidelity-: graf AMB NOMÉS les arestes importants
                ei_minus = edge_index[:, important_mask]
                ea_minus = edge_attr[important_mask] if edge_attr is not None else None
                if ei_minus.size(1) == 0:
                    ei_minus = torch.empty((2, 0), dtype=torch.long, device=self.device)
                    ea_minus = torch.empty((0, 13), dtype=torch.float, device=self.device)
                y_minus = self.model(x, ei_minus, ea_minus, batch).item()
                fid_minus_curve.append(round(abs(y_orig - y_minus), 4))

        # ── Gràfic ───────────────────────────────────────────────────────────
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(
            f'Corba de Fidelitat Progressiva — {method_name} | {pdb_id}\n'
            f'pIC50 original: {y_orig:.3f}',
            fontsize=13, fontweight='bold'
        )

        # ── Subplot esquerra: Fidelity+ ──────────────────────────────────────
        ax = axes[0]
        ax.plot(x_labels, fid_plus_curve, marker='o', markersize=4,
                color='#d94801', linewidth=2, label='Fidelity+')
        ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
        ax.fill_between(x_labels, fid_plus_curve, 0,
                        alpha=0.15, color='#d94801', label='Impacte acumulat')
        ax.set_xlabel('% arestes eliminades (les més importants primer)', fontsize=10)
        ax.set_ylabel('|Δ pIC50|  (valor absolut)', fontsize=10)
        ax.set_title('Fidelity+\nQuant cau la predicció en eliminar les arestes importants?',
                     fontsize=10, fontweight='bold')
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # ── Subplot dreta: Fidelity- ─────────────────────────────────────────
        ax = axes[1]
        ax.plot(x_labels, fid_minus_curve, marker='s', markersize=4,
                color='#2171b5', linewidth=2, label='Fidelity-')
        ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
        ax.fill_between(x_labels, fid_minus_curve, 0,
                        alpha=0.15, color='#2171b5', label='Divergència acumulada')
        ax.set_xlabel('% arestes conservades (les més importants primer)', fontsize=10)
        ax.set_ylabel('|Δ pIC50|  (valor absolut)', fontsize=10)
        ax.set_title('Fidelity−\nQuant captura el subgraf important la predicció original?',
                     fontsize=10, fontweight='bold')
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(
            save_dir, f'fidelity_curve_{method_name}_{pdb_id}.png'
        )
        plt.savefig(save_path, dpi=180, bbox_inches='tight')
        plt.close(fig)
        print(f"    [Corba] Guardada: {save_path}")

        return {
            'x_pct':           x_labels,
            'fidelity_plus':   fid_plus_curve,
            'fidelity_minus':  fid_minus_curve,
            'y_orig':          round(y_orig, 4),
            'n_edges':         n_edges,
        }

    def evaluate_explanation(self, data, edge_mask, explainer_func=None, gt_mask=None, threshold=0.5):
        f_plus, f_minus = self.get_fidelity(data, edge_mask, threshold)
        sparsity  = self.get_sparsity(edge_mask, threshold)
        accuracy  = self.get_accuracy(edge_mask, gt_mask)
        stability = self.get_stability(explainer_func, data)

        return {
            "Fidelity+": round(f_plus,  4),   # Δ pIC50 al eliminar aristas importantes
            "Fidelity-": round(f_minus, 4),   # Δ pIC50 al conservar solo aristas importantes
            "Sparsity":  round(sparsity, 4),
            "Accuracy":  round(accuracy, 4) if accuracy  is not None else "N/A",
            "Stability": round(stability, 4) if stability is not None else "N/A"
        }