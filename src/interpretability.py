import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx

from torch.nn import Linear, Sequential, ReLU, BatchNorm1d, Dropout
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool
from torch_geometric.loader import DataLoader
from torch_geometric.explain import Explainer, GNNExplainer, CaptumExplainer
from complex_dataset_loader import MProComplexDataset
from train_optuna import GINE_Model

from evaluation_metrics import GINEExplanationEvaluator

# =====================================================================
# WRAPPER PARA COMPATIBILIDAD CON EXPLAINER
# =====================================================================
class ModelWrapper(torch.nn.Module):
    """
    Wrapper que hace el argumento 'batch' opcional para el Explainer.
    El Explainer de PyTorch Geometric no pasa 'batch' automáticamente,
    pero el modelo GAT lo requiere. Este wrapper lo añade cuando falta.
    """
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model
    
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        return self.model(x, edge_index, edge_attr, batch)


# =====================================================================
# FUNCIÓN PARA DIBUJAR LOS MAPAS DE CALOR 2D
# =====================================================================
def plot_explanation(data, node_importance, title, save_path):
    """
    Dibuja el grafo coloreando los átomos del fármaco (Rojos) y de la
    proteína (Azules) según su importancia.
    """
    plt.figure(figsize=(12, 12))
    G = to_networkx(data, to_undirected=True)
    pos_2d = data.pos[:, :2].numpy()
    
    is_ligand = data.x[:, -1].numpy() == 1
    is_protein = ~is_ligand
    
    # Normalización independiente ligando / proteína
    imp_ligand = node_importance[is_ligand]
    if imp_ligand.max() > imp_ligand.min():
        imp_ligand = (imp_ligand - imp_ligand.min()) / (imp_ligand.max() - imp_ligand.min())
    else:
        imp_ligand = np.zeros_like(imp_ligand)
        
    imp_protein = node_importance[is_protein]
    if imp_protein.max() > imp_protein.min():
        imp_protein = (imp_protein - imp_protein.min()) / (imp_protein.max() - imp_protein.min())
    else:
        imp_protein = np.zeros_like(imp_protein)
        
    colors = []
    idx_ligand, idx_protein = 0, 0
    for lig in is_ligand:
        if lig:
            colors.append(plt.cm.Reds(imp_ligand[idx_ligand]))
            idx_ligand += 1
        else:
            colors.append(plt.cm.Blues(imp_protein[idx_protein]))
            idx_protein += 1
            
    nx.draw_networkx_edges(G, pos_2d, edge_color='lightgray', width=1.0, alpha=0.7)
    nx.draw_networkx_nodes(G, pos_2d, nodelist=np.where(is_ligand)[0].tolist(),
                           node_color=[colors[i] for i in np.where(is_ligand)[0]],
                           node_size=150, node_shape='o', edgecolors='black', linewidths=0.5, alpha=0.9)
    nx.draw_networkx_nodes(G, pos_2d, nodelist=np.where(is_protein)[0].tolist(),
                           node_color=[colors[i] for i in np.where(is_protein)[0]],
                           node_size=100, node_shape='o', alpha=0.8)

    plt.title(title, fontsize=18, fontweight='bold', pad=20)
    
    sm_rojo = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(vmin=0, vmax=1))
    cbar_rojo = plt.colorbar(sm_rojo, ax=plt.gca(), shrink=0.4, pad=0.01)
    cbar_rojo.set_label("Importancia Ligando", fontweight='bold')
    
    sm_azul = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=plt.Normalize(vmin=0, vmax=1))
    cbar_azul = plt.colorbar(sm_azul, ax=plt.gca(), shrink=0.4, pad=0.05)
    cbar_azul.set_label("Importancia Proteína", fontweight='bold')
    
    plt.axis('off')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Imagen 2D guardada en: {save_path}")


# =====================================================================
# EXPORTADOR AUTOMÁTICO A PYMOL (3D)
# =====================================================================
def save_to_pymol(data, node_importance, pdb_id, method_name, save_dir):
    is_ligand = data.x[:, -1].numpy() == 1
    
    b_factors = np.zeros(data.num_nodes)
    for mask in [is_ligand, ~is_ligand]:
        imp = node_importance[mask]
        if imp.max() > imp.min():
            norm_imp = (imp - imp.min()) / (imp.max() - imp.min())
            b_factors[mask] = norm_imp * 100.0
    
    atom_types = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'H']
    base_name = f"{method_name}_{pdb_id}"
    pdb_file = os.path.join(save_dir, f"{base_name}.pdb")
    pml_file = os.path.join(save_dir, f"{base_name}.pml")
    
    with open(pdb_file, 'w') as f:
        for i in range(data.num_nodes):
            one_hot = data.x[i, 0:10].numpy()
            idx = int(np.argmax(one_hot))
            element = atom_types[idx] if one_hot[idx] == 1 else 'C'
            x, y, z = data.pos[i].numpy()
            record = "HETATM" if is_ligand[i] else "ATOM  "
            chain = "L" if is_ligand[i] else "P"
            f.write(f"{record:<6}{i+1:>5} {element:<4} LIG {chain}{1:>4}    "
                    f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00{b_factors[i]:>6.2f}          {element:>2}\n")
        edges = data.edge_index.numpy()
        for col in range(edges.shape[1]):
            n1, n2 = edges[0, col], edges[1, col]
            if is_ligand[n1] == is_ligand[n2]:
                f.write(f"CONECT{n1+1:>5}{n2+1:>5}\n")
        f.write("END\n")
        
    with open(pml_file, 'w') as f:
        f.write(f"load {base_name}.pdb, {base_name}\n")
        f.write("hide all\n")
        f.write("bg_color white\n\n")
        f.write("set valence, 0\n")
        f.write("set stick_radius, 0.08\n\n")
        
        f.write(f"show spheres, {base_name} and chain L\n")
        f.write(f"set sphere_scale, 0.25, {base_name} and chain L\n")
        f.write(f"spectrum b, white red, {base_name} and chain L, minimum=0, maximum=100\n")
        f.write(f"show sticks, {base_name} and chain L\n\n")
        
        f.write(f"show spheres, {base_name} and chain P\n")
        f.write(f"set sphere_scale, 0.20, {base_name} and chain P\n")
        f.write(f"spectrum b, white blue, {base_name} and chain P, minimum=0, maximum=100\n")
        f.write(f"show sticks, {base_name} and chain P\n")
        f.write(f"set transparency, 0.6, {base_name} and chain P\n\n")
        
        f.write(f"center {base_name} and chain L\n")
        f.write(f"zoom {base_name} and chain L, 6\n")


# =====================================================================
# PROXY EDGE MASK A PARTIR DE IMPORTANCIAS DE NODO
# =====================================================================
def node_imp_to_edge_mask(node_importance_np, edge_index):
    """
    Para métodos como Saliency que no generan edge_mask nativa, deriva una
    proxy promediando la importancia de los dos nodos extremos de cada arista
    y normalizando a [0, 1].

    Con la normalización a [0,1] la mediana siempre está cerca de 0.5,
    lo que garantiza que el threshold adaptativo sea coherente con la escala.
    """
    node_imp = torch.tensor(node_importance_np, dtype=torch.float)
    src, dst = edge_index[0], edge_index[1]
    edge_mask = (node_imp[src] + node_imp[dst]) / 2.0
    min_val, max_val = edge_mask.min(), edge_mask.max()
    if max_val > min_val:
        edge_mask = (edge_mask - min_val) / (max_val - min_val)
    else:
        edge_mask = torch.zeros_like(edge_mask)
    return edge_mask


# =====================================================================
# THRESHOLD ADAPTATIVO (PERCENTIL)
# =====================================================================
def adaptive_threshold(edge_mask, top_k_pct=0.30):
    """
    Calcula un threshold adaptativo basado en percentil de la distribución
    de la edge_mask, de modo que aproximadamente el top_k_pct de las aristas
    queden marcadas como "importantes".

    Esto evita el problema del threshold fijo a 0.5: si la distribución de
    la máscara es asimétrica (típico en GNNExplainer y en proxies de Saliency
    normalizadas), un valor fijo puede seleccionar todas o casi ninguna arista,
    inflando artificialmente Fidelity+ y Fidelity-.

    Parámetros
    ----------
    edge_mask : torch.Tensor
        Máscara de importancia (1D o aplanable).
    top_k_pct : float, default=0.30
        Fracción de aristas a conservar como importantes (0,1].

    Devuelve
    --------
    float
        Threshold tal que aprox. top_k_pct de las aristas lo superan.
    """
    em = edge_mask.view(-1).detach().cpu().float()
    if em.numel() == 0:
        return 0.5
    top_k_pct = max(min(top_k_pct, 1.0), 1e-3)
    q = 1.0 - top_k_pct
    threshold = torch.quantile(em, q).item()
    # Si la máscara es constante (varianza nula) devolvemos un valor neutro
    if not np.isfinite(threshold):
        return 0.5
    return float(threshold)


# =====================================================================
# ESTRATIFICACIÓN POR RANGO DE pIC50
# =====================================================================
def get_pic50_stratum(pic50_value):
    """
    Asigna un rango cualitativo de potencia según el valor experimental
    de pIC50. Permite analizar si los métodos XAI funcionan de forma
    homogénea o si la varianza alta proviene de una región concreta del
    espacio químico.

    Convención farmacológica habitual:
      - pIC50 < 5     → baja potencia (IC50 > 10 µM)
      - 5 ≤ pIC50 < 7 → potencia media (10 µM ≥ IC50 > 100 nM)
      - pIC50 ≥ 7     → alta potencia (IC50 ≤ 100 nM)
    """
    if pic50_value < 5.0:
        return "low (<5)"
    if pic50_value < 7.0:
        return "medium (5-7)"
    return "high (>=7)"


# =====================================================================
# CURVAS DE FIDELIDAD AGREGADAS POR ESTRATO DE pIC50
# =====================================================================
def plot_aggregated_fidelity_curves(all_results, methods, strata, save_dir):
    """
    Agrega las curvas individuales de fidelitat (Fid+ y Fid-) almacenadas
    en `all_results[mol][method]['fidelity_curve']` y dibuja, para cada
    método, una figura con dos paneles (Fid+ y Fid-) donde cada estrato
    de pIC50 aparece como una curva con su banda de incertidumbre
    (media ± 1 desviación estándar).

    Esto permite visualizar:
      - Si la fidelidad escala con la potencia del inhibidor.
      - Cuál es el porcentaje óptimo de aristas a conservar (top-k%) para
        capturar la predicción, justificando empíricamente la elección
        del threshold (top-30%) usado en las métricas escalares.
    """
    import matplotlib.pyplot as plt

    colors_by_stratum = {
        "low (<5)":     "#1b9e77",
        "medium (5-7)": "#d95f02",
        "high (>=7)":   "#7570b3",
    }

    plots_generated = []

    for method in methods:
        # Recolectar curvas por estrato
        curves_by_stratum = {s: {'plus': [], 'minus': [], 'x': None}
                             for s in strata}

        for mol_data in all_results.values():
            if not isinstance(mol_data, dict):
                continue
            stratum = mol_data.get("pic50_stratum")
            if stratum not in curves_by_stratum:
                continue
            method_block = mol_data.get(method, {})
            curve = method_block.get("fidelity_curve") if isinstance(method_block, dict) else None
            if not curve:
                continue
            curves_by_stratum[stratum]['plus'].append(curve['fidelity_plus'])
            curves_by_stratum[stratum]['minus'].append(curve['fidelity_minus'])
            if curves_by_stratum[stratum]['x'] is None:
                curves_by_stratum[stratum]['x'] = curve['x_pct']

        # ── Generar figura con dos paneles ─────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(
            f'Curvas de Fidelidad Agregadas por Estrato de pIC50 — {method}\n'
            f'(media ± 1 sd entre moléculas; threshold top-30% marcado)',
            fontsize=12, fontweight='bold'
        )

        any_data = False
        for s in strata:
            data_s = curves_by_stratum[s]
            if not data_s['plus'] or data_s['x'] is None:
                continue
            any_data = True

            x_arr      = np.array(data_s['x'])
            plus_arr   = np.array(data_s['plus'])    # (n_mol, n_steps+1)
            minus_arr  = np.array(data_s['minus'])
            n_mol      = plus_arr.shape[0]
            color      = colors_by_stratum.get(s, "#666666")

            # ── Panel izquierdo: Fidelity+ ─────────────────────────────
            mean_p = plus_arr.mean(axis=0)
            std_p  = plus_arr.std(axis=0)
            axes[0].plot(x_arr, mean_p, marker='o', markersize=4,
                         color=color, linewidth=2,
                         label=f'{s}  (n={n_mol})')
            axes[0].fill_between(x_arr, mean_p - std_p, mean_p + std_p,
                                 alpha=0.15, color=color)

            # ── Panel derecho: Fidelity- ───────────────────────────────
            mean_m = minus_arr.mean(axis=0)
            std_m  = minus_arr.std(axis=0)
            axes[1].plot(x_arr, mean_m, marker='s', markersize=4,
                         color=color, linewidth=2,
                         label=f'{s}  (n={n_mol})')
            axes[1].fill_between(x_arr, mean_m - std_m, mean_m + std_m,
                                 alpha=0.15, color=color)

        if not any_data:
            plt.close(fig)
            print(f"  [{method}] sin datos de curvas para agregar.")
            continue

        # Línea vertical en top-30% para señalar el threshold usado
        for ax in axes:
            ax.axvline(30, color='black', linestyle=':', linewidth=1.2,
                       alpha=0.6, label='threshold escalar (30%)')
            ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, loc='best')
            ax.set_ylim(bottom=0)

        axes[0].set_title(
            'Fidelity+\nΔ pIC50 al eliminar las top-k% aristas más importantes',
            fontsize=10, fontweight='bold'
        )
        axes[0].set_xlabel('% aristas eliminadas (las más importantes primero)',
                           fontsize=10)
        axes[0].set_ylabel('|Δ pIC50|', fontsize=10)

        axes[1].set_title(
            'Fidelity−\nΔ pIC50 al conservar SOLO las top-k% aristas más importantes',
            fontsize=10, fontweight='bold'
        )
        axes[1].set_xlabel('% aristas conservadas (las más importantes primero)',
                           fontsize=10)
        axes[1].set_ylabel('|Δ pIC50|', fontsize=10)

        plt.tight_layout()
        save_path = os.path.join(save_dir, f'fidelity_curve_AGGREGATED_{method}.png')
        plt.savefig(save_path, dpi=180, bbox_inches='tight')
        plt.close(fig)
        plots_generated.append(save_path)
        print(f"  [Curva agregada] {method} → {save_path}")

    return plots_generated


# =====================================================================
# MAIN
# =====================================================================
if __name__ == "__main__":
    MODEL_DIR    = "results/GINE"
    DATASET_PATH = "./data"
    DEVICE       = torch.device('cpu')

    os.makedirs(MODEL_DIR, exist_ok=True)
    
    print("\n" + "="*60)
    print("ANÁLISIS DE INTERPRETABILIDAD (XAI) Y MÉTRICAS")
    print("Modelo: GINE (Graph Isomorphism Network with Edge features)")
    print("="*60)
    
    with open(os.path.join(MODEL_DIR, 'best_params.json'), 'r') as f:
        best_params = json.load(f)
    
    print(f"\nHiperparámetros:")
    print(f"  • Hidden:  {best_params['hidden_channels']}")
    print(f"  • Layers:  {best_params['num_layers']}")
    print(f"  • Dropout: {best_params['dropout']}")
    
        
    print(f"\nCargando dataset...")
    dataset = MProComplexDataset(root=DATASET_PATH, split='valid', fold_idx=4, mode='complex')
    print(f"  ✓ {len(dataset)} complejos")
    
    print(f"\nConstruyendo modelo GINE...")
    model_original = GINE_Model(
        in_channels=15,
        hidden_channels=best_params['hidden_channels'],
        num_layers=best_params['num_layers'],
        dropout=best_params['dropout']
    ).to(DEVICE)
    
    model_path = os.path.join(MODEL_DIR, 'best_model_overall.pth')
    
    print(f"  Cargando: {model_path}")
    model_original.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model_original.eval()
    
    print(f"\nInicializando Evaluador de Métricas...")
    evaluator = GINEExplanationEvaluator(
        model_path=model_path,
        params_path=os.path.join(MODEL_DIR, 'best_params.json'),
        device=DEVICE
    )
    
    model = ModelWrapper(model_original)
    model.eval()
    print(f"  ✓ Modelo listo (con wrapper para Explainer)")

    # ─────────────────────────────────────────────────────────────────
    # Top-30 predicciones con menor error absoluto
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("Buscando mejores predicciones...")
    print("="*60)
    
    resultados = []
    for idx in range(len(dataset)):
        data = dataset[idx].to(DEVICE)
        batch = torch.zeros(data.x.size(0), dtype=torch.long).to(DEVICE)
        pred = model_original(data.x, data.edge_index, data.edge_attr, batch).item()
        real = data.y.item()
        resultados.append({
            'indice':  idx,
            'error':   abs(pred - real),
            'pred':    pred,
            'real':    real,
            'stratum': get_pic50_stratum(real),
        })

    # ── Muestreo estratificado: K mejores predicciones por estrato ────────
    K_PER_STRATUM = 10
    strata_order  = ["low (<5)", "medium (5-7)", "high (>=7)"]

    print(f"\nDistribución de moléculas por estrato en el dataset:")
    for s in strata_order:
        n_total = sum(1 for r in resultados if r['stratum'] == s)
        print(f"  • {s:<14}  n={n_total}")

    print(f"\nSeleccionando hasta {K_PER_STRATUM} mejores predicciones por estrato...")
    top_estratificado = []
    for s in strata_order:
        candidatos = [r for r in resultados if r['stratum'] == s]
        seleccionados = sorted(candidatos, key=lambda x: x['error'])[:K_PER_STRATUM]
        print(f"  • {s:<14}  seleccionados={len(seleccionados)}")
        top_estratificado.extend(seleccionados)

    print(f"\nTop por estrato (total={len(top_estratificado)}):")
    print(f"{'#':>3} {'Idx':>5} {'Stratum':>14} {'Pred':>8} {'Real':>8} {'Error':>8}")
    print("-" * 60)
    for rank, res in enumerate(top_estratificado, 1):
        print(f"{rank:>3} {res['indice']:>5} {res['stratum']:>14} "
              f"{res['pred']:>8.3f} {res['real']:>8.3f} {res['error']:>8.3f}")

    indices_a_evaluar = [res['indice'] for res in top_estratificado]
    n_total_eval      = len(indices_a_evaluar)
    all_results = {}

    print("\n" + "="*60)
    print("Generando visualizaciones y calculando métricas...")
    print("="*60)
    
    for rank, idx in enumerate(indices_a_evaluar, 1):
        data = dataset[idx].to(DEVICE)
        
        if hasattr(data, 'pdb_id'):
            nombre = str(data.pdb_id)
            if isinstance(data.pdb_id, list):
                nombre = data.pdb_id[0]
        else:
            nombre = f"Mol_{idx}"
        
        batch = torch.zeros(data.x.size(0), dtype=torch.long).to(DEVICE)
        pred  = model_original(data.x, data.edge_index, data.edge_attr, batch).item()
        real  = data.y.item()
        error = abs(pred - real)

        print(f"\n[{rank}/{n_total_eval}] {nombre}")
        print(f"  Pred={pred:.3f}  Real={real:.3f}  Error={error:.3f}")

        all_results[nombre] = {
            "pred":  round(pred,  4),
            "real":  round(real,  4),
            "error": round(error, 4),
            "pic50_stratum": get_pic50_stratum(real),
        }

        # ── GNNExplainer ─────────────────────────────────────────────
        print(f"  → GNNExplainer...")
        try:
            explainer_gnn = Explainer(
                model=model,
                algorithm=GNNExplainer(epochs=200),
                explanation_type='model',
                node_mask_type='attributes',
                edge_mask_type='object',
                model_config=dict(mode='regression', task_level='graph', return_type='raw')
            )
            exp_gnn = explainer_gnn(data.x, data.edge_index, edge_attr=data.edge_attr)

            node_imp_gnn = exp_gnn.node_mask.sum(dim=1).detach().numpy()

            if hasattr(exp_gnn, 'edge_mask') and exp_gnn.edge_mask is not None:
                edge_mask_gnn = exp_gnn.edge_mask
            else:
                edge_mask_gnn = node_imp_to_edge_mask(node_imp_gnn, data.edge_index)

            def gnn_wrapper_stability(x_pert, edge_index_pert, edge_attr_pert):
                exp = explainer_gnn(x_pert, edge_index_pert, edge_attr=edge_attr_pert)
                node_imp = exp.node_mask.sum(dim=1).detach().numpy()
                if hasattr(exp, 'edge_mask') and exp.edge_mask is not None:
                    return exp.edge_mask
                return node_imp_to_edge_mask(node_imp, edge_index_pert)

            # Threshold adaptativo: top 30% de aristas como "importantes"
            threshold_gnn = adaptive_threshold(edge_mask_gnn, top_k_pct=0.30)
            print(f"    Threshold adaptativo (top-30%): {threshold_gnn:.4f}")

            metricas_gnn = evaluator.evaluate_explanation(
                data=data,
                edge_mask=edge_mask_gnn,
                explainer_func=gnn_wrapper_stability,
                threshold=threshold_gnn
            )
            metricas_gnn["threshold_used"] = round(threshold_gnn, 4)
            all_results[nombre]["GNNExplainer"] = metricas_gnn

            print("    Métricas:")
            for k, v in metricas_gnn.items():
                print(f"      - {k}: {v}")

            plot_explanation(data, node_imp_gnn, f"GNNExplainer - {nombre}",
                             os.path.join(MODEL_DIR, f"gnnexplainer_{nombre}.png"))
            save_to_pymol(data, node_imp_gnn, nombre, "gnnexplainer", MODEL_DIR)

            # ── Curva de fidelitat progressiva ────────────────────────
            curve_gnn = evaluator.get_fidelity_curve(
                data=data,
                edge_mask=edge_mask_gnn,
                pdb_id=nombre,
                method_name="GNNExplainer",
                save_dir=MODEL_DIR,
                n_steps=20
            )
            all_results[nombre]["GNNExplainer"]["fidelity_curve"] = curve_gnn

            print(f"    ✓ Completado")

        except Exception as e:
            import traceback
            print(f"    ✗ Error: {e}")
            traceback.print_exc()
            all_results[nombre]["GNNExplainer"] = {"error": str(e)}

        # ── Saliency ─────────────────────────────────────────────────
        print(f"  → Saliency...")
        try:
            explainer_sal = Explainer(
                model=model,
                algorithm=CaptumExplainer('Saliency'),
                explanation_type='model',
                node_mask_type='attributes',
                edge_mask_type=None,   
                model_config=dict(mode='regression', task_level='graph', return_type='raw')
            )
            exp_sal = explainer_sal(data.x, data.edge_index, edge_attr=data.edge_attr)

            
            node_imp_sal = exp_sal.node_mask.abs().sum(dim=1).detach().numpy()

            proxy_edge_mask = node_imp_to_edge_mask(node_imp_sal, data.edge_index)
            # Threshold adaptativo: top 30% de aristas como "importantes"
            threshold_sal = adaptive_threshold(proxy_edge_mask, top_k_pct=0.30)
            print(f"    Threshold adaptativo (top-30%): {threshold_sal:.4f}")

            def sal_wrapper_stability(x_pert, edge_index_pert, edge_attr_pert):
                exp = explainer_sal(x_pert, edge_index_pert, edge_attr=edge_attr_pert)
                node_imp = exp.node_mask.abs().sum(dim=1).detach().numpy()
                return node_imp_to_edge_mask(node_imp, edge_index_pert)

            metricas_sal = evaluator.evaluate_explanation(
                data=data,
                edge_mask=proxy_edge_mask,      
                explainer_func=sal_wrapper_stability,
                threshold=threshold_sal         
            )
            metricas_sal["threshold_used"] = round(threshold_sal, 4)
            metricas_sal["nota"] = (
                "Importancia de nodo = |gradiente| sumado por features. "
                "Fidelity/Sparsity calculadas sobre edge_mask proxy "
                "(media de importancias de nodos extremos, normalizada [0,1]). "
                "Threshold adaptativo por percentil (top 30%)."
            )

            all_results[nombre]["Saliency"] = metricas_sal

            print("    Métricas:")
            for k, v in metricas_sal.items():
                print(f"      - {k}: {v}")

            plot_explanation(data, node_imp_sal, f"Saliency - {nombre}",
                             os.path.join(MODEL_DIR, f"saliency_{nombre}.png"))
            save_to_pymol(data, node_imp_sal, nombre, "saliency", MODEL_DIR)

            # ── Curva de fidelitat progressiva ────────────────────────
            curve_sal = evaluator.get_fidelity_curve(
                data=data,
                edge_mask=proxy_edge_mask,
                pdb_id=nombre,
                method_name="Saliency",
                save_dir=MODEL_DIR,
                n_steps=20
            )
            all_results[nombre]["Saliency"]["fidelity_curve"] = curve_sal

            print(f"    ✓ Completado")

        except Exception as e:
            import traceback
            print(f"    ✗ Error: {e}")
            traceback.print_exc()
            all_results[nombre]["Saliency"] = {"error": str(e)}

    # ─────────────────────────────────────────────────────────────────
    # Guardar JSON con todas las métricas
    # ─────────────────────────────────────────────────────────────────
    metrics_path = os.path.join(MODEL_DIR, "xai_metrics.json")
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)
    print(f"\n  ✓ Métricas guardadas en: {metrics_path}")

    # ─────────────────────────────────────────────────────────────────
    # Resumen agregado por método (global + estratificado por pIC50)
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("RESUMEN AGREGADO DE MÉTRICAS")
    print("="*60)

    metric_keys = ["Fidelity+", "Fidelity-", "Sparsity", "Stability"]
    methods     = ["GNNExplainer", "Saliency"]
    strata      = ["low (<5)", "medium (5-7)", "high (>=7)"]

    summary_block = {}

    # ── Resumen GLOBAL ────────────────────────────────────────────────
    print("\n[GLOBAL]")
    for method in methods:
        vals = {k: [] for k in metric_keys}
        for mol_data in all_results.values():
            m = mol_data.get(method, {})
            if "error" in m:
                continue
            for key in metric_keys:
                v = m.get(key)
                if isinstance(v, (int, float)):
                    vals[key].append(v)

        print(f"\n  {method}:")
        method_summary = {}
        for key, lst in vals.items():
            if lst:
                mean_v, std_v = float(np.mean(lst)), float(np.std(lst))
                print(f"    {key:<25} media={mean_v:.4f}  std={std_v:.4f}  (n={len(lst)})")
                method_summary[key] = {"mean": round(mean_v, 4),
                                       "std":  round(std_v,  4),
                                       "n":    len(lst)}
            else:
                print(f"    {key:<25} N/A")
                method_summary[key] = None
        summary_block[method] = {"global": method_summary, "by_stratum": {}}

    # ── Resumen ESTRATIFICADO por pIC50 ───────────────────────────────
    print("\n[ESTRATIFICADO POR pIC50]")
    print("  Convención: low<5, medium∈[5,7), high≥7")

    for method in methods:
        print(f"\n  {method}:")
        for stratum in strata:
            vals = {k: [] for k in metric_keys}
            for mol_data in all_results.values():
                if mol_data.get("pic50_stratum") != stratum:
                    continue
                m = mol_data.get(method, {})
                if "error" in m:
                    continue
                for key in metric_keys:
                    v = m.get(key)
                    if isinstance(v, (int, float)):
                        vals[key].append(v)

            n_mol = max(len(v) for v in vals.values()) if vals else 0
            if n_mol == 0:
                print(f"    [{stratum}]  (sin moléculas)")
                summary_block[method]["by_stratum"][stratum] = None
                continue

            print(f"    [{stratum}]  n={n_mol}")
            stratum_summary = {}
            for key, lst in vals.items():
                if lst:
                    mean_v, std_v = float(np.mean(lst)), float(np.std(lst))
                    print(f"      {key:<23} media={mean_v:.4f}  std={std_v:.4f}")
                    stratum_summary[key] = {"mean": round(mean_v, 4),
                                            "std":  round(std_v,  4),
                                            "n":    len(lst)}
                else:
                    print(f"      {key:<23} N/A")
                    stratum_summary[key] = None
            summary_block[method]["by_stratum"][stratum] = stratum_summary

    # ── Persistir el resumen junto con las métricas individuales ──────
    all_results["__summary__"] = summary_block
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)
    print(f"\n  ✓ Resumen estratificado añadido a: {metrics_path}")

    # ─────────────────────────────────────────────────────────────────
    # Curvas de fidelidad agregadas por estrato de pIC50
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("Generando curvas de fidelidad agregadas por estrato...")
    print("="*60)
    aggregated_paths = plot_aggregated_fidelity_curves(
        all_results=all_results,
        methods=methods,
        strata=strata,
        save_dir=MODEL_DIR,
    )

    print("\n" + "="*60)
    print("ANÁLISIS COMPLETADO")
    print("="*60)
    print(f"\nResultados en: {MODEL_DIR}/")
    print(f"  • Métricas XAI:           xai_metrics.json")
    print(f"  • Imágenes 2D:            gnnexplainer_*.png, saliency_*.png")
    print(f"  • Archivos PyMOL:         gnnexplainer_*.pdb/.pml, saliency_*.pdb/.pml")
    print(f"  • Curvas individuales:    fidelity_curve_<method>_<pdb>.png")
    print(f"  • Curvas agregadas:       fidelity_curve_AGGREGATED_<method>.png")
    print(f"\nMoléculas evaluadas: {n_total_eval}")