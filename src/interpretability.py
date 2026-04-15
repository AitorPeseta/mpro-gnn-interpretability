import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx

from torch.nn import Linear, Sequential, ReLU, BatchNorm1d, Dropout
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.loader import DataLoader
from torch_geometric.explain import Explainer, GNNExplainer, CaptumExplainer
from complex_dataset_loader import MProComplexDataset
from train_optuna import GAT

# Importamos el evaluador de métricas que arreglamos antes
from evaluation_metrics import GATExplanationEvaluator

# =====================================================================
# WRAPPER PARA COMPATIBILIDAD CON EXPLAINER
# =====================================================================
class ModelWrapper(torch.nn.Module):
    """
    Wrapper que hace el argumento 'batch' opcional para el Explainer.
    
    El Explainer de PyTorch Geometric no pasa 'batch' automáticamente,
    pero tu modelo GAT lo requiere. Este wrapper lo añade cuando falta.
    """
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model
    
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        # Si batch no se proporciona, crearlo
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Llamar al modelo original con batch
        return self.model(x, edge_index, edge_attr, batch)

# =====================================================================
# FUNCIÓN PARA DIBUJAR LOS MAPAS DE CALOR 2D
# =====================================================================
def plot_explanation(data, node_importance, title, save_path):
    """
    Dibuja el grafo coloreando los átomos del fármaco (Rojos) y de la proteína (Azules)
    según su importancia.
    """
    plt.figure(figsize=(12, 12))
    G = to_networkx(data, to_undirected=True)
    pos_2d = data.pos[:, :2].numpy()
    
    is_ligand = data.x[:, -1].numpy() == 1
    is_protein = ~is_ligand
    
    # Normalización independiente
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
        
    # Asignar colores
    colors = []
    idx_ligand, idx_protein = 0, 0
    
    for lig in is_ligand:
        if lig:
            colors.append(plt.cm.Reds(imp_ligand[idx_ligand]))
            idx_ligand += 1
        else:
            colors.append(plt.cm.Blues(imp_protein[idx_protein]))
            idx_protein += 1
            
    # Dibujar grafo
    nx.draw_networkx_edges(G, pos_2d, edge_color='lightgray', width=1.0, alpha=0.7)
    
    nx.draw_networkx_nodes(G, pos_2d, nodelist=np.where(is_ligand)[0].tolist(), 
                           node_color=[colors[i] for i in np.where(is_ligand)[0]],
                           node_size=150, node_shape='o', edgecolors='black', linewidths=0.5, alpha=0.9)
    
    nx.draw_networkx_nodes(G, pos_2d, nodelist=np.where(is_protein)[0].tolist(), 
                           node_color=[colors[i] for i in np.where(is_protein)[0]],
                           node_size=100, node_shape='o', alpha=0.8)

    plt.title(title, fontsize=18, fontweight='bold', pad=20)
    
    # Leyendas
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
    
    # Normalización escala 0-100
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
            f.write(f"{record:<6}{i+1:>5} {element:<4} LIG {chain}{1:>4}    {x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00{b_factors[i]:>6.2f}          {element:>2}\n")
            
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
        f.write(f"unbond (chain L), (chain P)\n")
        f.write(f"h_add {base_name}\n")
        f.write(f"connect {base_name}\n")
        f.write(f"set stick_color, black, {base_name}\n")
        f.write(f"set line_color, black, {base_name}\n")
        f.write("set stick_radius, 0.08\n")
        f.write("set line_width, 1.5\n")
        f.write("set valence, 0\n\n")
        f.write(f"show spheres, {base_name} and chain L\n")
        f.write(f"set sphere_scale, 0.25, {base_name} and chain L\n")
        f.write(f"spectrum b, white red, {base_name} and chain L, minimum=0, maximum=100\n")
        f.write(f"show sticks, {base_name} and chain L\n\n")
        f.write(f"show spheres, {base_name} and chain P\n")
        f.write(f"set sphere_scale, 0.20, {base_name} and chain P\n")
        f.write(f"spectrum b, white blue, {base_name} and chain P, minimum=0, maximum=100\n")
        f.write("set transparency, 0.6, {base_name} and chain P\n")
        f.write(f"show lines, {base_name} and chain P\n\n")
        f.write(f"center {base_name} and chain L\n")
        f.write(f"zoom {base_name} and chain L, 6\n")


# =====================================================================
# UTILIDAD: PROXY DE EDGE_MASK A PARTIR DE NODE IMPORTANCE
# =====================================================================
def node_imp_to_edge_mask(node_importance_np, edge_index):
    """
    Para métodos como Saliency que no generan edge_mask nativa,
    deriva una proxy promediando la importancia de los dos nodos
    extremos de cada arista y normalizando a [0, 1].

    Esto permite calcular Fidelity y Sparsity con el mismo evaluador.
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
# MAIN
# =====================================================================
if __name__ == "__main__":
    MODEL_DIR = "results/GAT"
    DATASET_PATH = "./data"
    DEVICE = torch.device('cpu')

    # Crear directorio si no existe
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    print("\n" + "="*60)
    print("ANÁLISIS DE INTERPRETABILIDAD (XAI) Y MÉTRICAS")
    print("Modelo: GAT (Graph Attention Network)")
    print("="*60)
    
    # Cargar hiperparámetros
    with open(os.path.join(MODEL_DIR, 'best_params.json'), 'r') as f:
        best_params = json.load(f)
    
    print(f"\nHiperparámetros:")
    print(f"  • Hidden: {best_params['hidden_channels']}")
    print(f"  • Layers: {best_params['num_layers']}")
    print(f"  • Dropout: {best_params['dropout']}")
        
    # Cargar dataset
    print(f"\nCargando dataset...")
    dataset = MProComplexDataset(root=DATASET_PATH, split='valid', fold_idx=0, mode='complex')
    print(f"  ✓ {len(dataset)} complejos")
    
    # Construir modelo GAT original
    print(f"\nConstruyendo modelo GAT...")
    model_original = GAT( 
        in_channels=15,
        hidden_channels=best_params['hidden_channels'],
        num_layers=best_params['num_layers'],
        dropout=best_params['dropout']    
    ).to(DEVICE)
    
    # Cargar pesos
    model_path = os.path.join(MODEL_DIR, 'best_model_overall.pth')
    if not os.path.exists(model_path):
        model_path = os.path.join(MODEL_DIR, 'best_model_fold0.pth')
    
    print(f"  Cargando: {model_path}")
    model_original.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model_original.eval()
    
    # Inicializar el Evaluador de Métricas
    print(f"\nInicializando Evaluador de Métricas...")
    evaluator = GATExplanationEvaluator(
        model_path=model_path,
        params_path=os.path.join(MODEL_DIR, 'best_params.json'),
        device=DEVICE
    )
    
    # Envolver el modelo para el Explainer
    model = ModelWrapper(model_original)
    model.eval()
    print(f"  ✓ Modelo listo (con wrapper para Explainer)")

    # ─────────────────────────────────────────────────────────────────
    # Encontrar top-10 predicciones con menor error
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
        error = abs(pred - real)
        resultados.append({'indice': idx, 'error': error, 'pred': pred, 'real': real})
        
    resultados_ordenados = sorted(resultados, key=lambda x: x['error'])
    top_10 = resultados_ordenados[:10]
    
    print(f"\nTop 10 predicciones:")
    print(f"{'#':>3} {'Idx':>5} {'Pred':>8} {'Real':>8} {'Error':>8}")
    print("-" * 40)
    for rank, res in enumerate(top_10, 1):
        print(f"{rank:>3} {res['indice']:>5} {res['pred']:>8.3f} {res['real']:>8.3f} {res['error']:>8.3f}")
        
    indices_a_evaluar = [res['indice'] for res in top_10]
    
    # ─────────────────────────────────────────────────────────────────
    # Diccionario global para acumular todas las métricas
    # Estructura: { pdb_id: { "pred": ..., "real": ..., "error": ...,
    #                          "GNNExplainer": {...métricas...},
    #                          "Saliency":     {...métricas...} } }
    # ─────────────────────────────────────────────────────────────────
    all_results = {}

    print("\n" + "="*60)
    print("Generando visualizaciones y calculando métricas...")
    print("="*60)
    
    for rank, idx in enumerate(indices_a_evaluar, 1):
        data = dataset[idx].to(DEVICE)
        
        # Obtener nombre
        if hasattr(data, 'pdb_id'):
            nombre = str(data.pdb_id)
            if isinstance(data.pdb_id, list): 
                nombre = data.pdb_id[0]
        else:
            nombre = f"Mol_{idx}"
        
        batch = torch.zeros(data.x.size(0), dtype=torch.long).to(DEVICE)
        pred = model_original(data.x, data.edge_index, data.edge_attr, batch).item()
        real = data.y.item()
        error = abs(pred - real)

        print(f"\n[{rank}/10] {nombre}")
        print(f"  Pred={pred:.3f}  Real={real:.3f}  Error={error:.3f}")

        # Entrada base en el diccionario de resultados
        all_results[nombre] = {
            "pred":  round(pred, 4),
            "real":  round(real, 4),
            "error": round(error, 4),
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
            
            def gnn_wrapper_stability(x_pert, edge_index_pert, edge_attr_pert):
                exp = explainer_gnn(x_pert, edge_index_pert, edge_attr=edge_attr_pert)
                return exp.edge_mask

            metricas_gnn = evaluator.evaluate_explanation(
                data=data,
                edge_mask=exp_gnn.edge_mask,
                explainer_func=gnn_wrapper_stability,
                threshold=0.5
            )

            all_results[nombre]["GNNExplainer"] = metricas_gnn

            print("    Métricas:")
            for k, v in metricas_gnn.items():
                print(f"      - {k}: {v}")

            plot_explanation(data, node_imp_gnn, f"GNNExplainer - {nombre}", 
                           os.path.join(MODEL_DIR, f"gnnexplainer_{nombre}.png"))
            save_to_pymol(data, node_imp_gnn, nombre, "gnnexplainer", MODEL_DIR)
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
                edge_mask_type='object',
                model_config=dict(mode='regression', task_level='graph', return_type='raw')
            )
            exp_sal = explainer_sal(data.x, data.edge_index, edge_attr=data.edge_attr)
            node_imp_sal = exp_sal.node_mask.sum(dim=1).detach().numpy()

            # Derivar edge_mask proxy para poder calcular Fidelity y Sparsity
            proxy_edge_mask = node_imp_to_edge_mask(node_imp_sal, data.edge_index)

            def sal_wrapper_stability(x_pert, edge_index_pert, edge_attr_pert):
                exp = explainer_sal(x_pert, edge_index_pert, edge_attr=edge_attr_pert)
                node_imp = exp.node_mask.sum(dim=1).detach().numpy()
                return node_imp_to_edge_mask(node_imp, edge_index_pert)

            metricas_sal = evaluator.evaluate_explanation(
                data=data,
                edge_mask=exp_sal.edge_mask,
                explainer_func=sal_wrapper_stability,
                threshold=0.5
            )
            # Indicar en el JSON que Fidelity/Sparsity son estimaciones proxy
            metricas_sal["nota"] = "Fidelity y Sparsity calculadas sobre edge_mask proxy (media importancia nodos extremos)"

            all_results[nombre]["Saliency"] = metricas_sal

            print("    Métricas:")
            for k, v in metricas_sal.items():
                print(f"      - {k}: {v}")

            plot_explanation(data, node_imp_sal, f"Saliency - {nombre}", 
                           os.path.join(MODEL_DIR, f"saliency_{nombre}.png"))
            save_to_pymol(data, node_imp_sal, nombre, "saliency", MODEL_DIR)
            print(f"    ✓ Completado")

        except Exception as e:
            import traceback
            print(f"    ✗ Error: {e}")
            traceback.print_exc()
            all_results[nombre]["Saliency"] = {"error": str(e)}

    # ─────────────────────────────────────────────────────────────────
    # Guardar todas las métricas en un único JSON
    # ─────────────────────────────────────────────────────────────────
    metrics_path = os.path.join(MODEL_DIR, "xai_metrics.json")
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)
    print(f"\n  ✓ Métricas guardadas en: {metrics_path}")

    # ─────────────────────────────────────────────────────────────────
    # Resumen agregado por método
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("RESUMEN AGREGADO DE MÉTRICAS")
    print("="*60)

    for method in ["GNNExplainer", "Saliency"]:
        vals = {k: [] for k in ["Fidelity+ (prob)", "Fidelity- (prob)", "Sparsity", "Stability"]}
        for mol_data in all_results.values():
            m = mol_data.get(method, {})
            if "error" in m:
                continue
            for key in vals:
                v = m.get(key)
                if isinstance(v, (int, float)):
                    vals[key].append(v)

        print(f"\n  {method}:")
        for key, lst in vals.items():
            if lst:
                print(f"    {key:<25} media={np.mean(lst):.4f}  std={np.std(lst):.4f}  (n={len(lst)})")
            else:
                print(f"    {key:<25} N/A")

    print("\n" + "="*60)
    print("ANÁLISIS COMPLETADO")
    print("="*60)
    print(f"\nResultados en: {MODEL_DIR}/")
    print(f"  • Métricas XAI:  xai_metrics.json")
    print(f"  • Imágenes 2D:   gnnexplainer_*.png, saliency_*.png")
    print(f"  • Archivos PyMOL: gnnexplainer_*.pdb/.pml, saliency_*.pdb/.pml")
    print(f"\nTotal archivos de visualización: {len(indices_a_evaluar) * 4}")