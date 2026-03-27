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

# =====================================================================
# 1. RECONSTRUCCIÓN DEL MODELO GANADOR (GINE)
# =====================================================================
EDGE_DIM = 2

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
    
    def forward(self, x, edge_index, edge_attr, batch=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            
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

# =====================================================================
# 2. FUNCIÓN PARA DIBUJAR LOS MAPAS DE CALOR 2D (MEJORADO)
# =====================================================================
def plot_explanation(data, node_importance, title, save_path):
    """
    Dibuja el grafo coloreando los átomos del fármaco (Rojos) y de la proteína (Azules)
    según su importancia. Las aristas (enlaces) se pintan siempre de gris claro.
    El ligando se dibuja como bolitas.
    """
    plt.figure(figsize=(12, 12)) # Un poco más grande para que respire
    G = to_networkx(data, to_undirected=True)
    pos_2d = data.pos[:, :2].numpy() # Coordenadas X, Y reales
    
    # Separamos a quién pertenece cada átomo
    is_ligand = data.x[:, -1].numpy() == 1
    is_protein = ~is_ligand
    
    # 1. NORMALIZACIÓN INDEPENDIENTE (El truco para que la proteína se vea)
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
        
    # 2. Asignamos los colores y estilos
    colors = []
    idx_ligand, idx_protein = 0, 0
    
    for lig in is_ligand:
        if lig: # Fármaco
            # Escala de Rojos para el fármaco
            colors.append(plt.cm.Reds(imp_ligand[idx_ligand]))
            idx_ligand += 1
        else:   # Proteína
            # Escala de Azules para la proteína
            colors.append(plt.cm.Blues(imp_protein[idx_protein]))
            idx_protein += 1
            
    # 3. Dibujamos el grafo
    # Dibujamos las aristas (enlaces) PRIMERO, en gris claro, para que queden detrás
    nx.draw_networkx_edges(G, pos_2d, edge_color='lightgray', width=1.0, alpha=0.7)
    
    # Dibujamos los nodos del LIGANDO como bolitas (círculos rellenos)
    nx.draw_networkx_nodes(G, pos_2d, nodelist=np.where(is_ligand)[0].tolist(), 
                           node_color=[colors[i] for i in np.where(is_ligand)[0]],
                           node_size=150, node_shape='o', edgecolors='black', linewidths=0.5, alpha=0.9)
    
    # Dibujamos los nodos de la PROTEÍNA
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
    cbar_azul.set_label("Importancia Proteína (Normalizada)", fontweight='bold')
    
    # Eliminamos los ejes para una imagen más limpia
    plt.axis('off')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Imagen 2D guardada en: {save_path}")

# =====================================================================
# 2.5. EXPORTADOR AUTOMÁTICO A PYMOL (3D) - SIN TELARAÑA, GRIS Y BOLITAS
# =====================================================================
def save_to_pymol(data, node_importance, pdb_id, method_name, save_dir):
    is_ligand = data.x[:, -1].numpy() == 1
    
    # Normalización escala 0-100
    b_factors = np.zeros(data.num_nodes)
    for mask, color_range in [(is_ligand, "lig"), (~is_ligand, "prot")]:
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
            # Ocultamos enlaces entre fármaco y proteína 
            if is_ligand[n1] == is_ligand[n2]:
                f.write(f"CONECT{n1+1:>5}{n2+1:>5}\n")
        f.write("END\n")
        
    # 3. Escribir el Macro automático (.pml) - CONFIGURACIÓN FINAL DE VISUALIZACIÓN
    with open(pml_file, 'w') as f:
        f.write(f"load {base_name}.pdb, {base_name}\n")
        f.write("hide all\n")
        f.write("bg_color white\n\n")
        
        f.write(f"unbond (chain L), (chain P)\n") 
        
        f.write(f"h_add {base_name}\n") # Añade hidrógenos si faltan para mejorar enlaces
        f.write(f"connect {base_name}\n") 
        
        f.write("# --- CONFIGURACIÓN DE ARISTAS (NEGRO) ---\n")
        f.write(f"set stick_color, black, {base_name}\n")
        f.write(f"set line_color, black, {base_name}\n")
        f.write("set stick_radius, 0.08\n") # Un poco más grueso para que se vean bien
        f.write("set line_width, 1.5\n")
        f.write("set valence, 0\n\n")

        f.write("# --- FÁRMACO (Rojo) ---\n")
        f.write(f"show spheres, {base_name} and chain L\n")
        f.write(f"set sphere_scale, 0.25, {base_name} and chain L\n")
        f.write(f"spectrum b, white red, {base_name} and chain L, minimum=0, maximum=100\n")
        f.write(f"show sticks, {base_name} and chain L\n\n")
        
        f.write("# --- BOLSILLO PROTEÍNA (Azul) ---\n")
        f.write(f"show spheres, {base_name} and chain P\n")
        f.write(f"set sphere_scale, 0.20, {base_name} and chain P\n")
        f.write(f"spectrum b, white blue, {base_name} and chain P, minimum=0, maximum=100\n")
        f.write("set transparency, 0.6, {base_name} and chain P\n")
        f.write(f"show lines, {base_name} and chain P\n\n")
        
        f.write("# --- CÁMARA ---\n")
        f.write(f"center {base_name} and chain L\n")
        f.write(f"zoom {base_name} and chain L, 6\n")

# =====================================================================
# 3. EJECUCIÓN DEL INTERROGATORIO (MAIN)
# =====================================================================
if __name__ == "__main__":
    MODEL_DIR = "results/GINE"
    DATASET_PATH = "./data"
    DEVICE = torch.device('cpu') # Para explicabilidad, la CPU es más estable
    
    print("\n" + "="*60)
    print("INICIANDO ANÁLISIS DE INTERPRETABILIDAD (XAI)")
    print("="*60)
    
    # 1. Cargar la "Receta" del ganador
    with open(os.path.join(MODEL_DIR, 'best_params.json'), 'r') as f:
        best_params = json.load(f)
        
    # 2. Cargar una molécula de ejemplo (Usaremos la primera del Fold de validación)
    dataset = MProComplexDataset(root=DATASET_PATH, split='valid', fold_idx=0, mode='complex')
    
    # Construir modelo
    model = GINE_Model(in_channels=dataset.num_features, 
                       hidden_channels=best_params['hidden_channels'], 
                       num_layers=best_params['num_layers'], 
                       dropout=best_params['dropout']).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'best_model_overall.pth'), map_location=DEVICE))
    model.eval()

    # Vamos a analizar los 10 primeros fármacos del set de validación
    print("\n Escaneando el dataset para buscar las predicciones más exactas...")
    
    resultados = []
    # Pasamos rápidamente por todas las moléculas del valid_dataset
    for idx in range(len(dataset)):
        data = dataset[idx].to(DEVICE)
        pred_pIC50 = model(data.x, data.edge_index, data.edge_attr).item()
        real_pIC50 = data.y.item()
        
        # Calculamos el error absoluto (cuánto se ha desviado)
        error = abs(pred_pIC50 - real_pIC50)
        
        # Guardamos el índice, el error y los valores
        resultados.append({
            'indice': idx,
            'error': error,
            'pred': pred_pIC50,
            'real': real_pIC50
        })
        
    # Ordenamos la lista de resultados de menor error a mayor error
    resultados_ordenados = sorted(resultados, key=lambda x: x['error'])
    
    # Seleccionamos los 10 primeros (los que tienen el error más cercano a 0)
    top_10_mejores = resultados_ordenados[:10]
    
    print(" Los 10 fármacos con menor error son:")
    for rank, res in enumerate(top_10_mejores):
        print(f"  {rank+1}º -> Índice: {res['indice']} | Pred: {res['pred']:.3f} | Real: {res['real']:.3f} | Error: {res['error']:.3f}")
        
    # Extraemos solo los índices para pasárselos a tu bucle de imágenes
    indices_a_evaluar = [res['indice'] for res in top_10_mejores]
    
    for idx in indices_a_evaluar:
        data = dataset[idx].to(DEVICE)
        if hasattr(data, 'pdb_id'):
            # Si lo guardaste como pdb_id (ej. '7GGH')
            nombre_mol = str(data.pdb_id)
            if isinstance(data.pdb_id, list): nombre_mol = data.pdb_id[0]
        elif hasattr(data, 'id'):
            # Si lo guardaste como id
            nombre_mol = str(data.id)
            if isinstance(data.id, list): nombre_mol = data.id[0]
        elif hasattr(data, 'name'):
            nombre_mol = str(data.name)
        else:
            # Plan B: Si no se guardó el nombre en el dataset
            nombre_mol = f"Molecula_{idx}"
        
        pred_pIC50 = model(data.x, data.edge_index, data.edge_attr).item()
        print(f"\n{'-'*40}")
        print(f" Analizando {nombre_mol}...")
        print(f"   Predicción: {pred_pIC50:.3f} | Real: {data.y.item():.3f}")

        # 1. GNNExplainer
        explainer_gnn = Explainer(
            model=model, algorithm=GNNExplainer(epochs=200),
            explanation_type='model', node_mask_type='attributes', edge_mask_type='object',
            model_config=dict(mode='regression', task_level='graph', return_type='raw')
        )
        exp_gnn = explainer_gnn(data.x, data.edge_index, edge_attr=data.edge_attr)
        node_imp_gnn = exp_gnn.node_mask.sum(dim=1).detach().numpy()
        plot_explanation(data, node_imp_gnn, f"GNNExplainer - {nombre_mol}", os.path.join(MODEL_DIR, f"gnnexplainer_{nombre_mol}.png"))
        save_to_pymol(data, node_imp_gnn, nombre_mol, "gnnexplainer", MODEL_DIR)

        # 2. Saliency / CAM
        explainer_cam = Explainer(
            model=model, algorithm=CaptumExplainer('Saliency'),
            explanation_type='model', node_mask_type='attributes', edge_mask_type=None,
            model_config=dict(mode='regression', task_level='graph', return_type='raw')
        )
        exp_cam = explainer_cam(data.x, data.edge_index, edge_attr=data.edge_attr)
        node_imp_cam = exp_cam.node_mask.sum(dim=1).detach().numpy()
        plot_explanation(data, node_imp_cam, f"Saliency - {nombre_mol}", os.path.join(MODEL_DIR, f"saliency_{nombre_mol}.png"))
        save_to_pymol(data, node_imp_cam, nombre_mol, "saliency", MODEL_DIR)

    print(f"\nAnálisis completado: Revisa la carpeta {MODEL_DIR}/ para ver las 20 imágenes de XAI.")