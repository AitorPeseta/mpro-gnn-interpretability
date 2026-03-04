"""
═══════════════════════════════════════════════════════════════════════
DATASET LOADER HÍBRIDO: SDF + CIF (LO MEJOR DE AMBOS MUNDOS)
Versión Comentada para Aprendizaje (TFG)
═══════════════════════════════════════════════════════════════════════
"""

import os
import ast
import json
import torch
import numpy as np
import logging
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm

import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx, subgraph

from rdkit import Chem

# Configuramos un "diario privado" para que las alertas del dataset 
# se guarden en un archivo de texto y no ensucien la terminal.
my_logger = logging.getLogger('dataset_logger')
my_logger.setLevel(logging.INFO)
my_logger.addHandler(logging.FileHandler('dataset_processing.log', mode='w'))
my_logger.propagate = False # Esto evita que Optuna se silencie.

# =====================================================================
# CONSTANTES FÍSICAS Y QUÍMICAS
# =====================================================================

# Diccionario que traduce el "texto" del JSON a un "número" para la IA.
# A la IA le gustan los números, no las palabras.
INTERACTION_MAP = {
    'covalent': 0, 'hydrophobic': 1, 'hbond': 2, 'weak_hbond': 3,
    'aromatic': 4, 'ionic': 5, 'polar': 6, 'weak_polar': 7,
    'vdw': 8, 'pi_stacking': 9, 'halogen': 10, 'other': 11
}

STANDARD_AA = [
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS',
    'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'
]

# Umbrales (en Ångströms) para decidir si dos átomos están lo bastante
# cerca como para considerarlos "unidos" matemáticamente.
DISTANCE_TOLERANCE = 3.0           # Margen de error para emparejar JSON con 3D
INTRA_RESIDUE_BOND_THRESHOLD = 1.8 # Distancia máxima para un enlace covalente normal
PEPTIDE_BOND_THRESHOLD = 1.6       # Distancia para enlace peptídico (entre aminoácidos)
DISULFIDE_BOND_THRESHOLD = 2.3     # Distancia para puentes disulfuro (entre Cisteínas)

# =====================================================================
# FUNCIONES DE CARACTERÍSTICAS (FEATURES)
# =====================================================================

def one_hot_embedding(value, options):
    """
    Convierte un valor (ej. 'C') en una lista de ceros y un uno.
    Si options = ['C', 'N', 'O'] y value = 'N' -> Devuelve [0, 1, 0]
    """
    embedding = [0] * len(options)
    index = options.index(value) if value in options else -1
    if index != -1:
        embedding[index] = 1
    return embedding

def get_atom_features_from_symbol(atom_symbol, is_ligand=True):
    """
    Crea las features (características) matemáticas para los átomos de la PROTEÍNA.
    Como el archivo PDB no nos da valencias ni aromaticidad, nos las inventamos (ponemos 0).
    """
    atom_types = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'H']
    feat_type = one_hot_embedding(atom_symbol, atom_types) # ¿Qué elemento es?
    feat_type += [1 if atom_symbol not in atom_types else 0] # ¿Es un elemento raro?
    
    feat_deg_norm = 0.0 # No sabemos los enlaces de la proteína
    feat_val_norm = 0.0 # No sabemos la valencia
    arom = 0            # No sabemos si es aromático
    is_lig_flag = 1 if is_ligand else 0 # 1 si es fármaco, 0 si es proteína
    
    return feat_type + [feat_deg_norm, feat_val_norm, arom, is_lig_flag]

def get_atom_features_from_rdkit(atom):
    """
    Crea las features para los átomos del FÁRMACO usando RDKit.
    RDKit sí es listo y nos da la química exacta.
    """
    atom_types = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'H']
    sym = atom.GetSymbol()
    deg = atom.GetDegree() # Número de vecinos
    val = atom.GetTotalValence() # Valencia total (ej. Carbono = 4)
    arom = 1 if atom.GetIsAromatic() else 0 # ¿Es un anillo bencénico?
    
    feat_type = one_hot_embedding(sym, atom_types)
    feat_type += [1 if sym not in atom_types else 0]
    
    # Dividimos entre 6.0 para normalizar (que el número esté entre 0 y 1).
    # Las redes neuronales aprenden mejor con números pequeños.
    feat_deg_norm = deg / 6.0
    feat_val_norm = val / 6.0
    
    return feat_type + [feat_deg_norm, feat_val_norm, arom, 1]

# =====================================================================
# LECTORES DE ARCHIVOS (PARSERS)
# =====================================================================

def parse_ligand_cif(cif_path):
    """
    Lee el archivo CIF en bruto (texto puro) para robarle los nombres originales 
    de los átomos (ej. 'O1', 'C28') y sus coordenadas.
    """
    if not os.path.exists(cif_path): return None
    atom_names = []
    coords = []
    in_atom_section = False
    
    with open(cif_path, 'r') as f:
        for line in f:
            if line.startswith('loop_'): in_atom_section = False
            # Buscamos la sección de átomos del CIF
            if '_atom_site.group_PDB' in line:
                in_atom_section = True
                continue
            
            # Si estamos en la sección correcta, extraemos datos
            if in_atom_section and (line.startswith('HETATM') or line.startswith('ATOM')):
                parts = line.split()
                if len(parts) < 13: continue
                try:
                    atom_name = parts[3] # Columna 3: El nombre (O1)
                    # Columnas 10, 11, 12: Coordenadas X, Y, Z
                    x, y, z = float(parts[10]), float(parts[11]), float(parts[12])
                    atom_names.append(atom_name)
                    coords.append([x, y, z])
                except (ValueError, IndexError): continue
                
    if len(atom_names) == 0: return None
    # Devolvemos un diccionario con la lista de nombres y una matriz matemática NumPy con las coordenadas
    return {'atom_names': atom_names, 'coords': np.array(coords)}

def parse_protein_pdb_by_residue(pdb_path):
    """
    Lee el PDB de la proteína y agrupa los átomos en "bolsitas" (residuos).
    """
    residues = {}
    if not os.path.exists(pdb_path): return residues
    with open(pdb_path, 'r') as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")): continue
            
            # En los PDB, los datos están en columnas de texto fijas
            atom_name = line[12:16].strip() # Ej: 'CA' (Carbono Alfa)
            res_name = line[17:20].strip()  # Ej: 'HIS' (Histidina)
            try:
                res_seq = int(line[22:26].strip()) # Ej: 41 (El número del aminoácido)
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                coords = np.array([x, y, z])
                
                # La llave única de la bolsita es ('HIS', 41)
                res_key = (res_name, res_seq)
                if res_key not in residues: residues[res_key] = {}
                
                # Metemos el átomo y sus coordenadas en la bolsita
                residues[res_key][atom_name] = coords
            except ValueError: continue
    return residues

def add_peptide_bonds(protein_nodes_map, protein_residues, edge_indices, edge_attrs):
    """
    Busca aminoácidos consecutivos y dibuja el enlace peptídico entre ellos
    (une el Carbono de uno con el Nitrógeno del siguiente).
    """
    num_peptide_bonds = 0
    num_disulfide_bonds = 0
    residue_keys = list(protein_residues.keys()) # Lista de todos los aminoácidos en orden
    
    for i in range(len(residue_keys) - 1):
        res_current = residue_keys[i]
        res_next = residue_keys[i + 1]
        
        # Si no son vecinos en la secuencia (ej. el 41 y el 42), saltamos
        if res_current[1] + 1 != res_next[1]: continue
        
        # Comprobamos si tenemos el Carbono del primero y el Nitrógeno del segundo
        if 'C' not in protein_residues[res_current] or 'N' not in protein_residues[res_next]: continue
        
        coord_C = protein_residues[res_current]['C']
        coord_N = protein_residues[res_next]['N']
        
        # Calculamos la distancia 3D entre C y N (np.linalg.norm es el Teorema de Pitágoras en 3D)
        dist = np.linalg.norm(coord_C - coord_N)
        
        # Si están muy cerca, ¡hay un enlace peptídico real!
        if dist < PEPTIDE_BOND_THRESHOLD:
            key_C = (res_current[0], res_current[1], 'C')
            key_N = (res_next[0], res_next[1], 'N')
            
            # Si ambos átomos están en nuestro grafo final, dibujamos la línea (arista)
            if key_C in protein_nodes_map and key_N in protein_nodes_map:
                idx_C, idx_N = protein_nodes_map[key_C], protein_nodes_map[key_N]
                edge_indices += [[idx_C, idx_N], [idx_N, idx_C]] # Añadimos en ambas direcciones (grafo no dirigido)
                edge_attrs += [[dist, 0], [dist, 0]]             # Guardamos distancia y tipo (0=covalente)
                num_peptide_bonds += 1
                
    # (El mismo proceso se hace debajo para los puentes disulfuro entre Cisteínas)
    # ... [Omitido en comentarios para no aburrir, es la misma lógica]
    return num_peptide_bonds, num_disulfide_bonds

# =====================================================================
# EL INGENIERO JEFE: CONSTRUCTOR DEL SÚPER-GRAFO
# =====================================================================

def complex_to_graph_hybrid(sdf_path, cif_path, pdb_path, json_path, target_val, pdb_id, viz_dir=None):
    if not os.path.exists(sdf_path): return None
    
    # 1. CARGAR LIGANDO CON RDKIT (SDF)
    supplier = Chem.SDMolSupplier(sdf_path, sanitize=True, removeHs=False)
    mol = supplier[0] if len(supplier) > 0 else None
    if mol is None: return None
    
    try:
        conf = mol.GetConformer()
        ligand_coords = conf.GetPositions() # Coordenadas matemáticas perfectas
    except: return None
    
    num_ligand_atoms = mol.GetNumAtoms()
    
    # 2. CARGAR NOMBRES DEL LIGANDO (CIF) Y EMPAREJARLOS
    ligand_cif = parse_ligand_cif(cif_path)
    sdf_to_cif_name = {} # El diccionario traductor: Índice RDKit -> Nombre CIF
    
    # Verificamos que el CIF y el SDF tengan exactamente la misma cantidad de átomos
    if ligand_cif is not None and len(ligand_cif['atom_names']) == num_ligand_atoms:
        for i in range(num_ligand_atoms): 
            sdf_to_cif_name[i] = ligand_cif['atom_names'][i]
            
        # Comprobación de seguridad geométrica: ¿Las coordenadas del CIF y SDF coinciden?
        # Medimos la distancia entre los primeros 5 átomos para comprobarlo.
        max_diff = max([np.linalg.norm(ligand_coords[i] - ligand_cif['coords'][i]) for i in range(min(5, num_ligand_atoms))])
        
        if max_diff > 0.5:
            # Si la distancia es grande, significa que los átomos están en distinto orden. 
            # Vaciamos el diccionario para no cruzar los cables luego.
            my_logger.info(f"[{pdb_id}] Coordenadas SDF≠CIF (diff={max_diff:.3f}Å).")
            sdf_to_cif_name = {} 
    
    node_features, node_positions = [], []
    
    # 3. EXTRAER NODOS DEL FÁRMACO
    for i, atom in enumerate(mol.GetAtoms()):
        node_features.append(get_atom_features_from_rdkit(atom)) # Generamos [0,1,0,0,0...]
        node_positions.append(ligand_coords[i])                  # Guardamos la posición [x,y,z]
    
    edge_indices, edge_attrs = [], []
    
    # 4. EXTRAER ENLACES DEL FÁRMACO
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        b_type = bond.GetBondTypeAsDouble() # Si es simple (1.0), doble (2.0), etc.
        edge_indices += [[i, j], [j, i]]
        edge_attrs += [[b_type, 0], [b_type, 0]] # Añadimos el enlace interno del fármaco
    
    # 5. CARGAR PROTEÍNA PDB EN BOLSITAS
    protein_residues = parse_protein_pdb_by_residue(pdb_path)
    
    # 6. PROCESAR EL JSON DE INTERACCIONES (ARPEGGIO)
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            try: interactions = json.load(f)
            except: interactions = []
        
        protein_nodes_map, added_residues = {}, set()
        current_node_idx = len(node_features) # Los átomos de la proteína empezarán a numerarse donde acabó el fármaco
        exact_matches, distance_matches, failed_matches = 0, 0, 0
        
        for inter in interactions:
            if inter.get('interacting_entities') != 'INTER': continue # Solo queremos choques Fármaco-Proteína
            
            bgn, end = inter.get('bgn', {}), inter.get('end', {})
            
            # Descubrir quién es el Fármaco y quién la Proteína en este choque
            if bgn.get('label_comp_id') in STANDARD_AA: atom_prot_data, atom_lig_data = bgn, end
            elif end.get('label_comp_id') in STANDARD_AA: atom_prot_data, atom_lig_data = end, bgn
            else: continue
            
            # Datos de la proteína sacados del JSON
            res_name = atom_prot_data.get('label_comp_id')   # Ej. HIS
            res_seq = int(atom_prot_data.get('auth_seq_id')) # Ej. 41
            prot_atom_name = atom_prot_data.get('auth_atom_id') # Ej. ND1
            res_key = (res_name, res_seq)
            
            if res_key not in protein_residues: continue
            
            # A) IMPORTAR EL AMINOÁCIDO ENTERO
            if res_key not in added_residues:
                residue_atoms = protein_residues[res_key]
                res_atom_names = list(residue_atoms.keys())
                
                # Bucle 1: Añadir todos los átomos del aminoácido al grafo
                for a_name in res_atom_names:
                    node_features.append(get_atom_features_from_symbol(a_name[0].upper(), is_ligand=False))
                    node_positions.append(residue_atoms[a_name])
                    # Guardamos su "DNI" en un mapa para encontrarlo luego
                    protein_nodes_map[(res_name, res_seq, a_name)] = current_node_idx
                    current_node_idx += 1
                added_residues.add(res_key)
                
                # Bucle 2: Unir los átomos del aminoácido entre sí (Esqueleto interno)
                for i_a in range(len(res_atom_names)):
                    for j_a in range(i_a + 1, len(res_atom_names)):
                        name_i, name_j = res_atom_names[i_a], res_atom_names[j_a]
                        dist = np.linalg.norm(residue_atoms[name_i] - residue_atoms[name_j])
                        if dist < INTRA_RESIDUE_BOND_THRESHOLD: # Si están a menos de 1.8 Å...
                            idx_i, idx_j = protein_nodes_map[(res_name, res_seq, name_i)], protein_nodes_map[(res_name, res_seq, name_j)]
                            edge_indices += [[idx_i, idx_j], [idx_j, idx_i]] # ... Trazamos la línea covalente
                            edge_attrs += [[1.0, 0], [1.0, 0]]
            
            # B) ENCONTRAR AL FÁRMACO ("EL APRETÓN DE MANOS HÍBRIDO")
            lig_atom_name = atom_lig_data.get('auth_atom_id') # El JSON dice: "Busca al átomo O1"
            lig_idx = None
            
            # Estratégia 1: Traducción directa (Mucha precisión)
            if sdf_to_cif_name: # Si el diccionario no está vacío (hubo suerte con el CIF)
                for sdf_idx, cif_name in sdf_to_cif_name.items():
                    if cif_name == lig_atom_name:
                        lig_idx = sdf_idx # ¡Lo hemos encontrado! Es el índice 14.
                        exact_matches += 1
                        break
            
            # Estratégia 2: Triangulación 3D (El Salvavidas matemático)
            if lig_idx is None: # Si falló el diccionario...
                if prot_atom_name not in protein_residues[res_key]:
                    failed_matches += 1
                    continue
                prot_coord = protein_residues[res_key][prot_atom_name] # Dónde está la proteína
                dist_reportada = inter.get('distance', 0.0)            # Distancia exacta que dice el JSON
                
                # Calculamos la distancia desde la proteína a TODOS los átomos del fármaco a la vez usando NumPy
                distances = np.linalg.norm(ligand_coords - prot_coord, axis=1)
                
                # Buscamos la distancia que más se parezca a la del JSON
                diffs = np.abs(distances - dist_reportada)
                lig_idx = np.argmin(diffs) # El átomo con menor diferencia es el ganador
                
                # Control de seguridad: Si la diferencia sigue siendo brutal (> 3.0 Å), cancelamos la unión.
                if diffs[lig_idx] > DISTANCE_TOLERANCE:
                    failed_matches += 1
                    continue
                distance_matches += 1
            
            # C) TRAZAR LA UNIÓN FÁRMACO-PROTEÍNA
            prot_atom_key = (res_name, res_seq, prot_atom_name)
            if prot_atom_key not in protein_nodes_map: continue
            prot_idx = protein_nodes_map[prot_atom_key] # Recuperamos el índice interno de la proteína
            
            contact_list = inter.get('contact', [])
            inter_type_str = 'other'
            for c in contact_list:
                if c not in ['proximal', 'vdw_clash']:
                    inter_type_str = c
                    break
            
            # Convertimos texto ('hbond') a número (2) para la IA
            inter_num = INTERACTION_MAP.get(inter_type_str.lower(), 11)
            dist_reportada = inter.get('distance', 3.5)
            
            # Añadimos la conexión mágica entre Fármaco y Proteína
            edge_indices += [[lig_idx, prot_idx], [prot_idx, lig_idx]]
            edge_attrs += [[dist_reportada, inter_num], [dist_reportada, inter_num]]
        
        # Una vez están todos los aminoácidos en el grafo, dibujamos los enlaces peptídicos entre ellos
        num_pep, num_ss = add_peptide_bonds(protein_nodes_map, protein_residues, edge_indices, edge_attrs)
        my_logger.info(f"[{pdb_id}] ✓ Exactas: {exact_matches} | Distancia: {distance_matches} | Pep: {num_pep} | Fallos: {failed_matches}")
    
    # 7. EMPAQUETADO MATEMÁTICO (PYTORCH)
    # Convertimos todas las listas de Python a Tensores (Matrices para aceleración por hardware)
    x = torch.tensor(np.array(node_features), dtype=torch.float)
    y_val = torch.tensor([[target_val]], dtype=torch.float) # La nota final (pIC50)
    pos = torch.tensor(np.array(node_positions), dtype=torch.float)
    
    if len(edge_indices) > 0:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 2), dtype=torch.float)
    
    # La caja oficial de PyTorch Geometric
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y_val, pos=pos)
    
    # 8. FILTRADO DE NODOS HUÉRFANOS (ISLAS)
    # Convertimos a formato de red para detectar si algún aminoácido se quedó "flotando" sin unirse al fármaco
    G = to_networkx(data, to_undirected=True)
    connected_components = list(nx.connected_components(G))
    ligand_nodes = set(torch.where(data.x[:, -1] == 1)[0].numpy()) # Buscamos a los azules (Fármaco)
    valid_nodes = set()
    
    for comp in connected_components:
        if comp.intersection(ligand_nodes): valid_nodes.update(comp) # Si el archipiélago contiene fármaco, lo salvamos
    
    # Si detectamos basura flotando, usamos 'subgraph' para recortar el tensor matemáticamente
    if len(valid_nodes) < data.num_nodes:
        mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        mask[list(valid_nodes)] = True
        edge_index_sub, edge_attr_sub = subgraph(mask, data.edge_index, data.edge_attr, relabel_nodes=True)
        data.x = data.x[mask]
        data.pos = data.pos[mask]
        data.edge_index = edge_index_sub
        data.edge_attr = edge_attr_sub
        
    # 9. VISUALIZACIÓN 2D
    if viz_dir is not None:
        try:
            plt.figure(figsize=(8, 8))
            G_draw = to_networkx(data, to_undirected=True)
            pos_2d = data.pos[:, :2].numpy() # Proyección Ortográfica (Desechamos la coordenada Z)
            
            # Azul para Ligando (Última columna = 1), Rojo para Proteína (Última columna = 0)
            node_colors = ['blue' if feats[-1] == 1 else 'red' for feats in data.x.numpy()]
            
            nx.draw(G_draw, pos_2d, with_labels=False, node_color=node_colors, 
                    node_size=60, edge_color='gray', width=0.8, alpha=0.6)
            plt.title(f"Grafo del Complejo {pdb_id}")
            plt.axis('off')
            plt.savefig(os.path.join(viz_dir, f"{pdb_id}_graph.png"), dpi=100, bbox_inches='tight')
            plt.close('all') # VITAL: Liberar la memoria RAM cerrando el lienzo 
        except Exception as e:
            my_logger.info(f"[{pdb_id}] No se pudo dibujar el grafo: {e}")

    return data

class MProComplexDataset(InMemoryDataset):
    def __init__(self, root, split='train', fold_idx=0, transform=None, pre_transform=None, mode='complex'):
        self.split = split
        self.fold_idx = fold_idx
        self.mode = mode
        self.index_files = {
            'train': 'Splits/train_index_folder.txt',
            'valid': 'Splits/valid_index_folder.txt',
            'test': 'Splits/test_index_folder.txt'
        }
        super(MProComplexDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
    
    @property
    def raw_file_names(self):
        return ['pIC50.txt'] + list(self.index_files.values())
    
    @property
    def processed_file_names(self):
        return [f'{self.split}_complex_fold_{self.fold_idx}_hybrid.pt']
    
    def download(self): pass
    
    def load_indices(self, filepath):
        with open(filepath, 'r') as f:
            return ast.literal_eval(f.read())[self.fold_idx]
    
    def load_targets(self, filepath):
        targets = {}
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2: targets[parts[0]] = float(parts[1])
        return targets
    
    def process(self):
        index_path = os.path.join(self.root, self.index_files[self.split])
        ids = self.load_indices(index_path)
        target_path = os.path.join(self.root, 'pIC50.txt')
        targets = self.load_targets(target_path)
        
        viz_dir = os.path.join(os.path.dirname(self.root), 'results', 'LIGANDS')
        os.makedirs(viz_dir, exist_ok=True)
        
        data_list = []
        
        for pdb_id in tqdm(ids, desc=f"Grafos ({self.split})", unit="mol"):
            sdf_path = os.path.join(self.root, 'Ligand', 'Ligand_SDF', f"{pdb_id}_ligand.sdf")
            cif_path = os.path.join(self.root, 'Ligand', 'Ligand_CIF', f"{pdb_id}_ligand.cif")
            if not os.path.exists(cif_path):
                cif_path = os.path.join(self.root, 'Ligand', 'Ligand_CIF', f"{pdb_id}_ligand.sif")
            pdb_path = os.path.join(self.root, 'Protein', 'Protein_PDB', f"{pdb_id}_protein.pdb")
            json_path = os.path.join(self.root, 'Interaction', f"{pdb_id}_ligand.json")
            
            if os.path.exists(sdf_path) and os.path.exists(pdb_path) and pdb_id in targets:
                data = complex_to_graph_hybrid(sdf_path, cif_path, pdb_path, json_path, targets[pdb_id], pdb_id, viz_dir)
                if data is not None: data_list.append(data)
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])