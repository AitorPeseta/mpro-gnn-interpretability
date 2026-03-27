"""
═══════════════════════════════════════════════════════════════════════
DATASET LOADER HÍBRIDO: SDF + CIF (LO MEJOR DE AMBOS MUNDOS)
Versión Comentada para Aprendizaje (TFG) - NORMALIZACIÓN AVANZADA
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
my_logger.propagate = False

# =====================================================================
# CONSTANTES FÍSICAS Y QUÍMICAS
# =====================================================================

INTERACTION_MAP = {
    'covalent': 0, 'hydrophobic': 1, 'hbond': 2, 'weak_hbond': 3,
    'aromatic': 4, 'ionic': 5, 'polar': 6, 'weak_polar': 7,
    'vdw': 8, 'pi_stacking': 9, 'halogen': 10, 'other': 11
}

STANDARD_AA = [
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS',
    'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'
]

DISTANCE_TOLERANCE = 3.0           
INTRA_RESIDUE_BOND_THRESHOLD = 1.8 
PEPTIDE_BOND_THRESHOLD = 1.6       
DISULFIDE_BOND_THRESHOLD = 2.3     

# [NUEVO] Límite máximo para normalizar distancias de enlaces (entre 0 y 1)
MAX_DIST_THEORETICAL = 6.0 

# [NUEVO] Rango global de coordenadas X, Y, Z de todo el dataset para escalado Min-Max
POS_MIN = torch.tensor([-28.0, -36.0, -34.0], dtype=torch.float)
POS_MAX = torch.tensor([39.0, 37.0, 42.0], dtype=torch.float)

# =====================================================================
# FUNCIONES DE CARACTERÍSTICAS (FEATURES)
# =====================================================================

def one_hot_embedding(value, options):
    embedding = [0] * len(options)
    index = options.index(value) if value in options else -1
    if index != -1:
        embedding[index] = 1
    return embedding

# [NUEVO] Función para codificar interacciones como One-Hot
def one_hot_interaction(inter_num, num_classes=12):
    vec = [0.0] * num_classes
    if 0 <= inter_num < num_classes:
        vec[inter_num] = 1.0
    return vec

def get_atom_features_from_symbol(atom_symbol, is_ligand=True):
    atom_types = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'H']
    feat_type = one_hot_embedding(atom_symbol, atom_types) 
    feat_type += [1 if atom_symbol not in atom_types else 0] 
    
    feat_deg_norm = 0.0 
    feat_val_norm = 0.0 
    arom = 0            
    is_lig_flag = 1 if is_ligand else 0 
    
    return feat_type + [feat_deg_norm, feat_val_norm, arom, is_lig_flag]

def get_atom_features_from_rdkit(atom):
    atom_types = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'H']
    sym = atom.GetSymbol()
    deg = atom.GetDegree() 
    val = atom.GetTotalValence() 
    arom = 1 if atom.GetIsAromatic() else 0 
    
    feat_type = one_hot_embedding(sym, atom_types)
    feat_type += [1 if sym not in atom_types else 0]
    
    feat_deg_norm = deg / 6.0
    feat_val_norm = val / 6.0
    
    return feat_type + [feat_deg_norm, feat_val_norm, arom, 1]

# =====================================================================
# LECTORES DE ARCHIVOS (PARSERS)
# =====================================================================

def parse_ligand_cif(cif_path):
    if not os.path.exists(cif_path): return None
    atom_names = []
    coords = []
    in_atom_section = False
    
    with open(cif_path, 'r') as f:
        for line in f:
            if line.startswith('loop_'): in_atom_section = False
            if '_atom_site.group_PDB' in line:
                in_atom_section = True
                continue
            
            if in_atom_section and (line.startswith('HETATM') or line.startswith('ATOM')):
                parts = line.split()
                if len(parts) < 13: continue
                try:
                    atom_name = parts[3] 
                    x, y, z = float(parts[10]), float(parts[11]), float(parts[12])
                    atom_names.append(atom_name)
                    coords.append([x, y, z])
                except (ValueError, IndexError): continue
                
    if len(atom_names) == 0: return None
    return {'atom_names': atom_names, 'coords': np.array(coords)}

def parse_protein_pdb_by_residue(pdb_path):
    residues = {}
    if not os.path.exists(pdb_path): return residues
    with open(pdb_path, 'r') as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")): continue
            
            atom_name = line[12:16].strip() 
            res_name = line[17:20].strip()  
            try:
                res_seq = int(line[22:26].strip()) 
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                coords = np.array([x, y, z])
                
                res_key = (res_name, res_seq)
                if res_key not in residues: residues[res_key] = {}
                
                residues[res_key][atom_name] = coords
            except ValueError: continue
    return residues

def add_peptide_bonds(protein_nodes_map, protein_residues, edge_indices, edge_attrs):
    num_peptide_bonds = 0
    num_disulfide_bonds = 0
    residue_keys = list(protein_residues.keys()) 
    
    for i in range(len(residue_keys) - 1):
        res_current = residue_keys[i]
        res_next = residue_keys[i + 1]
        
        if res_current[1] + 1 != res_next[1]: continue
        if 'C' not in protein_residues[res_current] or 'N' not in protein_residues[res_next]: continue
        
        coord_C = protein_residues[res_current]['C']
        coord_N = protein_residues[res_next]['N']
        dist = np.linalg.norm(coord_C - coord_N)
        
        if dist < PEPTIDE_BOND_THRESHOLD:
            key_C = (res_current[0], res_current[1], 'C')
            key_N = (res_next[0], res_next[1], 'N')
            
            if key_C in protein_nodes_map and key_N in protein_nodes_map:
                idx_C, idx_N = protein_nodes_map[key_C], protein_nodes_map[key_N]
                
                # [MODIFICADO] Normalización y One-Hot para enlaces peptídicos
                dist_norm = min(dist / MAX_DIST_THEORETICAL, 1.0)
                inter_one_hot = one_hot_interaction(0) # 0 = covalente
                
                edge_indices += [[idx_C, idx_N], [idx_N, idx_C]] 
                edge_attrs += [[dist_norm] + inter_one_hot, [dist_norm] + inter_one_hot]             
                num_peptide_bonds += 1
                
    # (Lógica simplificada para puentes disulfuro también omitida para mantener brevedad)
    return num_peptide_bonds, num_disulfide_bonds

# =====================================================================
# EL INGENIERO JEFE: CONSTRUCTOR DEL SÚPER-GRAFO
# =====================================================================

def complex_to_graph_hybrid(sdf_path, cif_path, pdb_path, json_path, target_val, pdb_id, viz_dir=None):
    if not os.path.exists(sdf_path): return None
    
    supplier = Chem.SDMolSupplier(sdf_path, sanitize=True, removeHs=False)
    mol = supplier[0] if len(supplier) > 0 else None
    if mol is None: return None
    
    try:
        conf = mol.GetConformer()
        ligand_coords = conf.GetPositions() 
    except: return None
    
    num_ligand_atoms = mol.GetNumAtoms()
    
    ligand_cif = parse_ligand_cif(cif_path)
    sdf_to_cif_name = {} 
    
    if ligand_cif is not None and len(ligand_cif['atom_names']) == num_ligand_atoms:
        for i in range(num_ligand_atoms): 
            sdf_to_cif_name[i] = ligand_cif['atom_names'][i]
            
        max_diff = max([np.linalg.norm(ligand_coords[i] - ligand_cif['coords'][i]) for i in range(min(5, num_ligand_atoms))])
        
        if max_diff > 0.5:
            my_logger.info(f"[{pdb_id}] Coordenadas SDF≠CIF (diff={max_diff:.3f}Å).")
            sdf_to_cif_name = {} 
    
    node_features, node_positions = [], []
    
    for i, atom in enumerate(mol.GetAtoms()):
        node_features.append(get_atom_features_from_rdkit(atom)) 
        node_positions.append(ligand_coords[i])                  
    
    edge_indices, edge_attrs = [], []
    
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        b_type = bond.GetBondTypeAsDouble() 
        
        # [MODIFICADO] Normalización y One-Hot para el fármaco
        dist_norm = min(b_type / MAX_DIST_THEORETICAL, 1.0)
        inter_one_hot = one_hot_interaction(0) # 0 = covalente interno
        
        edge_indices += [[i, j], [j, i]]
        edge_attrs += [[dist_norm] + inter_one_hot, [dist_norm] + inter_one_hot]
    
    protein_residues = parse_protein_pdb_by_residue(pdb_path)
    
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            try: interactions = json.load(f)
            except: interactions = []
        
        protein_nodes_map, added_residues = {}, set()
        current_node_idx = len(node_features) 
        exact_matches, distance_matches, failed_matches = 0, 0, 0
        
        for inter in interactions:
            if inter.get('interacting_entities') != 'INTER': continue 
            
            bgn, end = inter.get('bgn', {}), inter.get('end', {})
            
            if bgn.get('label_comp_id') in STANDARD_AA: atom_prot_data, atom_lig_data = bgn, end
            elif end.get('label_comp_id') in STANDARD_AA: atom_prot_data, atom_lig_data = end, bgn
            else: continue
            
            res_name = atom_prot_data.get('label_comp_id')   
            res_seq = int(atom_prot_data.get('auth_seq_id')) 
            prot_atom_name = atom_prot_data.get('auth_atom_id') 
            res_key = (res_name, res_seq)
            
            if res_key not in protein_residues: continue
            
            if res_key not in added_residues:
                residue_atoms = protein_residues[res_key]
                res_atom_names = list(residue_atoms.keys())
                
                for a_name in res_atom_names:
                    node_features.append(get_atom_features_from_symbol(a_name[0].upper(), is_ligand=False))
                    node_positions.append(residue_atoms[a_name])
                    protein_nodes_map[(res_name, res_seq, a_name)] = current_node_idx
                    current_node_idx += 1
                added_residues.add(res_key)
                
                for i_a in range(len(res_atom_names)):
                    for j_a in range(i_a + 1, len(res_atom_names)):
                        name_i, name_j = res_atom_names[i_a], res_atom_names[j_a]
                        dist = np.linalg.norm(residue_atoms[name_i] - residue_atoms[name_j])
                        if dist < INTRA_RESIDUE_BOND_THRESHOLD:
                            idx_i, idx_j = protein_nodes_map[(res_name, res_seq, name_i)], protein_nodes_map[(res_name, res_seq, name_j)]
                            
                            # [MODIFICADO] Normalización y One-Hot para el esqueleto de la proteína
                            dist_norm = min(dist / MAX_DIST_THEORETICAL, 1.0)
                            inter_one_hot = one_hot_interaction(0)
                            
                            edge_indices += [[idx_i, idx_j], [idx_j, idx_i]] 
                            edge_attrs += [[dist_norm] + inter_one_hot, [dist_norm] + inter_one_hot]
            
            lig_atom_name = atom_lig_data.get('auth_atom_id') 
            lig_idx = None
            
            if sdf_to_cif_name: 
                for sdf_idx, cif_name in sdf_to_cif_name.items():
                    if cif_name == lig_atom_name:
                        lig_idx = sdf_idx 
                        exact_matches += 1
                        break
            
            if lig_idx is None: 
                if prot_atom_name not in protein_residues[res_key]:
                    failed_matches += 1
                    continue
                prot_coord = protein_residues[res_key][prot_atom_name] 
                dist_reportada = inter.get('distance', 0.0)            
                
                distances = np.linalg.norm(ligand_coords - prot_coord, axis=1)
                diffs = np.abs(distances - dist_reportada)
                lig_idx = np.argmin(diffs) 
                
                if diffs[lig_idx] > DISTANCE_TOLERANCE:
                    failed_matches += 1
                    continue
                distance_matches += 1
            
            prot_atom_key = (res_name, res_seq, prot_atom_name)
            if prot_atom_key not in protein_nodes_map: continue
            prot_idx = protein_nodes_map[prot_atom_key] 
            
            contact_list = inter.get('contact', [])
            inter_type_str = 'other'
            for c in contact_list:
                if c not in ['proximal', 'vdw_clash']:
                    inter_type_str = c
                    break
            
            inter_num = INTERACTION_MAP.get(inter_type_str.lower(), 11)
            dist_reportada = inter.get('distance', 3.5)
            
            # [MODIFICADO] Normalización y One-Hot para enlaces Fármaco-Proteína
            dist_norm = min(dist_reportada / MAX_DIST_THEORETICAL, 1.0)
            inter_one_hot = one_hot_interaction(inter_num)
            
            edge_indices += [[lig_idx, prot_idx], [prot_idx, lig_idx]]
            edge_attrs += [[dist_norm] + inter_one_hot, [dist_norm] + inter_one_hot]
        
        num_pep, num_ss = add_peptide_bonds(protein_nodes_map, protein_residues, edge_indices, edge_attrs)
        my_logger.info(f"[{pdb_id}] ✓ Exactas: {exact_matches} | Distancia: {distance_matches} | Pep: {num_pep} | Fallos: {failed_matches}")
    
    # 7. EMPAQUETADO MATEMÁTICO (PYTORCH)
    x = torch.tensor(np.array(node_features), dtype=torch.float)
    y_val = torch.tensor([[target_val]], dtype=torch.float) 
    
    # [MODIFICADO] Aplicando la normalización Min-Max espacial de tu jefa
    pos_raw = torch.tensor(np.array(node_positions), dtype=torch.float)
    pos_norm = (pos_raw - POS_MIN) / (POS_MAX - POS_MIN) # Fórmula matemática: (X - Min) / (Max - Min)
    
    if len(edge_indices) > 0:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 13), dtype=torch.float) # [MODIFICADO] Ahora tiene dimensión 13
    
    # [MODIFICADO] Usamos 'pos_norm' en lugar de 'pos' (coordenadas brutas)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y_val, pos=pos_norm, pdb_id=pdb_id)
    
    # 8. FILTRADO DE NODOS HUÉRFANOS (ISLAS)
    G = to_networkx(data, to_undirected=True)
    connected_components = list(nx.connected_components(G))
    ligand_nodes = set(torch.where(data.x[:, -1] == 1)[0].numpy()) 
    valid_nodes = set()
    
    for comp in connected_components:
        if comp.intersection(ligand_nodes): valid_nodes.update(comp) 
    
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
            pos_2d = data.pos[:, :2].numpy() 
            
            node_colors = ['blue' if feats[-1] == 1 else 'red' for feats in data.x.numpy()]
            
            nx.draw(G_draw, pos_2d, with_labels=False, node_color=node_colors, 
                    node_size=60, edge_color='gray', width=0.8, alpha=0.6)
            plt.title(f"Grafo del Complejo {pdb_id}")
            plt.axis('off')
            plt.savefig(os.path.join(viz_dir, f"{pdb_id}_graph.png"), dpi=100, bbox_inches='tight')
            plt.close('all')  
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