import os

import dgl
from dgl.data import DGLDataset
from dgl.data.utils import download, extract_archive, makedirs, save_info, load_info
import dgl.function as fn
from dgl import save_graphs, load_graphs

from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import numpy as np

import Bio.PDB
from Bio.PDB.SASA import ShrakeRupley

from tqdm.notebook import tqdm

from numpy.random import RandomState
rng = RandomState(0)

from utils import aa_to_int

def calc_dihedral(v1, v2, v3, v4):
    u1 = v2 - v1
    u2 = v3 - v2
    u3 = v4 - v3

    a = torch.cross(u1, u2, dim=-1)
    b = torch.cross(u2, u3, dim=-1)
    c = torch.norm(a, dim=-1) * torch.norm(b, dim=-1)
    
    cos = torch.einsum('ij,ij->i', a, b) / c
    sin = torch.norm(u2, dim=-1) * torch.einsum('ij,ij->i', u1, b) / c

    return cos, sin

L = torch.tensor([1, 2, 5, 10, 30])

def mean_forces(nodes):
    
    w = torch.exp(-torch.linalg.norm(nodes.mailbox['Xv-Xu'], dim=-1).unsqueeze(-1) / L) # (N, k, L)
    w /= torch.sum(w, dim=1).unsqueeze(1)

    rho = torch.linalg.norm(torch.sum(w.unsqueeze(-1) * nodes.mailbox['Xv-Xu'].unsqueeze(-2), dim=1), dim=-1)
    rho /= torch.sum(w * torch.linalg.norm(nodes.mailbox['Xv-Xu'], dim=-1).unsqueeze(-1), dim=1)

    return {'rho': rho}

sigma = 2 * (1.5 ** torch.arange(0, 15)) ** 2

def gaussian_basis(edges):
    return {'Erbf': torch.exp(-torch.linalg.norm(edges.data['Xv-Xu'], dim=-1).unsqueeze(-1) ** 2 / sigma)}

def sequential_pos(edges):
    src, dst, _ = edges.edges()
    d = torch.clamp(torch.abs(src - dst), max=64)
    one_hot = F.one_hot(d, num_classes=65)
    return {'d': one_hot}

def local_coords(edges):
    return {
        'p': torch.einsum('ijk,ik->ij', edges.src['orientation'], edges.data['Xv-Xu']),
        'qkt': torch.einsum('ijk,isk->ijs', edges.src['orientation'], edges.dst['orientation'])
    }

def contact_fn(edges):
    return {'contact': torch.linalg.norm(edges.data['Xv-Xu'], dim=-1) < 8}

class CATH_S40_Dataset(DGLDataset):
    """CATH S40 Dataset

    Parameters
    ----------
    url : str
        URL to download the raw dataset
    raw_dir : str
        Specifying the directory that will store the
        downloaded data or the directory that
        already stores the input data.
        Default: ~/.dgl/
    save_dir : str
        Directory to save the processed dataset.
        Default: the value of `raw_dir`
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose : bool
        Whether to print out progress information
    """

    _url = "http://download.cathdb.info/cath/releases/latest-release/non-redundant-data-sets/cath-dataset-nonredundant-S40.pdb.tgz"

    def __init__(self,
                 raw_dir=None,
                 save_dir=None,
                 force_reload=False,
                 verbose=False):
        super(CATH_S40_Dataset, self).__init__(name='CATH-S40',
                                        url=self._url,
                                        raw_dir=raw_dir,
                                        save_dir=save_dir,
                                        force_reload=force_reload,
                                        verbose=verbose)

    def download(self):
        if os.path.exists(self.raw_path):
            return
        tgz_file_path = os.path.join(self.raw_dir, self.name + '.tgz')
        download(self.url, path=tgz_file_path)
        extract_archive(tgz_file_path, self.raw_path)

    def process(self):
        # process raw data to graphs, labels, splitting masks

        self.graphs = []
        self.labels = []
        self.bad_files = []

        pdb_folder = os.path.join(self.raw_path, 'dompdb')

        for pdb_file in tqdm(os.listdir(pdb_folder)):
            try:
                file_path = os.path.join(pdb_folder, pdb_file)
                pdbparser = Bio.PDB.PDBParser(QUIET=True)
                struct = pdbparser.get_structure(pdb_file[:4], file_path)

                residues = list(struct.get_residues())

                # Remove termini that are missing atoms
                if len(list(residues[0].get_atoms())) < 3:
                    residues = residues[1:]
                if len(list(residues[-1].get_atoms())) < 3:
                    residues = residues[:-1]

                residue_atoms = [list(res.get_atoms()) for res in residues]

                N_coords  = torch.tensor(np.array([atoms[0].get_coord() for atoms in residue_atoms]))
                CA_coords = torch.tensor(np.array([atoms[1].get_coord() for atoms in residue_atoms]))
                C_coords  = torch.tensor(np.array([atoms[2].get_coord() for atoms in residue_atoms]))
                
                g = dgl.knn_graph(CA_coords[1:-1], k=6)
                g = dgl.reorder_graph(g)
                g.ndata['X'] = CA_coords[1:-1]
                bfactor = \
                    torch.tensor(np.array([np.mean([atom.get_bfactor() for atom in atoms]) 
                            for atoms in residue_atoms]))[1:-1]
                g.ndata['bfactor'] = bfactor
                
                g.apply_edges(fn.v_sub_u('X', 'X', 'Xv-Xu'))
                g.apply_edges(contact_fn)

                sr = ShrakeRupley()
                sr.compute(struct, level='R')
                sasa = torch.tensor([res.sasa for res in residues])[1:-1]
                g.ndata['sasa'] = sasa

                u = N_coords - CA_coords
                t = C_coords - CA_coords
                n = torch.cross(u, t)
                n /= torch.linalg.norm(n, dim=-1, keepdim=True)
                v = torch.cross(n, u)
                g.ndata['orientation'] = torch.hstack((u[1:-1], v[1:-1], n[1:-1])).reshape(-1, 3, 3)

                g.apply_edges(local_coords)
                
                angles = torch.empty((CA_coords.shape[0]-2, 6)) 
                cosw, sinw     = calc_dihedral(CA_coords[:-2],  C_coords [:-2],  N_coords [1:-1], CA_coords[1:-1])
                cosphi, sinphi = calc_dihedral(C_coords [:-2],  N_coords [1:-1], CA_coords[1:-1], C_coords [1:-1])
                cospsi, sinpsi = calc_dihedral(N_coords [1:-1],  CA_coords[1:-1], C_coords[1:-1],  N_coords[2:])
                angles[:, 0] = cosw
                angles[:, 1] = sinw
                angles[:, 2] = cosphi
                angles[:, 3] = sinphi
                angles[:, 4] = cospsi
                angles[:, 5] = sinpsi
                g.ndata['angles'] = angles

                aa = torch.zeros((CA_coords.shape[0]-2, 20))
                for i, res in enumerate(residues[1:-1]):
                    aa[i, aa_to_int(res.get_resname())] = 1
                g.ndata['Xaa'] = aa

                g.apply_edges(gaussian_basis)
                g.update_all(fn.copy_e('Xv-Xu', 'Xv-Xu'), mean_forces)
                g.apply_edges(sequential_pos)

                # Concatenate all node features
                g.ndata['h'] = torch.hstack([t.reshape(g.num_nodes(), -1) for t in [g.ndata['Xaa'], g.ndata['bfactor'], g.ndata['sasa'], g.ndata['rho'], g.ndata['angles']]])                
                del g.ndata['Xaa']
                del g.ndata['bfactor']
                del g.ndata['sasa']
                del g.ndata['rho']
                del g.ndata['angles']
                del g.ndata['orientation']

                # Concatenate all edge features
                g.edata['e'] = torch.hstack([t.reshape(g.num_edges(), -1) for t in [g.edata['contact'], g.edata['p'], g.edata['qkt'], g.edata['Erbf'], g.edata['d']]])
                del g.edata['contact']
                del g.edata['p']
                del g.edata['qkt']
                del g.edata['Erbf']
                del g.edata['d']
                del g.edata['Xv-Xu']

                self.graphs.append(g)

            except Exception as e:
                print("Exception parsing: ", file_path)
                print(e)
                self.bad_files.append(file_path)

    def __getitem__(self, idx):
        # get one example by index
        return self.graphs[idx]

    def __len__(self):
        # number of data examples
        return len(self.graphs)

    def save(self):
        # save graphs
        graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
        save_graphs(graph_path, self.graphs)
        # save other information in python dict
        info_path = os.path.join(self.save_path, 'info.pkl')
        save_info(info_path, {'bad_files': self.bad_files})

    def load(self):
        # load processed data from directory `self.save_path`
        graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
        self.graphs, _ = load_graphs(graph_path)
        info_path = os.path.join(self.save_path, 'info.pkl')
        self.bad_files = load_info(info_path)['bad_files']
        
    def has_cache(self):
        # check whether there are processed data in `self.save_path`
        graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
        info_path = os.path.join(self.save_path, 'info.pkl')
        return os.path.exists(graph_path) and os.path.exists(info_path)
    