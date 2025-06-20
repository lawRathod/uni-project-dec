import os
import pickle

from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Geometry import Point3D
from rdkit import RDLogger
import imageio
import networkx as nx
import numpy as np
import rdkit.Chem
import wandb
import matplotlib.pyplot as plt


class MolecularVisualization:
    def __init__(self, remove_h, dataset_infos):
        self.remove_h = remove_h
        self.dataset_infos = dataset_infos

    def mol_from_graphs(self, node_list, adjacency_matrix):
        """
        Convert graphs to rdkit molecules
        node_list: the nodes of a batch of nodes (bs x n)
        adjacency_matrix: the adjacency_matrix of the molecule (bs x n x n)
        """
        # dictionary to map integer value to the char of atom
        atom_decoder = self.dataset_infos.atom_decoder

        # create empty editable mol object
        mol = Chem.RWMol()

        # add atoms to mol and keep track of index
        node_to_idx = {}
        for i in range(len(node_list)):
            if node_list[i] == -1:
                continue
            a = Chem.Atom(atom_decoder[int(node_list[i])])
            molIdx = mol.AddAtom(a)
            node_to_idx[i] = molIdx

        for ix, row in enumerate(adjacency_matrix):
            for iy, bond in enumerate(row):
                # only traverse half the symmetric matrix
                if iy <= ix:
                    continue
                if bond == 1:
                    bond_type = Chem.rdchem.BondType.SINGLE
                elif bond == 2:
                    bond_type = Chem.rdchem.BondType.DOUBLE
                elif bond == 3:
                    bond_type = Chem.rdchem.BondType.TRIPLE
                elif bond == 4:
                    bond_type = Chem.rdchem.BondType.AROMATIC
                else:
                    continue
                mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)

        try:
            mol = mol.GetMol()
        except rdkit.Chem.KekulizeException:
            print("Can't kekulize molecule")
            mol = None
        return mol

    def visualize(self, path: str, molecules: list, num_molecules_to_visualize: int, log='graph'):
        # define path to save figures
        if not os.path.exists(path):
            os.makedirs(path)

        # visualize the final molecules
        print(f"Visualizing {num_molecules_to_visualize} of {len(molecules)}")
        if num_molecules_to_visualize > len(molecules):
            print(f"Shortening to {len(molecules)}")
            num_molecules_to_visualize = len(molecules)

        for i in range(num_molecules_to_visualize):
            file_path = os.path.join(path, 'molecule_{}.png'.format(i))
            mol = self.mol_from_graphs(molecules[i][0], molecules[i][1])
            try:
                Draw.MolToFile(mol, file_path)
                if wandb.run and log is not None:
                    print(f"Saving {file_path} to wandb")
                    wandb.log({log: wandb.Image(file_path)}, commit=True)
            except rdkit.Chem.KekulizeException:
                print("Can't kekulize molecule")


    def visualize_chain(self, path, nodes_list, adjacency_matrix):
        RDLogger.DisableLog('rdApp.*')
        # convert graphs to the rdkit molecules
        mols = [self.mol_from_graphs(nodes_list[i], adjacency_matrix[i]) for i in range(len(nodes_list))]

        # find the coordinates of atoms in the final molecule
        final_molecule = mols[-1]
        AllChem.Compute2DCoords(final_molecule)

        coords = []
        for i, atom in enumerate(final_molecule.GetAtoms()):
            positions = final_molecule.GetConformer().GetAtomPosition(i)
            coords.append((positions.x, positions.y, positions.z))

        # align all the molecules
        for i, mol in enumerate(mols):
            AllChem.Compute2DCoords(mol)
            conf = mol.GetConformer()
            for j, atom in enumerate(mol.GetAtoms()):
                x, y, z = coords[j]
                conf.SetAtomPosition(j, Point3D(x, y, z))

        # draw gif
        save_paths = []
        num_frams = nodes_list.shape[0]

        for frame in range(num_frams):
            file_name = os.path.join(path, 'fram_{}.png'.format(frame))
            Draw.MolToFile(mols[frame], file_name, size=(300, 300), legend=f"Frame {frame}")
            save_paths.append(file_name)

        imgs = [imageio.imread(fn) for fn in save_paths]
        gif_path = os.path.join(os.path.dirname(path), '{}.gif'.format(path.split('/')[-1]))
        imgs.extend([imgs[-1]] * 10)
        imageio.mimsave(gif_path, imgs, subrectangles=True, duration=20)

        if wandb.run:
            print(f"Saving {gif_path} to wandb")
            wandb.log({"chain": wandb.Video(gif_path, fps=5, format="gif")}, commit=True)

        # draw grid image
        try:
            img = Draw.MolsToGridImage(mols, molsPerRow=10, subImgSize=(200, 200))
            img.save(os.path.join(path, '{}_grid_image.png'.format(path.split('/')[-1])))
        except Chem.rdchem.KekulizeException:
            print("Can't kekulize molecule")
        return mols


class NonMolecularVisualization:
    def __init__(self, seed: int) -> None:
        self.seed = seed

    def to_networkx(self, node_list, adjacency_matrix):
        """
        Convert graphs to networkx graphs
        node_list: the nodes of a batch of nodes (bs x n)
        adjacency_matrix: the adjacency_matrix of the molecule (bs x n x n)
        """
        graph = nx.Graph()

        for i in range(len(node_list)):
            if node_list[i] == -1:
                continue
            graph.add_node(i, number=i, symbol=node_list[i], color_val=node_list[i])

        rows, cols = np.where(adjacency_matrix >= 1)
        edges = zip(rows.tolist(), cols.tolist())
        for edge in edges:
            edge_type = adjacency_matrix[edge[0]][edge[1]]
            graph.add_edge(edge[0], edge[1], color=float(edge_type), weight=3 * edge_type)

        return graph

    def visualize_non_molecule(self, graph, pos, path, iterations=100, node_size=100, largest_component=False):
        if largest_component:
            CGs = [graph.subgraph(c) for c in nx.connected_components(graph)]
            CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)
            graph = CGs[0]

        # Plot the graph structure with colors
        if pos is None:
            pos = nx.spring_layout(graph, iterations=iterations)

        # Set node colors based on the eigenvectors
        w, U = np.linalg.eigh(nx.normalized_laplacian_matrix(graph).toarray())
        vmin, vmax = np.min(U[:, 1]), np.max(U[:, 1])
        m = max(np.abs(vmin), vmax)
        vmin, vmax = -m, m

        plt.figure()
        nx.draw(graph, pos, font_size=5, node_size=node_size, with_labels=False, node_color=U[:, 1],
                cmap=plt.cm.coolwarm, vmin=vmin, vmax=vmax, edge_color='grey')

        plt.tight_layout()
        plt.savefig(path)
        plt.close("all")

    def visualize(self, path: str, graphs: list, num_graphs_to_visualize: int, log='graph'):
        # define path to save figures
        if not os.path.exists(path):
            os.makedirs(path)

        # visualize the final molecules
        for i in range(num_graphs_to_visualize):
            file_path = os.path.join(path, 'graph_{}.png'.format(i))
            graph = self.to_networkx(graphs[i][0], graphs[i][1])
            self.visualize_non_molecule(graph=graph, pos=None, path=file_path)
            im = plt.imread(file_path)
            if wandb.run and log is not None:
                wandb.log({log: [wandb.Image(im, caption=file_path)]})

    def visualize_chain(self, path, nodes_list, adjacency_matrix):
        # convert graphs to networkx
        graphs = [self.to_networkx(nodes_list[i], adjacency_matrix[i]) for i in range(len(nodes_list))]
        # find the coordinates of atoms in the final molecule
        final_graph = graphs[-1]
        final_pos = nx.spring_layout(final_graph, seed=self.seed)

        # draw gif
        save_paths = []
        num_frams = nodes_list.shape[0]

        for frame in range(num_frams):
            file_name = os.path.join(path, 'fram_{}.png'.format(frame))
            self.visualize_non_molecule(graph=graphs[frame], pos=final_pos, path=file_name)
            save_paths.append(file_name)

        imgs = [imageio.imread(fn) for fn in save_paths]
        gif_path = os.path.join(os.path.dirname(path), '{}.gif'.format(path.split('/')[-1]))
        imgs.extend([imgs[-1]] * 10)
        imageio.mimsave(gif_path, imgs, subrectangles=True, duration=20)
        if wandb.run:
            wandb.log({'chain': [wandb.Video(gif_path, caption=gif_path, format="gif")]})

from typing import Optional, Union, cast

import pandas as pd
from matplotlib import cm

from src.datasets.abstract_dataset import TableGraphDatasetInfos

class NonMolecularNxVisualization(NonMolecularVisualization):
    def __init__(self, ds_name: str) -> None:
        self.ds_name = ds_name
        self.curr_epoch_nr = -1

    def visualize(self, path: str, graphs: list, num_graphs_to_visualize: int):
        abs_path = os.path.abspath(path)
        dname_parent = os.path.basename(abs_path)
        index_underscore = dname_parent.index('_')
        epoch_nr = int(dname_parent[dname_parent.index('h') + 1:index_underscore])
        dname_write = os.path.join(os.path.dirname(abs_path), f'epoch{epoch_nr}/')

        if epoch_nr != self.curr_epoch_nr:
            self.curr_epoch_nr = epoch_nr

        # define path to save figures
        if not os.path.exists(dname_write) and num_graphs_to_visualize:
            os.makedirs(dname_write)

        # visualize the final molecules
        for i in range(num_graphs_to_visualize):
            file_path = os.path.join(dname_write, f'digress-{self.ds_name}.pickle')
            graph = self.to_networkx(graphs[i][0].numpy(), graphs[i][1].numpy())

            for _, nd in graph.nodes(data=True):
                del nd["number"]
                del nd["color_val"]

            for _, _, ed in graph.edges(data=True):
                ed.clear()

            with open(file_path, 'ab') as f:
                pickle.dump(graph, f)


class TableGraphVisualization:
    def __init__(self, dataset_infos: TableGraphDatasetInfos, seed: int) -> None:
        self.dataset_infos = dataset_infos
        self.seed = seed

    def _visualize_table_graph(self, path: str, table: pd.DataFrame, graph: nx.Graph,
                               pos: Optional[dict]=None, iterations=100, node_size=100,
                               table_entries_vis=10, table_last_cols_vis=10) -> None:
        fig, axs = plt.subplots(2)

        pos_ = pos or nx.spring_layout(graph, iterations=iterations, seed=self.seed)

        # Set node colors based on the eigenvectors
        w, U = np.linalg.eigh(nx.normalized_laplacian_matrix(graph).toarray())
        vmin, vmax = np.min(U[:, 1]), np.max(U[:, 1])
        m = max(np.abs(vmin), vmax)
        vmin, vmax = -m, m

        nx.draw(graph, pos_, font_size=8.5, node_size=node_size, with_labels=True, node_color=U[:, 1],
                cmap=cm.get_cmap("coolwarm"), vmin=vmin, vmax=vmax, edge_color='grey', ax=axs[0])

        axs[1].axis('off')
        table_portion = table.head(table_entries_vis)
        table_portion = table_portion[table_portion.columns[-table_last_cols_vis:]]
        tab = pd.plotting.table(axs[1], table_portion.astype('string'),
                          loc='center', cellLoc='center')
        tab.auto_set_font_size(False)
        tab.set_fontsize(7.5)

        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)

    def visualize(
            self, path: str,
            x_e_list: list[tuple[Union[np.ndarray, pd.DataFrame], np.ndarray]],
            num_graphs_to_visualize: int) -> None:
        # define path to save figures
        if not os.path.exists(path):
            os.makedirs(path)

        # visualize the final data
        for i in range(num_graphs_to_visualize):
            file_path = os.path.join(path, 'graph_{}.png'.format(i))
            x, e_tens = x_e_list[i]
            self._visualize_table_graph(
                path=file_path,
                table=self.dataset_infos.restore_table(x),
                graph=nx.from_numpy_array(e_tens))
            im = plt.imread(file_path)

            if wandb.run:
                wandb.log({'graph-table': [wandb.Image(im, caption=file_path)]})

    def visualize_chain(
            self, path: str,
            nodes_list: Union[np.ndarray, list[pd.DataFrame]], adjacency_matrix: np.ndarray,
            fps=15, **kwargs) -> None:
        save_paths = []

        tables_pd: tuple[pd.DataFrame]
        graphs_nx: tuple[nx.Graph]

        tables_pd, graphs_nx = zip(*[
            (self.dataset_infos.restore_table(x), nx.from_numpy_array(e))
            for x, e in zip(nodes_list, adjacency_matrix)
        ])
        final_graph = graphs_nx[-1]
        final_pos = cast(dict, nx.spring_layout(final_graph, seed=self.seed))

        # draw gif
        for frame in range(len(nodes_list)):
            file_name = os.path.join(path, 'fram_{}.png'.format(frame))
            table, graph = tables_pd[frame], graphs_nx[frame]
            self._visualize_table_graph(path=file_name, table=table, graph=graph, pos=final_pos)
            save_paths.append(file_name)

        imgs = [imageio.imread(fn) for fn in save_paths]
        gif_path = os.path.join(os.path.dirname(path), '{}.gif'.format(path.split('/')[-1]))
        imgs.extend([imgs[-1]] * 10)
        imageio.mimsave(gif_path, cast(list, imgs), subrectangles=True, duration=1000. / fps)

        if wandb.run:
            wandb.log({'chain': [wandb.Video(gif_path, caption=gif_path, format="gif")]})


class TableGraphCompressedVisualization:
    def __init__(self, dataset_infos: TableGraphDatasetInfos, seed:int) -> None:
        self.dataset_infos = dataset_infos
        self.seed = seed

    def _visualize_table_graph(self, path: str, table: pd.DataFrame, graph: nx.Graph,
                               latent: np.ndarray, graph_coarse: nx.Graph,
                               pos: Optional[dict]=None, pos_coarse: Optional[dict] = None,
                               iterations=1000, node_size=100,
                               table_entries_vis=10, table_last_cols_vis=10) -> None:
        fig, axs = plt.subplots(nrows=2, ncols=2, width_ratios=[25, 75])

        pos_ = pos or nx.spring_layout(graph, iterations=iterations, seed=self.seed)
        pos_coarse_ = pos_coarse or nx.spring_layout(graph_coarse, iterations=iterations, seed=self.seed)

        # Set node colors based on the eigenvectors
        w, U = np.linalg.eigh(nx.normalized_laplacian_matrix(graph).toarray())
        vmin, vmax = np.min(U[:, 1]), np.max(U[:, 1])
        m = max(np.abs(vmin), vmax)
        vmin, vmax = -m, m

        nx.draw(graph, pos_, font_size=8.5, node_size=node_size, with_labels=True, node_color=U[:, 1],
                cmap=cm.get_cmap("coolwarm"), vmin=vmin, vmax=vmax, edge_color='grey', ax=axs[0][1])

        nx.draw(graph_coarse, pos_coarse_, node_size=node_size // 4, node_color='black',
                edge_color='grey', ax=axs[0][0])

        axs[1][1].axis('off')
        table_portion = table.head(table_entries_vis)
        table_portion = table_portion[table_portion.columns[-table_last_cols_vis:]]
        tab = pd.plotting.table(axs[1][1], table_portion.astype('string'),
                          loc='center', cellLoc='center')
        tab.auto_set_font_size(False)
        tab.set_fontsize(5.5)

        axs[1][0].imshow(latent)
        axs[1][0].set_xticks([])
        axs[1][0].set_yticks([])

        fig.tight_layout()
        fig.savefig(path, dpi=300)
        plt.close(fig)

    def visualize(
            self, path: str,
            x_e_list: list[tuple[Union[np.ndarray, pd.DataFrame], np.ndarray]],
            x_e_list_comp: list[tuple[np.ndarray, np.ndarray]],
            num_graphs_to_visualize: int) -> None:
        # define path to save figures
        if not os.path.exists(path):
            os.makedirs(path)

        # visualize the final data
        for i in range(num_graphs_to_visualize):
            file_path = os.path.join(path, 'graph_{}.png'.format(i))
            x, e_tens = x_e_list[i]
            x_comp, e_tens_comp = x_e_list_comp[i]
            self._visualize_table_graph(
                path=file_path,
                table=self.dataset_infos.restore_table(x),
                graph=nx.from_numpy_array(e_tens),
                latent=x_comp,
                graph_coarse=nx.from_numpy_array(e_tens_comp))
            im = plt.imread(file_path)

            if wandb.run:
                wandb.log({'graph-table': [wandb.Image(im, caption=file_path)]})

    def visualize_chain(
            self, path: str,
            nodes_list: Union[np.ndarray, list[pd.DataFrame]], adjacency_matrix: np.ndarray,
            X_chain_latent: np.ndarray, E_chain_coarse: np.ndarray,
            fps=15) -> None:
        save_paths = []

        tables_pd: tuple[pd.DataFrame]
        graphs_nx: tuple[nx.Graph]

        tables_pd, graphs_nx = zip(*[
            (self.dataset_infos.restore_table(x), nx.from_numpy_array(e))
            for x, e in zip(nodes_list, adjacency_matrix)
        ])
        final_graph = graphs_nx[-1]
        final_pos = cast(dict, nx.spring_layout(final_graph, seed=self.seed))
        latents, graphs_nx_coarse = zip(*[
            (x, nx.from_numpy_array(e))
            for x, e in zip(X_chain_latent, E_chain_coarse)
        ])
        final_graph_coarse = graphs_nx_coarse[-1]
        final_pos_coarse = cast(dict, nx.spring_layout(final_graph_coarse, seed=self.seed))

        # draw gif
        for frame in range(len(nodes_list)):
            file_name = os.path.join(path, 'fram_{}.png'.format(frame))
            table, graph = tables_pd[frame], graphs_nx[frame]
            latent, graph_coarse = latents[frame], graphs_nx_coarse[frame]
            self._visualize_table_graph(
                path=file_name, table=table, graph=graph,
                latent=latent, graph_coarse=graph_coarse,
                pos=final_pos, pos_coarse=final_pos_coarse)
            save_paths.append(file_name)

        imgs = [imageio.imread(fn) for fn in save_paths]
        gif_path = os.path.join(os.path.dirname(path), '{}.gif'.format(path.split('/')[-1]))
        imgs.extend([imgs[-1]] * 10)
        imageio.mimsave(gif_path, cast(list, imgs), subrectangles=True, duration=1000. / fps)

        if wandb.run:
            wandb.log({'chain': [wandb.Video(gif_path, caption=gif_path, format="gif")]})
