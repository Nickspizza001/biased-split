import os
import tempfile
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image
from scipy.stats import gaussian_kde
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

try:
    from app.descriptors import compute_similarity_matrix, smiles_to_ecfp4
except ImportError:
    from descriptors import compute_similarity_matrix, smiles_to_ecfp4

# Assuming these are accessible in your environment based on the notebook imports
try:
    from biased_split import (
        smiles_to_ecfp4_bitvect,
        molecular_network_from_list,
        visualise_molnet_split,
    )
except ImportError:
    pass

UNASSIGNED_NODE = 0
TRAIN_NODE = 1
TEST_NODE = 2

# ==========================================
# Original Splitters (Activity Cliff, Scaffold, Random)
# ==========================================

def find_cliff_edges(similarity_matrix, activity_values, similarity_threshold, activity_threshold):
    n = len(activity_values)
    cliff_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if similarity_matrix[i, j] >= similarity_threshold:
                diff = abs(float(activity_values[i]) - float(activity_values[j]))
                if diff >= activity_threshold:
                    cliff_edges.append((i, j, diff))
    return cliff_edges

def compute_cliff_degrees(cliff_edges, n_molecules):
    degrees = np.zeros(n_molecules, dtype=int)
    for a, b, _ in cliff_edges:
        degrees[a] += 1
        degrees[b] += 1
    return degrees

def walk_cliff_edges(cliff_edges, cliff_degrees, n_molecules, n_cliff_test_target, rng):
    assignment = np.full(n_molecules, UNASSIGNED_NODE)
    placed = 0

    for a, b, _ in cliff_edges:
        if placed >= n_cliff_test_target:
            break
        sa, sb = assignment[a], assignment[b]

        if sa == UNASSIGNED_NODE and sb == UNASSIGNED_NODE:
            if cliff_degrees[a] > cliff_degrees[b]:
                tr, te = a, b
            elif cliff_degrees[b] > cliff_degrees[a]:
                tr, te = b, a
            else:
                tr, te = (a, b) if rng.random() < 0.5 else (b, a)
            assignment[tr] = TRAIN_NODE
            assignment[te] = TEST_NODE
            placed += 1
        elif sa == TRAIN_NODE and sb == UNASSIGNED_NODE:
            assignment[b] = TEST_NODE
            placed += 1
        elif sb == TRAIN_NODE and sa == UNASSIGNED_NODE:
            assignment[a] = TEST_NODE
            placed += 1
        elif sa == TEST_NODE and sb == UNASSIGNED_NODE:
            assignment[b] = TRAIN_NODE
        elif sb == TEST_NODE and sa == UNASSIGNED_NODE:
            assignment[a] = TRAIN_NODE

    return assignment

def evaluate_effective_bias(test_idx, train_idx, activity, act_thresh, sim_mat, sim_thresh):
    if len(test_idx) == 0 or len(train_idx) == 0:
        return 0.0
    sim_sub = sim_mat[test_idx[:, None], train_idx]
    act_diff = np.abs(activity[test_idx][:, None] - activity[train_idx])
    has_cliff = ((sim_sub >= sim_thresh) & (act_diff >= act_thresh)).any(axis=1)
    return float(has_cliff.mean())

class ActivityCliffSplitter:
    def __init__(self, similarity_threshold=0.7, activity_threshold=1.0, test_fraction=0.2):
        self.sim_thresh = similarity_threshold
        self.act_thresh = activity_threshold
        self.test_fraction = test_fraction

    def split(self, smiles_list, activity_values, intended_bias, random_seed=42):
        rng = np.random.default_rng(random_seed)
        n = len(smiles_list)
        target_test_size = int(self.test_fraction * n)
        n_cliff_test = int(intended_bias * target_test_size)

        fps = [smiles_to_ecfp4(s) for s in smiles_list]
        sim_mat = compute_similarity_matrix(fps)

        cliff_edges = find_cliff_edges(sim_mat, activity_values, self.sim_thresh, self.act_thresh)
        rng.shuffle(cliff_edges)
        degrees = compute_cliff_degrees(cliff_edges, n)

        assignment = walk_cliff_edges(cliff_edges, degrees, n, n_cliff_test, rng)

        unassigned = np.where(assignment == UNASSIGNED_NODE)[0]
        unassigned_non_cliff = unassigned[degrees[unassigned] == 0]
        unassigned_cliff = unassigned[degrees[unassigned] > 0]

        n_fill = target_test_size - int((assignment == TEST_NODE).sum())
        if n_fill > 0:
            if len(unassigned_non_cliff) >= n_fill:
                rand_test = rng.choice(unassigned_non_cliff, size=n_fill, replace=False)
            else:
                shortfall = n_fill - len(unassigned_non_cliff)
                topup = rng.choice(unassigned_cliff, size=min(shortfall, len(unassigned_cliff)), replace=False)
                rand_test = np.concatenate([unassigned_non_cliff, topup])
            assignment[rand_test] = TEST_NODE

        assignment[assignment == UNASSIGNED_NODE] = TRAIN_NODE
        train_idx = np.where(assignment == TRAIN_NODE)[0]
        test_idx = np.where(assignment == TEST_NODE)[0]

        effective_bias = evaluate_effective_bias(
            test_idx, train_idx, np.asarray(activity_values), self.act_thresh, sim_mat, self.sim_thresh
        )
        return train_idx, test_idx, effective_bias

class ScaffoldSplitter:
    def __init__(self, test_fraction=0.2):
        self.test_fraction = test_fraction

    def split(self, smiles_list, random_seed=42):
        scaffolds = {}
        for idx, smi in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smi)
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False) if mol else ""
            scaffolds.setdefault(scaffold, []).append(idx)

        rng = np.random.default_rng(random_seed)
        scaffold_groups = list(scaffolds.values())
        rng.shuffle(scaffold_groups)

        target_test_size = int(self.test_fraction * len(smiles_list))
        test_idx, train_idx = [], []

        for group in sorted(scaffold_groups, key=len, reverse=True):
            if len(test_idx) + len(group) <= target_test_size:
                test_idx.extend(group)
            else:
                train_idx.extend(group)

        return np.array(train_idx), np.array(test_idx), 0.0

class RandomSplitter:
    def __init__(self, test_fraction=0.2):
        self.test_fraction = test_fraction

    def split(self, smiles_list, random_seed=42):
        n = len(smiles_list)
        n_test = int(self.test_fraction * n)
        rng = np.random.default_rng(random_seed)
        shuffled = rng.permutation(n)
        return shuffled[n_test:], shuffled[:n_test], 0.0

# ==========================================
# KNN Failure Splitter
# ==========================================

def find_failure_n_edges(similarity_matrix, activity_values, similarity_threshold, activity_threshold, n_neighbors):
    n_molecules = len(activity_values)
    n_edges = []
    for molecule_index in range(n_molecules):
        similarities = similarity_matrix[molecule_index].copy()
        similarities[molecule_index] = -1.0
        qualifying = np.where(similarities >= similarity_threshold)[0]
        if len(qualifying) < n_neighbors:
            continue
        top_k = qualifying[np.argsort(similarities[qualifying])[::-1][:n_neighbors]]
        consensus = float(activity_values[top_k].mean())
        disagreement = abs(consensus - float(activity_values[molecule_index]))
        if disagreement >= activity_threshold:
            n_edges.append((int(molecule_index), tuple(int(n) for n in top_k)))
    return n_edges

def walk_failure_n_edges(failure_n_edges, n_molecules, n_failure_test_target):
    assignment = np.full(n_molecules, UNASSIGNED_NODE, dtype=np.int8)
    n_failures_placed = 0
    for molecule_index, neighbor_indices in failure_n_edges:
        if n_failures_placed >= n_failure_test_target:
            break
        if assignment[molecule_index] == TRAIN_NODE:
            continue
        if any(assignment[n] == TEST_NODE for n in neighbor_indices):
            continue
        assignment[molecule_index] = TEST_NODE
        for neighbor_index in neighbor_indices:
            assignment[neighbor_index] = TRAIN_NODE
        n_failures_placed += 1
    return assignment

def evaluate_knn_failure_question(test_indices, train_indices, activity_values, similarity_matrix, similarity_threshold, activity_threshold, n_neighbors):
    results = np.full(len(test_indices), np.nan, dtype=float)
    if len(test_indices) == 0 or len(train_indices) == 0:
        return results
    for position, test_idx in enumerate(test_indices):
        similarities_to_train = similarity_matrix[test_idx][train_indices]
        qualifying = np.where(similarities_to_train >= similarity_threshold)[0]
        if len(qualifying) < n_neighbors:
            continue
        top_k_positions = qualifying[np.argsort(similarities_to_train[qualifying])[::-1][:n_neighbors]]
        top_k_train_indices = train_indices[top_k_positions]
        consensus = float(activity_values[top_k_train_indices].mean())
        disagreement = abs(consensus - float(activity_values[test_idx]))
        results[position] = 1.0 if disagreement >= activity_threshold else 0.0
    return results

def effective_bias_from_question_results(question_results):
    if question_results.size == 0:
        return 0.0
    evaluable = question_results[~np.isnan(question_results)]
    if evaluable.size == 0:
        return 0.0
    return float(evaluable.mean())

class KNNFailureSplitter:
    def __init__(self, similarity_threshold, activity_threshold, n_neighbors, test_fraction=0.2):
        self.similarity_threshold = similarity_threshold
        self.activity_threshold = activity_threshold
        self.n_neighbors = n_neighbors
        self.test_fraction = test_fraction

    def split_for_intended_bias(self, smiless, activity_values, intended_bias, random_seed):
        if not (0.0 <= intended_bias <= 1.0):
            raise ValueError(f"intended_bias must be in [0, 1], got {intended_bias}")

        rng = np.random.default_rng(random_seed)
        n_molecules = len(smiless)
        target_test_size = int(self.test_fraction * n_molecules)
        n_failure_test_target = int(intended_bias * target_test_size)

        fps_bitvect = [smiles_to_ecfp4_bitvect(s) for s in smiless]
        similarity_matrix = compute_similarity_matrix(fps_bitvect)

        failure_n_edges = find_failure_n_edges(
            similarity_matrix, activity_values,
            self.similarity_threshold, self.activity_threshold, self.n_neighbors,
        )
        shuffled_order = rng.permutation(len(failure_n_edges))
        failure_n_edges = [failure_n_edges[i] for i in shuffled_order]

        assignment = walk_failure_n_edges(failure_n_edges, n_molecules, n_failure_test_target)

        candidate_set = {molecule_index for molecule_index, _ in failure_n_edges}
        is_candidate_mask = np.zeros(n_molecules, dtype=bool)
        if candidate_set:
            is_candidate_mask[list(candidate_set)] = True

        unassigned_indices = np.where(assignment == UNASSIGNED_NODE)[0]
        unassigned_non_candidate_indices = unassigned_indices[~is_candidate_mask[unassigned_indices]]
        unassigned_candidate_indices = unassigned_indices[is_candidate_mask[unassigned_indices]]

        n_random_fill = target_test_size - int((assignment == TEST_NODE).sum())
        if n_random_fill > 0:
            if len(unassigned_non_candidate_indices) >= n_random_fill:
                random_test_indices = rng.choice(unassigned_non_candidate_indices, size=n_random_fill, replace=False)
            else:
                shortfall = n_random_fill - len(unassigned_non_candidate_indices)
                candidate_topup_indices = rng.choice(
                    unassigned_candidate_indices,
                    size=min(shortfall, len(unassigned_candidate_indices)),
                    replace=False,
                )
                random_test_indices = np.concatenate([unassigned_non_candidate_indices, candidate_topup_indices])
            assignment[random_test_indices] = TEST_NODE

        assignment[assignment == UNASSIGNED_NODE] = TRAIN_NODE

        train_indices = np.where(assignment == TRAIN_NODE)[0]
        test_indices = np.where(assignment == TEST_NODE)[0]

        question_results = evaluate_knn_failure_question(
            test_indices, train_indices,
            np.asarray(activity_values, dtype=float),
            similarity_matrix, self.similarity_threshold, self.activity_threshold, self.n_neighbors,
        )
        effective_bias = effective_bias_from_question_results(question_results)

        return train_indices, test_indices, effective_bias

    def split(self, smiless, activity_values, intended_biases, n_repeats):
        for intended_bias in intended_biases:
            for repeat_index in range(n_repeats):
                train_indices, test_indices, effective_bias = self.split_for_intended_bias(
                    smiless=smiless, activity_values=activity_values,
                    intended_bias=intended_bias, random_seed=repeat_index,
                )
                yield train_indices, test_indices, effective_bias, intended_bias, repeat_index

    def visualise_splits(self, smiless, activity_values, intended_biases, n_repeats, output_path, duration=500):
        G = molecular_network_from_list(smiless, activity_values, self.similarity_threshold, self.activity_threshold)
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = []
            for frame_index, (train_idx, test_idx, effective_bias, intended_bias, _) in enumerate(
                self.split(smiless, activity_values, intended_biases, n_repeats)
            ):
                p = os.path.join(tmpdir, f"frame_{frame_index:04d}.png")
                visualise_molnet_split(G, train_idx, test_idx, effective_bias, intended_bias, filepath=p)
                paths.append(p)
            frames = [Image.open(p) for p in paths]
            frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=duration, loop=0)

# ==========================================
# Substructure Distance Splitter
# ==========================================

class SubstructureDistanceSplitter:
    def __init__(self, similarity_threshold, test_fraction=0.2):
        self.similarity_threshold = similarity_threshold
        self.test_fraction = test_fraction

    def split_for_intended_bias(
        self, smiless, tversky_matrix, activity_values, intended_bias, random_seed
    ):
        if not (0.0 <= intended_bias <= 1.0):
            raise ValueError(f"intended_bias must be in [0, 1], got {intended_bias}")

        rng = np.random.default_rng(random_seed)
        n_molecules = len(smiless)
        target_test_size = int(self.test_fraction * n_molecules)
        n_isolated_test_target = int(intended_bias * target_test_size)

        components = self.find_components(tversky_matrix, self.similarity_threshold)
        assignment = self.walk_components(components, n_molecules, n_isolated_test_target, rng)

        unassigned_indices = np.where(assignment == UNASSIGNED_NODE)[0]
        n_random_fill = target_test_size - int((assignment == TEST_NODE).sum())
        if n_random_fill > 0 and len(unassigned_indices) > 0:
            n_to_sample = min(n_random_fill, len(unassigned_indices))
            random_test_indices = rng.choice(unassigned_indices, size=n_to_sample, replace=False)
            assignment[random_test_indices] = TEST_NODE

        assignment[assignment == UNASSIGNED_NODE] = TRAIN_NODE

        train_indices = np.where(assignment == TRAIN_NODE)[0]
        test_indices = np.where(assignment == TEST_NODE)[0]

        question_results = self.evaluate_substructure_question(
            test_indices, train_indices, tversky_matrix, self.similarity_threshold
        )
        effective_bias = self.effective_bias_from_question_results(question_results)

        return train_indices, test_indices, effective_bias

    def split(self, smiless, activity_values, intended_biases, n_repeats):
        fps_bitvect = [smiles_to_ecfp4_bitvect(s) for s in smiless]
        tversky_matrix = compute_similarity_matrix(fps_bitvect, method="tversky")
        for intended_bias in intended_biases:
            for repeat_index in range(n_repeats):
                train_indices, test_indices, effective_bias = self.split_for_intended_bias(
                    smiless, tversky_matrix, activity_values, intended_bias, repeat_index
                )
                yield train_indices, test_indices, effective_bias, intended_bias, repeat_index

    @staticmethod
    def find_components(tversky_matrix, similarity_threshold):
        adj_matrix = np.triu(tversky_matrix, k=1)
        adj_matrix[adj_matrix < similarity_threshold] = 0
        similarity_graph = nx.from_numpy_array(adj_matrix)
        return sorted(nx.connected_components(similarity_graph), key=len, reverse=True)

    @staticmethod
    def walk_components(components, n_molecules, n_isolated_test_target, rng):
        assignment = np.full(n_molecules, UNASSIGNED_NODE, dtype=np.int8)
        remaining_budget = n_isolated_test_target
        unused_components = list(components)
        while True:
            fitting = [c for c in unused_components if len(c) <= remaining_budget]
            if not fitting:
                break
            max_size = max(len(c) for c in fitting)
            largest = [c for c in fitting if len(c) == max_size]
            chosen = largest[int(rng.integers(len(largest)))]
            for molecule_index in chosen:
                assignment[molecule_index] = TEST_NODE
            unused_components.remove(chosen)
            remaining_budget -= len(chosen)
        return assignment

    @staticmethod
    def evaluate_substructure_question(
        test_indices, train_indices, tversky_matrix, similarity_threshold
    ):
        if len(test_indices) == 0:
            return np.array([], dtype=float)
        if len(train_indices) == 0:
            return np.ones(len(test_indices), dtype=float)
        similarity_test_vs_train = tversky_matrix[np.ix_(test_indices, train_indices)]
        max_train_similarity = similarity_test_vs_train.max(axis=1)
        is_isolated = max_train_similarity < similarity_threshold
        return is_isolated.astype(float)

    @staticmethod
    def effective_bias_from_question_results(question_results):
        if question_results.size == 0:
            return 0.0
        return float(question_results.mean())

    @staticmethod
    def build_visualization_network(smiless, activity_values, tversky_matrix, similarity_threshold):
        adj_matrix = np.triu(tversky_matrix, k=1)
        adj_matrix[adj_matrix < similarity_threshold] = 0
        G = nx.from_numpy_array(adj_matrix)
        node_attrs = {
            n: {"smiles": smi, "activity": act}
            for n, (smi, act) in enumerate(zip(smiless, activity_values))
        }
        nx.set_node_attributes(G, node_attrs)
        G.graph["activity_label"] = "activity"
        G.graph["activity_threshold"] = np.inf
        G.graph["similarity_threshold"] = similarity_threshold
        G.graph["similarity_fp"] = "2048bit ECFP4"
        G.graph["similarity_distance"] = "tversky"
        return G

    def visualise_splits(
        self, smiless, activity_values, intended_biases, n_repeats, output_path, duration=500
    ):
        fps_bitvect = [smiles_to_ecfp4_bitvect(s) for s in smiless]
        tversky_matrix = compute_similarity_matrix(fps_bitvect, method="tversky")
        G = self.build_visualization_network(
            smiless, activity_values, tversky_matrix, self.similarity_threshold
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = []
            frame_index = 0
            for intended_bias in intended_biases:
                for repeat_index in range(n_repeats):
                    train_idx, test_idx, effective_bias = self.split_for_intended_bias(
                        smiless, tversky_matrix, activity_values, intended_bias, repeat_index
                    )
                    p = os.path.join(tmpdir, f"frame_{frame_index:04d}.png")
                    visualise_molnet_split(
                        G, train_idx, test_idx, effective_bias, intended_bias, filepath=p, cliff=False
                    )
                    paths.append(p)
                    frame_index += 1
            frames = [Image.open(p) for p in paths]
            frames[0].save(
                output_path, save_all=True, append_images=frames[1:], duration=duration, loop=0
            )

# ==========================================
# Proxy Sorted Splitter
# ==========================================

def visualise_proxy_split(
    proxy_values,
    train_idx,
    test_idx,
    ideal_range_min,
    ideal_range_max,
    effective_bias,
    intended_bias,
    proxy_label="proxy",
    x_range=None,
    filepath=None,
):
    TEST = (0.20, 0.40, 0.60)
    TRAIN = (0.5, 0.5, 0.5)
    IDEAL = (0.65, 0.15, 0.20)

    fig, ax = plt.subplots(figsize=(10, 5))

    train_values = proxy_values[train_idx]
    test_values = proxy_values[test_idx]

    if x_range is None:
        x_min, x_max = float(proxy_values.min()), float(proxy_values.max())
        pad = (x_max - x_min) * 0.05
        x_range = (x_min - pad, x_max + pad)

    x = np.linspace(x_range[0], x_range[1], 500)
    train_kde = gaussian_kde(train_values)
    test_kde = gaussian_kde(test_values)
    train_density = train_kde(x)
    test_density = test_kde(x)

    ax.axvspan(ideal_range_min, ideal_range_max, color=IDEAL, alpha=0.10, linewidth=0)
    ax.fill_between(x, train_density, color=TRAIN, alpha=0.35, linewidth=0)
    ax.fill_between(x, test_density, color=TEST, alpha=0.45, linewidth=0)
    ax.plot(x, train_density, color=TRAIN, linewidth=1)
    ax.plot(x, test_density, color=TEST, linewidth=1)

    ax.set_xlabel(proxy_label)
    ax.set_ylabel("density")
    ax.set_xlim(x_range)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles = [
        Line2D([0], [0], color=TRAIN, linewidth=2, label="train"),
        Line2D([0], [0], color=TEST, linewidth=2, label="test"),
        Patch(facecolor=IDEAL, alpha=0.30, label="ideal range"),
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        frameon=False,
        ncol=3,
        fontsize=9,
    )

    caption = (
        f"{len(proxy_values)} molecules ({len(train_idx)} train, {len(test_idx)} test). "
        f"ideal range [{ideal_range_min}, {ideal_range_max}]. "
        f"intended bias {intended_bias:.2f}, effective bias {effective_bias:.2f}"
    )
    fig.text(0.5, 0.00, caption, ha="center", fontsize=8, color="0.4")

    if filepath:
        plt.savefig(filepath, dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

class ProxySortedSplitter:
    def __init__(self, proxy_function, ideal_range_min, ideal_range_max, test_fraction=0.2):
        self.proxy_function = proxy_function
        self.ideal_range_min = ideal_range_min
        self.ideal_range_max = ideal_range_max
        self.test_fraction = test_fraction

    def split_for_intended_bias(
        self, smiless, proxy_values, activity_values, intended_bias, random_seed
    ):
        if not (0.0 <= intended_bias <= 1.0):
            raise ValueError(f"intended_bias must be in [0, 1], got {intended_bias}")

        rng = np.random.default_rng(random_seed)
        n_molecules = len(smiless)
        target_test_size = int(self.test_fraction * n_molecules)
        n_in_range_test_target = int(intended_bias * target_test_size)

        in_range_mask = self.find_in_range_mask(
            proxy_values, self.ideal_range_min, self.ideal_range_max
        )

        assignment = self.walk_in_range_molecules(
            in_range_mask, n_molecules, n_in_range_test_target, rng
        )

        unassigned_indices = np.where(assignment == UNASSIGNED_NODE)[0]
        unassigned_out_of_range_indices = unassigned_indices[~in_range_mask[unassigned_indices]]
        unassigned_in_range_indices = unassigned_indices[in_range_mask[unassigned_indices]]

        n_random_fill = target_test_size - int((assignment == TEST_NODE).sum())
        if n_random_fill > 0:
            if len(unassigned_out_of_range_indices) >= n_random_fill:
                random_test_indices = rng.choice(
                    unassigned_out_of_range_indices, size=n_random_fill, replace=False
                )
            else:
                shortfall = n_random_fill - len(unassigned_out_of_range_indices)
                in_range_topup_indices = rng.choice(
                    unassigned_in_range_indices,
                    size=min(shortfall, len(unassigned_in_range_indices)),
                    replace=False,
                )
                random_test_indices = np.concatenate(
                    [unassigned_out_of_range_indices, in_range_topup_indices]
                )
            assignment[random_test_indices] = TEST_NODE

        assignment[assignment == UNASSIGNED_NODE] = TRAIN_NODE

        train_indices = np.where(assignment == TRAIN_NODE)[0]
        test_indices = np.where(assignment == TEST_NODE)[0]

        question_results = self.evaluate_proxy_question(
            test_indices, proxy_values, self.ideal_range_min, self.ideal_range_max
        )
        effective_bias = self.effective_bias_from_question_results(question_results)

        return train_indices, test_indices, effective_bias

    def split(self, smiless, activity_values, intended_biases, n_repeats):
        proxy_values = np.array([self.proxy_function(s) for s in smiless], dtype=float)
        for intended_bias in intended_biases:
            for repeat_index in range(n_repeats):
                train_indices, test_indices, effective_bias = self.split_for_intended_bias(
                    smiless, proxy_values, activity_values, intended_bias, repeat_index
                )
                yield train_indices, test_indices, effective_bias, intended_bias, repeat_index

    @staticmethod
    def find_in_range_mask(proxy_values, ideal_range_min, ideal_range_max):
        return (proxy_values >= ideal_range_min) & (proxy_values <= ideal_range_max)

    @staticmethod
    def walk_in_range_molecules(in_range_mask, n_molecules, n_in_range_test_target, rng):
        assignment = np.full(n_molecules, UNASSIGNED_NODE, dtype=np.int8)
        in_range_indices = np.where(in_range_mask)[0]
        if n_in_range_test_target == 0 or len(in_range_indices) == 0:
            return assignment
        n_to_place = min(n_in_range_test_target, len(in_range_indices))
        selected = rng.choice(in_range_indices, size=n_to_place, replace=False)
        assignment[selected] = TEST_NODE
        return assignment

    @staticmethod
    def evaluate_proxy_question(test_indices, proxy_values, ideal_range_min, ideal_range_max):
        if len(test_indices) == 0:
            return np.array([], dtype=float)
        test_proxy = proxy_values[test_indices]
        in_range = (test_proxy >= ideal_range_min) & (test_proxy <= ideal_range_max)
        return in_range.astype(float)

    @staticmethod
    def effective_bias_from_question_results(question_results):
        if question_results.size == 0:
            return 0.0
        return float(question_results.mean())

    def visualise_splits(
        self,
        smiless,
        activity_values,
        intended_biases,
        n_repeats,
        output_path,
        duration=500,
        proxy_label="proxy",
    ):
        proxy_values = np.array([self.proxy_function(s) for s in smiless], dtype=float)
        x_min, x_max = float(proxy_values.min()), float(proxy_values.max())
        pad = (x_max - x_min) * 0.05
        x_range = (x_min - pad, x_max + pad)

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = []
            frame_index = 0
            for intended_bias in intended_biases:
                for repeat_index in range(n_repeats):
                    train_idx, test_idx, effective_bias = self.split_for_intended_bias(
                        smiless, proxy_values, activity_values, intended_bias, repeat_index
                    )
                    p = os.path.join(tmpdir, f"frame_{frame_index:04d}.png")
                    visualise_proxy_split(
                        proxy_values,
                        train_idx,
                        test_idx,
                        self.ideal_range_min,
                        self.ideal_range_max,
                        effective_bias,
                        intended_bias,
                        proxy_label=proxy_label,
                        x_range=x_range,
                        filepath=p,
                    )
                    paths.append(p)
                    frame_index += 1
            frames = [Image.open(p) for p in paths]
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:],
                duration=duration,
                loop=0,
            )