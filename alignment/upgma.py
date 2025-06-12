def upgma_algorithm(dist_matrix, item_names):
    """Performs the UPGMA hierarchical clustering algorithm.

    Args:
        dist_matrix (numpy.ndarray): A symmetric 2D array representing the distance matrix.
        item_names (list of str): List of names corresponding to the items in the distance matrix.

    Returns:
        str: Newick string representation of the resulting UPGMA tree, or an error message if input is invalid.

    Raises:
        None: Returns error messages as strings instead of raising exceptions.
    """
    n = len(item_names)
    if dist_matrix.shape[0] != n:
        return "ERROR: Matrix and names mismatch."
    if n == 0:
        return "();"
    if n == 1:
        return f"({item_names[0]}:0.0);"

    clusters = {i: UPGMACluster(id_val=i, name=name) for i, name in enumerate(item_names)}
    dists = {frozenset({i, j}): dist_matrix[i, j] for i in range(n) for j in range(i + 1, n)}
    next_cluster_id = n

    while len(clusters) > 1:
        if not dists:
            return "ERROR: Distance map became empty unexpectedly."

        min_pair_ids = min(dists, key=dists.get)
        id1, id2 = min_pair_ids
        c1 = clusters[id1]
        c2 = clusters[id2]
        new_height = dists[min_pair_ids] / 2.0
        new_cluster = UPGMACluster(next_cluster_id, left=c1, right=c2, height=new_height)

        new_dists = {}
        for other_id, other_cluster in clusters.items():
            if other_id not in min_pair_ids:
                d1 = dists[frozenset({id1, other_id})]
                d2 = dists[frozenset({id2, other_id})]
                new_dists[frozenset({next_cluster_id, other_id})] = (
                        (d1 * c1.leaf_count + d2 * c2.leaf_count) / (c1.leaf_count + c2.leaf_count)
                )

        del clusters[id1]
        del clusters[id2]
        dists = {k: v for k, v in dists.items() if id1 not in k and id2 not in k}
        clusters[next_cluster_id] = new_cluster
        dists.update(new_dists)
        next_cluster_id += 1

    return list(clusters.values())[0].to_newick() + ";"


class UPGMACluster:
    """Represents a cluster in the UPGMA hierarchical clustering algorithm.

    Args:
        id_val (int): Unique identifier for the cluster.
        name (str, optional): Name of the cluster (for leaves).
        left (UPGMACluster, optional): Left child cluster.
        right (UPGMACluster, optional): Right child cluster.
        height (float, optional): Height of the cluster in the tree.

    Attributes:
        id (int): Unique identifier for the cluster.
        name (str): Name of the cluster.
        left (UPGMACluster): Left child cluster.
        right (UPGMACluster): Right child cluster.
        height (float): Height of the cluster.
        leaf_count (int): Number of leaves under this cluster.
    """

    def __init__(self, id_val, name=None, left=None, right=None, height=0.0):
        self.id = id_val
        self.name = name
        self.left = left
        self.right = right
        self.height = height
        self.leaf_count = 1 if not left else left.leaf_count + right.leaf_count

    def to_newick(self):
        """Converts the cluster and its children to Newick format.

        Returns:
            str: Newick string representation of the cluster.
        """
        if not self.left:
            return self.name
        return f"({self.left.to_newick()}:{self.height - self.left.height:.4f},{self.right.to_newick()}:{self.height - self.right.height:.4f})"
