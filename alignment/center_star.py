import itertools

import numpy as np
from PyQt5.QtWidgets import QMessageBox

from .needleman_wunsch import needleman_wunsch


def calculate_all_pairwise_alignments(
        sequences: list[tuple[str, str]], match: int, mismatch: int, gap: int
) -> tuple[int, np.ndarray, dict]:
    """Performs all-vs-all pairwise alignments, finds the center sequence, and returns all results for reuse.

    Args:
        sequences (list[tuple[str, str]]): List of (id, sequence) tuples.
        match (int): Match score.
        mismatch (int): Mismatch penalty.
        gap (int): Gap penalty.

    Returns:
        tuple: A tuple containing:
            - center_idx (int): The index of the center sequence.
            - pairwise_scores (np.ndarray): The matrix of all pairwise scores.
            - all_alignments (dict): A dictionary where keys are (i, j) tuples and
              values are the (aligned_i, aligned_j) sequence strings.
    """
    n = len(sequences)
    if n == 0:
        return -1, np.array([]), {}

    pairwise_scores = np.zeros((n, n))
    all_alignments = {}
    for i in range(n):
        for j in range(i + 1, n):
            score, aligned_i, aligned_j = needleman_wunsch(
                sequences[i][1], sequences[j][1], match, mismatch, gap
            )
            pairwise_scores[i, j] = pairwise_scores[j, i] = score
            all_alignments[(i, j)] = (aligned_i, aligned_j)
            all_alignments[(j, i)] = (aligned_j, aligned_i)  # both directions
    total_scores = np.sum(pairwise_scores, axis=1)
    center_idx = np.argmax(total_scores) if n > 0 else -1
    return center_idx, pairwise_scores, all_alignments


def build_msa(
        center_idx: int, all_sequences: list[tuple[str, str]], all_alignments: dict
) -> list[str]:
    """Builds a multiple sequence alignment (MSA) using the center star method.

    Args:
        center_idx (int): Index of the center sequence.
        all_sequences (list[tuple[str, str]]): List of (id, sequence) tuples.
        all_alignments (dict): Dictionary of pairwise alignments.

    Returns:
        list[str]: List of aligned sequences in the original order.
    """
    if center_idx == -1 or not all_sequences:
        return [""] * len(all_sequences)

    center_id, center_seq = all_sequences[center_idx]
    n_seqs = len(all_sequences)

    # Determine the center profile by finding the maximum number of gaps needed at each position
    gaps_needed_at_pos = [0] * (len(center_seq) + 1)
    for i in range(n_seqs):
        if i == center_idx:
            continue
        aligned_center, _ = all_alignments[(center_idx, i)]
        center_char_idx = 0
        current_gap_run = 0
        for char in aligned_center:
            if char == "-":
                current_gap_run += 1
            else:
                gaps_needed_at_pos[center_char_idx] = max(
                    gaps_needed_at_pos[center_char_idx], current_gap_run
                )
                current_gap_run = 0
                center_char_idx += 1
        gaps_needed_at_pos[center_char_idx] = max(
            gaps_needed_at_pos[center_char_idx], current_gap_run
        )

    # Build the final MSA using the center profile
    final_msa_lists = {}
    master_center_profile = []
    for i in range(len(center_seq)):
        master_center_profile.extend(["-"] * gaps_needed_at_pos[i])
        master_center_profile.append(center_seq[i])
    master_center_profile.extend(["-"] * gaps_needed_at_pos[len(center_seq)])
    final_msa_lists[center_id] = master_center_profile

    for i in range(n_seqs):
        if i == center_idx:
            continue
        other_id, _ = all_sequences[i]
        aligned_center_for_pair, aligned_other_for_pair = all_alignments[(center_idx, i)]
        new_row = []
        master_ptr = 0
        pair_ptr = 0
        while master_ptr < len(master_center_profile):
            if pair_ptr >= len(aligned_center_for_pair):
                new_row.append("-")
                master_ptr += 1
                continue
            master_char = master_center_profile[master_ptr]
            pair_center_char = aligned_center_for_pair[pair_ptr]
            if master_char == "-" and pair_center_char != "-":
                new_row.append("-")
                master_ptr += 1
            else:
                new_row.append(aligned_other_for_pair[pair_ptr])
                master_ptr += 1
                pair_ptr += 1
        final_msa_lists[other_id] = new_row

    all_ids_original_order = [seq_id for seq_id, _ in all_sequences]
    final_msa_strings = [
        "".join(final_msa_lists[seq_id]) for seq_id in all_ids_original_order
    ]
    return final_msa_strings


def reduce_msa(full_msa: list[str]) -> list[str]:
    """Removes columns containing gaps from a multiple sequence alignment.

    Args:
        full_msa (list[str]): List of aligned sequences.

    Returns:
        list[str]: List of reduced sequences with gap columns removed.
    """
    if not full_msa:
        return []
    n = len(full_msa)
    aln_len = len(full_msa[0])
    reduced_seqs = [[] for _ in range(n)]
    for i in range(aln_len):
        column = [full_msa[j][i] for j in range(n)]
        if "-" not in column:
            for j in range(n):
                reduced_seqs[j].append(column[j])
    return ["".join(seq) for seq in reduced_seqs]


def calculate_distance_matrix_from_msa(msa: list[str]) -> np.ndarray | None:
    """Calculates a distance matrix from a multiple sequence alignment.

    Args:
        msa (list[str]): List of aligned sequences.

    Returns:
        np.ndarray | None: Distance matrix, or None if input is empty.
    """
    if not msa or not msa[0]:
        QMessageBox.warning(
            None, "Distance Error", "Cannot calculate distance from empty alignment."
        )
        return None
    n, aln_len = len(msa), len(msa[0])
    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            diff = sum(1 for k in range(aln_len) if msa[i][k] != msa[j][k])
            dists[i, j] = dists[j, i] = diff
    return dists


def check_ultrametric_conditions(
        dist_matrix: np.ndarray, item_names: list[str]
) -> tuple[bool, list[str]]:
    """Checks if a distance matrix meets the strong ultrametric condition.

    For any triplet of items (i, j, k), the two largest of the three distances
    (d_ij, d_jk, d_ik) must be equal.

    Args:
        dist_matrix (np.ndarray): Distance matrix.
        item_names (list[str]): List of item names.

    Returns:
        tuple: (is_ultrametric, violations), where `is_ultrametric` is a boolean
        and `violations` is a list of formatted strings describing failures.
    """
    n = dist_matrix.shape[0]
    if n < 3:
        return True, []
    violations = []
    for i, j, k in itertools.combinations(range(n), 3):
        distances = sorted([dist_matrix[i, j], dist_matrix[j, k], dist_matrix[i, k]])
        d_mid, d_large = distances[1], distances[2]
        if not np.isclose(d_mid, d_large):
            triplet_names = (item_names[i], item_names[j], item_names[k])
            violation_msg = (
                f"Triplet ({triplet_names[0]}, {triplet_names[1]}, {triplet_names[2]}): "
                f"The two largest distances ({d_mid:.4f} and {d_large:.4f}) are not equal."
            )
            violations.append(violation_msg)
    if violations:
        return False, violations[:5]
    else:
        return True, []


def calculate_sp_score(
        msa: list[str], match_score: int, mismatch_penalty: int, gap_penalty: int
) -> int:
    """Calculates the sum-of-pairs (SP) score for a multiple sequence alignment (MSA).

    Args:
        msa (list[str]): List of aligned sequences (all of equal length).
        match_score (int): Score to add for a match between two characters.
        mismatch_penalty (int): Penalty to add for a mismatch between two characters.
        gap_penalty (int): Penalty to add when a gap is present in one of the sequences.

    Returns:
        int: The total SP score for the alignment.
    """
    if not msa or not msa[0]:
        return 0

    num_seqs, aln_len = len(msa), len(msa[0])
    total_score = 0

    for i in range(num_seqs):
        for j in range(i + 1, num_seqs):
            pair_score = 0
            for k in range(aln_len):
                char1, char2 = msa[i][k], msa[j][k]
                if char1 == '-' and char2 == '-':
                    continue
                elif char1 == '-' or char2 == '-':
                    pair_score += gap_penalty
                elif char1 == char2:
                    pair_score += match_score
                else:
                    pair_score += mismatch_penalty
            total_score += pair_score

    return total_score
