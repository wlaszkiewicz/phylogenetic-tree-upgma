import numpy as np

def needleman_wunsch(seq1_str: str, seq2_str: str, match_score: int = 1,
                     mismatch_penalty: int = -1, gap_penalty: int = -1) -> tuple[float, str, str]:
    seq1 = list(seq1_str)
    seq2 = list(seq2_str)
    n = len(seq1)
    m = len(seq2)
    dp = np.zeros((n + 1, m + 1))

    # Initialization
    for i in range(n + 1):
        dp[i][0] = i * gap_penalty
    for j in range(m + 1):
        dp[0][j] = j * gap_penalty

    # DP Matrix Fill
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            char1 = seq1[i - 1]
            char2 = seq2[j - 1]

            current_match_mismatch_score = 0
            if char1 == char2 and char1 != '-':
                current_match_mismatch_score = match_score
            elif char1 == '-' and char2 == '-':
                current_match_mismatch_score = 0
            elif char1 == '-' or char2 == '-':
                current_match_mismatch_score = gap_penalty
            else:
                current_match_mismatch_score = mismatch_penalty

            score_diag = dp[i - 1][j - 1] + current_match_mismatch_score
            score_up = dp[i - 1][j] + gap_penalty
            score_left = dp[i][j - 1] + gap_penalty
            dp[i][j] = max(score_diag, score_up, score_left)

    # Traceback
    aligned_s1 = []
    aligned_s2 = []
    i_tb = n
    j_tb = m
    score = dp[n][m]

    while i_tb > 0 or j_tb > 0:
        current_score_val = dp[i_tb][j_tb]
        s1_char_tb = seq1[i_tb - 1] if i_tb > 0 else ''
        s2_char_tb = seq2[j_tb - 1] if j_tb > 0 else ''

        diag_path_score_component = 0
        if i_tb > 0 and j_tb > 0:
            if s1_char_tb == s2_char_tb and s1_char_tb != '-':
                diag_path_score_component = match_score
            elif s1_char_tb == '-' and s2_char_tb == '-':
                diag_path_score_component = 0
            elif s1_char_tb == '-' or s2_char_tb == '-':
                diag_path_score_component = gap_penalty
            else:
                diag_path_score_component = mismatch_penalty

        if (i_tb > 0 and j_tb > 0 and
                abs(current_score_val - (dp[i_tb - 1][j_tb - 1] + diag_path_score_component)) < 1e-9):
            aligned_s1.append(s1_char_tb)
            aligned_s2.append(s2_char_tb)
            i_tb -= 1
            j_tb -= 1
        elif i_tb > 0 and abs(current_score_val - (dp[i_tb - 1][j_tb] + gap_penalty)) < 1e-9:
            aligned_s1.append(s1_char_tb)
            aligned_s2.append('-')
            i_tb -= 1
        elif j_tb > 0 and abs(current_score_val - (dp[i_tb][j_tb - 1] + gap_penalty)) < 1e-9:
            aligned_s1.append('-')
            aligned_s2.append(s2_char_tb)
            j_tb -= 1
        else:
            if i_tb > 0:
                aligned_s1.append(s1_char_tb)
                aligned_s2.append('-')
                i_tb -= 1
            elif j_tb > 0:
                aligned_s1.append('-')
                aligned_s2.append(s2_char_tb)
                j_tb -= 1
            else:
                break

    return score, "".join(reversed(aligned_s1)), "".join(reversed(aligned_s2))