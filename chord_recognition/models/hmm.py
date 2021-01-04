import numpy as np


def uniform_transition_matrix(p=0.01, N=25):
    """Computes uniform transition matrix

    Args:
        p: Self transition probability
        N: Column and row dimension

    Returns:
        A: Output transition matrix
    """
    off_diag_entries = (1-p) / (N-1)  # rows should sum up to 1
    A = off_diag_entries * np.ones([N, N])
    np.fill_diagonal(A, p)
    return A


def viterbi_log_likelihood(A, C, B_O):
    """Viterbi algorithm (log variant) for solving the uncovering problem

    Args:
        A: State transition probability matrix of dimension I x I
        C: Initial state distribution  of dimension I
        B_O: Likelihood matrix of dimension I x N

    Returns:
        S_opt: Optimal state sequence of length N
        S_mat: Binary matrix representation of optimal state sequence
        D_log: Accumulated log probability matrix
        E: Backtracking matrix
    """
    I = A.shape[0]  # Number of states
    N = B_O.shape[1]  # Length of observation sequence
    tiny = np.finfo(0.).tiny
    A_log = np.log(A + tiny)
    C_log = np.log(C + tiny)
    B_O_log = np.log(B_O + tiny)

    # Initialize D and E matrices
    D_log = np.zeros((I, N))
    E = np.zeros((I, N-1)).astype(np.int32)
    D_log[:, 0] = C_log + B_O_log[:, 0]

    # Compute D and E in a nested loop
    for n in range(1, N):
        for i in range(I):
            temp_sum = A_log[:, i] + D_log[:, n-1]
            D_log[i, n] = np.max(temp_sum) + B_O_log[i, n]
            E[i, n-1] = np.argmax(temp_sum)

    # Backtracking
    S_opt = np.zeros(N).astype(np.int32)
    S_opt[-1] = np.argmax(D_log[:, -1])
    S_opt[0] = np.argmax(D_log[:, 0])
    for n in range(N-2, 0, -1):
        S_opt[n] = E[int(S_opt[n+1]), n]

    # Matrix representation of result
    S_mat = np.zeros((I, N)).astype(np.int32)
    for n in range(N):
        S_mat[S_opt[n], n] = 1

    return S_mat, S_opt, D_log, E


def postprocess_HMM(chord_sim, p=0.15):
    """Conduct HMM-based chord recognition

    Args:
        chord_sim: Chord similarity matrix
        p: Self-transition probability used for HMM

    Returns:
        chord_HMM: HMM-based chord recogntion result given as binary matrix
    """
    # HMM-based chord recogntion
    A = uniform_transition_matrix(p=p)
    C = 1 / 25 * np.ones((1, 25))
    B_O = chord_sim
    chord_HMM, _, _, _ = viterbi_log_likelihood(A, C, B_O)
    return chord_HMM
