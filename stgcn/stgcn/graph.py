import numpy as np

class Graph:
    def __init__(self):
        self.o2r = [(1, 0), (1, 5), (1, 4), (3, 5), (3, 0), (3, 6), (2, 4), (2, 0), (2, 6)]
        self.r2o = [(j, i) for (i, j) in self.o2r]
        self.num_node = 7
        # 7: [x, head_x, lhand_x, rhand_x, h_lh_re_x, h_rh_re_x, rh_lh_re_x]
        self.node_rel = [1, 0, 0, 0, 1, 1, 1]
        self.rel = self.get_adjacency_matrix()

    def get_adjacency_matrix(self):
        o2rmat = edge2mat(self.o2r, self.num_node)
        o2rmat = normalize_digraph(o2rmat)

        r2omat = edge2mat(self.r2o, self.num_node)
        r2omat = normalize_digraph(r2omat)

        rel = np.stack([o2rmat, r2omat])
        return rel

def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A

def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD