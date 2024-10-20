import numpy as np
import scipy.linalg as linalg


# Orthogonal complement of a subspace. Both input and output subspaces are given as bases, namely, lists of column vectors.
def orthogonal_complement(A):
    A = np.concatenate(A, axis=1).T
    Aperp = linalg.null_space(A).T
    Aperp_cols = [np.array(Aperp[i])[:,np.newaxis] for i in range(len(Aperp))]
    return Aperp_cols


# Using (U \cap V)^\perp = (U^\perp + V^\perp)^\perp
def subspace_intersection(basis_list):
    complement_spanning_set = []
    for A in basis_list:
        B = orthogonal_complement(A)
        complement_spanning_set.extend(B)
    intersection = orthogonal_complement(complement_spanning_set)
    if len(intersection) == 0:
        return np.zeros(len(basis_list[0][0]))[:,np.newaxis]
    else:
        return intersection



# Construct new basis for a direct sum of vector spaces, along with projection maps
def direct_sum(bases):
    basis_dimensions = np.array([ len(bases[k]) for k in range(len(bases)) ]) 
    direct_sum_indices = np.cumsum(basis_dimensions) - basis_dimensions[0]
    direct_sum_dimension = np.sum(basis_dimensions)

    # Embed basis vectors in direct sum
    direct_sum_basis = []
    for k in range(len(bases)):
        basis = bases[k]
        starting_index = direct_sum_indices[k]
        for i in range(len(basis)):
            dim = len(basis[0])
            v = np.zeros(direct_sum_dimension)[:,np.newaxis]
            for j in range(dim):
                v[starting_index+j][0] = basis[i][j][0]
            direct_sum_basis.append(v)

    # Construct projection maps
    projection_maps = []
    id_mat = np.eye(direct_sum_dimension)
    for k in range(len(bases)):
        node = bases[k]
        dim = len(bases[k][0])
        proj = id_mat[direct_sum_indices[k] : direct_sum_indices[k] + dim].copy()
        projection_maps.append(proj)

    return direct_sum_basis, projection_maps