import numpy as np
import pandas as pd
import scipy.linalg as linalg


def construct_standard_basis(dim):
    """
    Constructs a standard basis for a Euclidean vector space of a given dimension.

    Parameters:
        dim: A positive integer

    Returns:
        A list of column vectors forming the standard basis for a vector space of dimension dim
    """
    basematrix = np.eye(dim)
    return [basematrix[i][:, np.newaxis] for i in range(dim)]


def construct_standard_bases(dims: dict):
    """
    Constructs standard bases for a collection of Euclidean vector spaces.

    Parameters:
        dims: a dictionary whose keys (str) are the names of nodes from a NetworkX DiGraph, and whose values (int)
            are the specified dimensions of vector spaces

    Returns:
        A dictionary whose value for each key is the standard basis for a vector space of dimension dims[key]
    """
    bases = {}
    for node in list(dims.keys()):
        bases[node] = construct_standard_basis(dims[node])
    return bases


def orthogonal_complement(basis):
    """
    Constructs (an orthogonal basis for) the orthogonal complement of a subspace.

    Parameters:
        A: A list of basis vectors of fixed length m, viewed as the basis for a subspace of m-dimensional Euclidean space

    Returns:
        A list of column vectors forming the basis for the orthogonal complement. If the orthogonal complement is
            zero-dimensional, returns a list consisting of a single zero-vector of the correct length, instead of
            a basis of length 0
    """

    # Concatentate the basis a the column basis of a matrix
    A = np.concatenate(basis, axis=1).T
    # Compute the null space of this matrix
    Aperp = linalg.null_space(A)

    # If the null space is 0-dimensional, return a list consisting of a single zero-vector of the correct length.
    if Aperp.shape[1] == 0:
        return [np.zeros(len(A[0]))[:, np.newaxis]]

    # Transpose the null space matrix and list the columns
    Aperp = Aperp.T
    Aperp_cols = [np.array(Aperp[i])[:, np.newaxis] for i in range(len(Aperp))]
    return Aperp_cols


def subspace_intersection(basis_list):
    """
    Constructs (an orthogonal basis for) the intersection of a list of subspaces of Euclidean space, using the formula
        U_1 \\cap \\cdots \\cap U_m = (U_1^\\perp + \\cdots + U_m^\\perp)^\\perp.

    Parameters:
        basis_list: A list of bases, each basis being a list of column vectors. All vectors must have the same length.

    Returns:
        A list of column vectors forming the basis for the intersection
    """

    # List the orthogonal complements of each subspace's basis. This gives a spanning set for the orthogonal complement
    # of the intersection
    complement_spanning_set = []
    for A in basis_list:
        B = orthogonal_complement(A)
        complement_spanning_set.extend(B)

    # Now compute the orthogonal complement. If this is zero-dimensional, return a list with one zero-vector of the
    # correct length. Otherwise, this is the intersection of the subspaces in basis_list
    intersection = orthogonal_complement(complement_spanning_set)
    if len(intersection) == 0:
        return np.zeros(len(basis_list[0][0]))[:, np.newaxis]
    else:
        return intersection


def direct_sum(basis_list):
    """
    Constructs (an orthogonal basis for) the direct sum of a list of multiple subspaces of Euclidean spaces.

    Parameters:
        basis_list: A list of bases, each basis being a list of column vectors. The vectors in each basis must have
            the same length. In practice all

    Returns:
        - A list of column vectors forming the basis for the direct sum
        - A list of projection maps mapping the direct sum onto each summand
    """

    # Get the ambient dimensions of the subspaces (lengths of basis vectors)
    basis_dimensions = np.array([len(basis_list[k]) for k in range(len(basis_list))])
    # Determine the (ambient) dimension of the direct sum
    direct_sum_dimension = np.sum(basis_dimensions)
    # Determine the initial column indices for each summand of the direct sum
    direct_sum_indices = np.concatenate([np.array([0]), np.cumsum(basis_dimensions)])[
        :-1
    ]

    # Embed basis vectors in direct sum
    direct_sum_basis = []
    for k in range(len(basis_list)):
        basis = basis_list[k]
        starting_index = direct_sum_indices[k]
        for i in range(len(basis)):
            dim = len(basis[0])
            v = np.zeros(direct_sum_dimension)[:, np.newaxis]
            for j in range(dim):
                v[starting_index + j][0] = basis[i][j][0]
            direct_sum_basis.append(v)

    # Construct projection maps
    projection_maps = []
    id_mat = np.eye(direct_sum_dimension)
    for k in range(len(basis_list)):
        node = basis_list[k]
        dim = len(basis_list[k][0])
        proj = id_mat[direct_sum_indices[k] : direct_sum_indices[k] + dim].copy()
        projection_maps.append(proj)

    return direct_sum_basis, projection_maps


def project_onto_orthogonal_basis(s, basis):
    """
    Project a data instance onto an orthogonal basis of column vectors, in the original coordinates of the data sample.

    Parameters:
        s: A Pandas series with a single data row
        basis: A list of basis vectors of fixed length m, viewed as the basis for a subspace of m-dimensional Euclidean space

    Returns:
        A Pandas series containing the projection of s onto the basis vectors, in the original coordinates of s
    """
    # Check whether the data instance and basis vectors are of the same length
    if len(s) != len(basis[0]):
        print("Data instance and basis vectors are not of the same length.")
        return
    v = np.array(s)
    # Calculate projection of data instance
    components = [v.dot(b) * b for b in basis]
    proj = np.sum(components, axis=0).reshape(-1)
    return pd.Series(proj)


def project_onto_subspace_in_basis(s, basis):
    """
    Project a data instance onto a subspace, and express the projection in the coordinates of the subspace basis.

    Parameters:
        s: A Pandas series with a single data row
        basis: A list of basis vectors of fixed length m, viewed as the basis for a subspace of m-dimensional Euclidean space

    Returns:
        A Pandas series containing the projection of s onto the basis vectors, in the coordinates of the subspace basis
    """
    v = np.array(s)
    # Calculate projection of data instance
    components = [v.dot(b)[0] for b in basis]
    return pd.Series(components)


def equalizer_subspace(lmap_collection, res_basis=None):
    """
    Compute a basis for the equalizer subspace of a collection of linear maps.

    Parameters:
        lmap_collection: A list of linear maps with the same dimensions
        res_basis: An optional subspace specifying the domain of the linear maps, given as a list of basis vectors

    Returns:
        A basis for the maximal subspace on which the maps in lmap_collection are equal
    """
    lmap_collection = np.stack(lmap_collection, axis=0)

    # Take differences of maps to generate the subspace orthogonal to the equalizer
    lmap_diffs = np.diff(lmap_collection, axis=0)
    lmap_diffs = np.unstack(lmap_diffs)

    # Concatenate differences to express the kernel as the null space of a single matrix
    if len(lmap_diffs) == 0:
        lmap_diffs = [np.zeros(lmap_collection[0].shape)]
    lmap_diffs = np.concatenate(lmap_diffs)

    eq = linalg.null_space(lmap_diffs).T

    # Handle case of zero-dimensional equalizer subspace
    if len(eq) == 0:
        return [np.zeros(lmap_collection.shape[2])[:, np.newaxis]]

    # Reshape equalizer basis as column vectors
    eq_basis = [np.array(eq[i])[:, np.newaxis] for i in range(len(eq))]

    # Intersect with the domain, if the domain specified as a subspace
    if res_basis is not None:
        eq_basis = subspace_intersection([eq_basis, res_basis])

    return eq_basis
