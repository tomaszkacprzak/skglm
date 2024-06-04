import numpy as np
from numpy.linalg import norm
from numba import njit, prange, boolean


@njit
def spectral_norm(X_data, X_indptr, X_indices, n_samples,
                  max_iter=100, tol=1e-6):
    """Compute the spectral norm of sparse matrix ``X`` with power method.

    Parameters
    ----------
    X_data : array, shape (n_elements,)
         ``data`` attribute of the sparse CSC matrix ``X``.

    X_indptr : array, shape (n_features + 1,)
         ``indptr`` attribute of the sparse CSC matrix ``X``.

    X_indices : array, shape (n_elements,)
         ``indices`` attribute of the sparse CSC matrix ``X``.

    n_samples : int
        number of rows of ``X``.

    max_iter : int, default 20
        Maximum number of power method iterations.

    tol : float, default 1e-6
        Tolerance for convergence.

    Returns
    -------
    eigenvalue : float
        The largest singular value of ``X``.

    References
    ----------
    .. [1] Alfio Quarteroni, Riccardo Sacco, Fausto Saleri "Numerical Mathematics",
        chapter 5, page 192-195.
    """
    # init vec with norm(vec) == 1.
    eigenvector = np.random.randn(n_samples)
    eigenvector /= norm(eigenvector)
    eigenvalue = 1.

    for _ in range(max_iter):
        vec = _XXT_dot_vec(X_data, X_indptr, X_indices, eigenvector, n_samples)
        norm_vec = norm(vec)
        eigenvalue = vec @ eigenvector

        # this seems to be necessary to prevent numerical precision-related crashes with njit
        if not np.isfinite(norm_vec):
            break

        # norm(X @ X.T @ eigenvector - eigenvalue * eigenvector) <= tol
        # inequality (5.25) in ref [1] is squared
        if (norm_vec ** 2 - eigenvalue ** 2) <= tol ** 2:
            break

        eigenvector = vec / norm_vec

    return np.sqrt(eigenvalue)


@njit
def sparse_columns_slice(cols, X_data, X_indptr, X_indices):
    """Select a sub matrix from CSC sparse matrix.

    Similar to ``X[:, cols]`` but for ``X`` a CSC sparse matrix.

    Parameters
    ----------
    cols : array of int
        Columns to select in matrix ``X``.

    X_data : array, shape (n_elements,)
        ``data`` attribute of the sparse CSC matrix ``X``.

    X_indptr : array, shape (n_features + 1,)
        ``indptr`` attribute of the sparse CSC matrix ``X``.

    X_indices : array, shape (n_elements,)
        ``indices`` attribute of the sparse CSC matrix ``X``.

    Returns
    -------
    sub_X_data, sub_X_indptr, sub_X_indices
        The ``data``, ``indptr``, and ``indices`` attributes of the sub matrix.
    """
    nnz = sum([X_indptr[j+1] - X_indptr[j] for j in cols])

    sub_X_indptr = np.zeros(len(cols) + 1, dtype=cols.dtype)
    sub_X_indices = np.zeros(nnz, dtype=X_indices.dtype)
    sub_X_data = np.zeros(nnz, dtype=X_data.dtype)

    for idx, j in enumerate(cols):
        n_elements = X_indptr[j+1] - X_indptr[j]
        sub_X_indptr[idx + 1] = sub_X_indptr[idx] + n_elements

        col_j_slice = slice(X_indptr[j], X_indptr[j+1])
        col_idx_slice = slice(sub_X_indptr[idx], sub_X_indptr[idx+1])

        sub_X_indices[col_idx_slice] = X_indices[col_j_slice]
        sub_X_data[col_idx_slice] = X_data[col_j_slice]

    return sub_X_data, sub_X_indptr, sub_X_indices


@njit
def _XXT_dot_vec(X_data, X_indptr, X_indices, vec, n_samples):
    # computes X @ X.T @ vec, with X csc encoded
    return _X_dot_vec(X_data, X_indptr, X_indices,
                      _XT_dot_vec(X_data, X_indptr, X_indices, vec), n_samples)


@njit
def _X_dot_vec(X_data, X_indptr, X_indices, vec, n_samples):
    # compute X @ vec, with X csc encoded
    result = np.zeros(n_samples)

    # loop over features
    for j in range(len(X_indptr) - 1):
        if vec[j] == 0:
            continue

        col_j_rows_idx = slice(X_indptr[j], X_indptr[j+1])
        result[X_indices[col_j_rows_idx]] += vec[j] * X_data[col_j_rows_idx]

    return result


@njit
def _XT_dot_vec(X_data, X_indptr, X_indices, vec):
    # compute X.T @ vec, with X csc encoded
    n_features = len(X_indptr) - 1
    result = np.zeros(n_features)

    for j in range(n_features):
        for idx in range(X_indptr[j], X_indptr[j+1]):
            result[j] += X_data[idx] * vec[X_indices[idx]]

    return result


@njit(fastmath=True)
def _sparse_xj_dot(X_data, X_indptr, X_indices, j, other):
    # Compute X[:, j] @ other in case X sparse
    res = 0.
    for i in range(X_indptr[j], X_indptr[j+1]):
        res += X_data[i] * other[X_indices[i]]
    return res

@njit(fastmath=True)
def sparse_subselect_cols(cols, X_data, X_indptr, X_indices):
    """Summary
    
    Parameters
    ----------
    cols : TYPE
        Description
    X_data : TYPE
        Description
    X_indices : TYPE
        Description
    X_indptr : TYPE
        Description
    """
    n_indices_j = np.zeros(len(cols), dtype=X_indices.dtype)
    X_indptr_g = np.zeros(len(cols)+1, dtype=X_indices.dtype)
    for i, j in enumerate(cols):
        n_indices_j[i] += X_indptr[j+1]-X_indptr[j]
        X_indptr_g[i+1] = X_indptr_g[i] + n_indices_j[i]
    n_indices_g = np.sum(n_indices_j)
    X_indices_g = np.zeros(n_indices_g, dtype=X_indices.dtype)
    X_data_g = np.zeros(n_indices_g, dtype=X_data.dtype)

    for i, j in enumerate(cols):
        X_indices_g[X_indptr_g[i]:X_indptr_g[i+1]] = X_indices[X_indptr[j]:X_indptr[j+1]]
        X_data_g[X_indptr_g[i]:X_indptr_g[i+1]] = X_data[X_indptr[j]:X_indptr[j+1]]

    return X_data_g, X_indptr_g, X_indices_g


@njit
def spectral_norm_dense(X, max_iter=100, tol=1e-6):
    """Compute the spectral norm of matrix ``X`` with power method.

    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        Design matrix.

    max_iter : int, default 20
        Maximum number of power method iterations.

    tol : float, default 1e-6
        Tolerance for convergence.

    Returns
    -------
    eigenvalue : float
        The largest singular value of ``X``.

    References
    ----------
    .. [1] Alfio Quarteroni, Riccardo Sacco, Fausto Saleri "Numerical Mathematics",
        chapter 5, page 192-195.
    """
    # init vec with norm(vec) == 1.
    eigenvector = np.random.randn(X.shape[0])
    eigenvector /= norm(eigenvector)
    eigenvalue = 1.

    XXT = np.dot(X, X.T)
    for _ in range(max_iter):

        vec = np.dot(XXT, eigenvector)
        norm_vec = norm(vec)
        eigenvalue = vec @ eigenvector

        # norm(X @ X.T @ eigenvector - eigenvalue * eigenvector) <= tol
        # inequality (5.25) in ref [1] is squared
        if norm_vec ** 2 - eigenvalue ** 2 <= tol ** 2:
            break
        
        eigenvector = vec / norm_vec

    return np.sqrt(eigenvalue)

# @njit(fastmath=True)
def sparse_subselect_rows(rows, X_data, X_indptr, X_indices):
    """Summary
    
    Parameters
    ----------
    cols : TYPE
        Description
    X_data : TYPE
        Description
    X_indices : TYPE
        Description
    X_indptr : TYPE
        Description
    """

    # calculate number of indices for each column
    # for r in rows:
    #     n_indices_j[i] += 

    X_indptr_g = np.zeros(len(X_indptr), dtype=X_indptr.dtype)
    n_indices_g = np.zeros(len(X_indptr)-1, dtype=X_indptr.dtype)
    for i in range(len(X_indptr)-1):
        inds_g = X_indices[X_indptr[i]:X_indptr[i+1]]
        select = in1d_vec_nb(inds_g, rows)
        n_indices_g[i] = np.count_nonzero(select)
        X_indptr_g[i+1] = np.sum(n_indices_g)

    n_indices = np.sum(n_indices_g)
    X_data_g = np.zeros(n_indices, dtype=X_data.dtype)
    X_indices_g = np.zeros(n_indices, dtype=X_indices.dtype)
    for i in range(len(X_indptr)-1):
        inds_g = X_indices[X_indptr[i]:X_indptr[i+1]]
        select = in1d_vec_nb(inds_g, rows)
        X_indices_g[X_indptr_g[i]:X_indptr_g[i+1]] = inds_g[select]
        X_data_g[X_indptr_g[i]:X_indptr_g[i+1]] = X_data[X_indptr[i]:X_indptr[i+1]][select]

    return X_data_g, X_indptr_g, X_indices_g



@njit(fastmath=True)
def sparse_sum_rows(X_data, X_indptr, X_indices, n_samples):
    """Summary
    
    Parameters
    ----------
    X_data : TYPE
        Description
    X_indices : TYPE
        Description
    X_indptr : TYPE
        Description
    """

    sum_rows = np.zeros(n_samples, dtype=X_data.dtype)
    for j in range(len(X_indices)):
        sum_rows[X_indices[j]] += X_data[j]

    return sum_rows


# @njit(parallel=True)
def in1d_vec_nb(ar1, ar2):
  #arr and ar2 have to be numpy arrays
  #if ar2 is a list with different dtypes this 
  #function will fail

  out=np.empty(ar1.shape[0], dtype=bool)
  ar2_set=set(ar2)

  for i in prange(ar1.shape[0]):
    if ar1[i] in ar2_set:
      out[i]=True
    else:
      out[i]=False

  return out
