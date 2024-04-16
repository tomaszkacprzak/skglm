import numpy as np
from numpy.linalg import norm
from numba import int32, float64
from scipy import sparse

from skglm.datafits.base import BaseDatafit
from skglm.datafits.single_task import Logistic
from skglm.utils.sparse_ops import spectral_norm, sparse_subselect_cols, spectral_norm_dense

class QuadraticGroupCorr(BaseDatafit):
    r"""Quadratic datafit used with group penalties, with additional
    penalty on correlation of weights.

    The datafit reads:

    .. math:: 1 / (2 xx n_"samples") ||y - Xw||_2 ^ 2 + w^T W w

    Attributes
    ----------
    grp_indices : array, shape (n_features,)
        The group indices stacked contiguously
        ([grp1_indices, grp2_indices, ...]).

    grp_ptr : array, shape (n_groups + 1,)
        The group pointers such that two consecutive elements delimit
        the indices of a group in ``grp_indices``.

    rho: float
        Correlation coefficient for correlation penalty.
    """

    def __init__(self, grp_ptr, grp_indices, rho):
        self.grp_ptr, self.grp_indices = grp_ptr, grp_indices
        self.rho = rho

    def get_spec(self):
        spec = (
            ('grp_ptr', int32[:]),
            ('grp_indices', int32[:]),
        )
        return spec

    def params_to_dict(self):
        return dict(grp_ptr=self.grp_ptr,
                    grp_indices=self.grp_indices)

    def get_lipschitz(self, X, y, marg_thresh=1e-20):
        grp_ptr, grp_indices = self.grp_ptr, self.grp_indices
        n_groups = len(grp_ptr) - 1

        lipschitz = np.zeros(n_groups)
        for g in range(n_groups):
            grp_g_indices = grp_indices[grp_ptr[g]: grp_ptr[g+1]]
            X_g = X[:, grp_g_indices]
            lipschitz[g] = norm(X_g, ord=2) ** 2 / len(y)
            # print('get lipschitz', g, lipschitz[g])

        return lipschitz

    def get_lipschitz_sparse(self, X_data, X_indptr, X_indices, y):
        grp_ptr, grp_indices = self.grp_ptr, self.grp_indices
        n_groups = len(grp_ptr) - 1
        n_rows = len(y)
        
        lipschitz = np.zeros(n_groups, dtype=X_data.dtype)
        for g in range(n_groups):
            grp_g_indices = grp_indices[grp_ptr[g]: grp_ptr[g+1]]
            X_data_g, X_indptr_g, X_indices_g = sparse_subselect_cols(
                grp_g_indices, X_data, X_indptr, X_indices)

            lipschitz[g] = spectral_norm(
                X_data_g, X_indptr_g, X_indices_g, n_rows) ** 2 / len(y)
            # print('get lipschitz sparse', g, lipschitz[g])

        return lipschitz

    def value(self, y, w, Xw):
        return norm(y - Xw) ** 2 / (2 * len(y))

    def gradient_g(self, X, y, w, Xw, g):
        grp_ptr, grp_indices = self.grp_ptr, self.grp_indices
        grp_g_indices = grp_indices[grp_ptr[g]: grp_ptr[g+1]]

        ng = len(grp_g_indices)
        grad_g = np.zeros(ng)
        for idx, j in enumerate(grp_g_indices):
            grad_g[idx] = self.gradient_scalar(X, y, w, Xw, j, ng)

        return grad_g

    def gradient_g_sparse(self, X_data, X_indptr, X_indices, y, w, Xw, g):
        grp_ptr, grp_indices = self.grp_ptr, self.grp_indices
        grp_g_indices = grp_indices[grp_ptr[g]: grp_ptr[g+1]]

        ng = len(grp_g_indices)
        grad_g = np.zeros(ng)
        for idx, j in enumerate(grp_g_indices):
            grad_g[idx] = self.gradient_scalar_sparse(
                X_data, X_indptr, X_indices, y, w, Xw, j, ng)

        return grad_g

    def gradient_scalar_sparse(self, X_data, X_indptr, X_indices, y, w, Xw, j, ng):

        nrm2 = 0.
        for i in range(X_indptr[j], X_indptr[j+1]):
            nrm2 += X_data[i] * (Xw[X_indices[i]] - y[X_indices[i]])

        # TODO: here add penalty for correlation using rho, ng
        # nrm2+=

        return nrm2/(len(y) + ng)

    def gradient_scalar(self, X, y, w, Xw, j, ng):

        nrm2 = (Xw - y) @ X[:, j] / len(y)
        
        # TODO: here add penalty for correlation using rho, ng
        # rnm2+= 

        return nrm2/(len(y) + ng)

    def intercept_update_step(self, y, Xw):
        return np.mean(Xw - y)



class QuadraticGroup(BaseDatafit):
    r"""Quadratic datafit used with group penalties.

    The datafit reads:

    .. math:: 1 / (2 xx n_"samples") ||y - Xw||_2 ^ 2

    Attributes
    ----------
    grp_indices : array, shape (n_features,)
        The group indices stacked contiguously
        ([grp1_indices, grp2_indices, ...]).

    grp_ptr : array, shape (n_groups + 1,)
        The group pointers such that two consecutive elements delimit
        the indices of a group in ``grp_indices``.
    """

    def __init__(self, grp_ptr, grp_indices):
        self.grp_ptr, self.grp_indices = grp_ptr, grp_indices

    def get_spec(self):
        spec = (
            ('grp_ptr', int32[:]),
            ('grp_indices', int32[:]),
        )
        return spec

    def params_to_dict(self):
        return dict(grp_ptr=self.grp_ptr,
                    grp_indices=self.grp_indices)

    def get_lipschitz(self, X, y):
        grp_ptr, grp_indices = self.grp_ptr, self.grp_indices
        n_groups = len(grp_ptr) - 1

        lipschitz = np.zeros(n_groups)
        for g in range(n_groups):
            grp_g_indices = grp_indices[grp_ptr[g]: grp_ptr[g+1]]
            X_g = X[:, grp_g_indices]
            lipschitz[g] = norm(X_g, ord=2) ** 2 / len(y)
            # print('get lipschitz', g, lipschitz[g])

        return lipschitz

    def get_lipschitz_sparse(self, X_data, X_indptr, X_indices, y):
        grp_ptr, grp_indices = self.grp_ptr, self.grp_indices
        n_groups = len(grp_ptr) - 1
        n_rows = len(y)
        
        lipschitz = np.zeros(n_groups, dtype=X_data.dtype)
        for g in range(n_groups):
            grp_g_indices = grp_indices[grp_ptr[g]: grp_ptr[g+1]]
            X_data_g, X_indptr_g, X_indices_g = sparse_subselect_cols(
                grp_g_indices, X_data, X_indptr, X_indices)

            lipschitz[g] = spectral_norm(
                X_data_g, X_indptr_g, X_indices_g, n_rows) ** 2 / len(y)
            # print('get lipschitz sparse', g, lipschitz[g])

        return lipschitz

    def value(self, y, w, Xw):
        return norm(y - Xw) ** 2 / (2 * len(y))

    def gradient_g(self, X, y, w, Xw, g):
        grp_ptr, grp_indices = self.grp_ptr, self.grp_indices
        grp_g_indices = grp_indices[grp_ptr[g]: grp_ptr[g+1]]

        grad_g = np.zeros(len(grp_g_indices))
        for idx, j in enumerate(grp_g_indices):
            grad_g[idx] = self.gradient_scalar(X, y, w, Xw, j)

        return grad_g

    def gradient_g_sparse(self, X_data, X_indptr, X_indices, y, w, Xw, g):
        grp_ptr, grp_indices = self.grp_ptr, self.grp_indices
        grp_g_indices = grp_indices[grp_ptr[g]: grp_ptr[g+1]]

        grad_g = np.zeros(len(grp_g_indices))
        for idx, j in enumerate(grp_g_indices):
            grad_g[idx] = self.gradient_scalar_sparse(
                X_data, X_indptr, X_indices, y, w, Xw, j)

        return grad_g

    def gradient_scalar_sparse(self, X_data, X_indptr, X_indices, y, w, Xw, j):

        nrm2 = 0.
        for i in range(X_indptr[j], X_indptr[j+1]):
            nrm2 += X_data[i] * (Xw[X_indices[i]] - y[X_indices[i]])

        return nrm2/len(y)

    def gradient_scalar(self, X, y, w, Xw, j):

        return (Xw - y) @ X[:, j] / len(y)

    def intercept_update_step(self, y, Xw):
        return np.mean(Xw - y)


class LogisticGroup(Logistic):
    r"""Logistic datafit used with group penalties.

    The datafit reads:

    .. math:: 1 / n_"samples" sum_(i=1)^(n_"samples") log(1 + exp(-y_i (Xw)_i))

    Attributes
    ----------
    grp_indices : array, shape (n_features,)
        The group indices stacked contiguously
        ``[grp1_indices, grp2_indices, ...]``.

    grp_ptr : array, shape (n_groups + 1,)
        The group pointers such that two consecutive elements delimit
        the indices of a group in ``grp_indices``.

    lipschitz : array, shape (n_groups,)
        The lipschitz constants for each group.
    """

    def __init__(self, grp_ptr, grp_indices):
        self.grp_ptr, self.grp_indices = grp_ptr, grp_indices

    def get_spec(self):
        spec = (
            ('grp_ptr', int32[:]),
            ('grp_indices', int32[:]),
            ('lipschitz', float64[:])
        )
        return spec

    def params_to_dict(self):
        return dict(grp_ptr=self.grp_ptr,
                    grp_indices=self.grp_indices)

    def initialize(self, X, y):
        grp_ptr, grp_indices = self.grp_ptr, self.grp_indices
        n_groups = len(grp_ptr) - 1

        lipschitz = np.zeros(n_groups)
        for g in range(n_groups):
            grp_g_indices = grp_indices[grp_ptr[g]: grp_ptr[g+1]]
            X_g = X[:, grp_g_indices]
            lipschitz[g] = norm(X_g, ord=2) ** 2 / (4 * len(y))

        self.lipschitz = lipschitz

    def gradient_g(self, X, y, w, Xw, g):
        grp_ptr, grp_indices = self.grp_ptr, self.grp_indices
        grp_g_indices = grp_indices[grp_ptr[g]: grp_ptr[g+1]]
        raw_grad_val = self.raw_grad(y, Xw)

        grad_g = np.zeros(len(grp_g_indices))
        for idx, j in enumerate(grp_g_indices):
            grad_g[idx] = X[:, j] @ raw_grad_val

        return grad_g
