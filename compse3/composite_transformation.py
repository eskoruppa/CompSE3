from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import sys
from typing import Optional
import numpy as np
import scipy as sp
from scipy.sparse import csr_matrix, csc_matrix, eye, lil_matrix
from scipy.sparse import issparse
from scipy.sparse.linalg import spsolve
from scipy.linalg import solve

from .composite_junctions import SE3CompositeJunction, JunctionTopology, Topo


COMP_TRANSFORM_FRONTS = ['front', 'first']
COMP_TRANSFORM_BACKS = ['back', 'last']

COMP_TRANSFORM_FRONT = COMP_TRANSFORM_FRONTS[0]
COMP_TRANSFORM_BACK = COMP_TRANSFORM_BACKS[0]   

FE_ATTRIBUTE_KEY_F   = 'F'
FE_ATTRIBUTE_KEY_FFL = 'F_fluctuation'
FE_ATTRIBUTE_KEY_FEN = 'F_enthalpy'
FE_ATTRIBUTE_KEY_FJC = 'F_jacob'
FE_ATTRIBUTE_KEY_FFR = 'F_free'
FE_ATTRIBUTE_KEY_DF  = 'dF'
FE_ATTRIBUTE_KEY_GS  = 'gs'


class SE3CompositeTransform:
    
    DIM_PER_JUNCTION = 6
    
    JunctionTopology = JunctionTopology
    Topo = JunctionTopology

    def __init__(
        self, 
        static_junctions: np.ndarray,
        composite_order: str = 'last',
        supress_warnings: bool = False,
        iterative: bool = False
        ):
        """Initialize the transformation matrix builder.
        
        Parameters:
            static_junctions: Array of static junction transforms
            composite_order: Where to place composites. 
                           'last'/'back' - place at end (default)
                           'first'/'front' - place at beginning
        """
        self.static_junctions = static_junctions
        self.njuncs = len(static_junctions)
        self.composites = []
        self.composite_order = self._normalize_composite_order(composite_order)
        self.supress_warnings = supress_warnings
        self.iterative = iterative

        self._reset_ptrs()
    
    def add_composite(
        self, 
        junction_ids: list[int], 
        junction_topologies: list[JunctionTopology] | list[str],
        replaced_id: int,
        key: str | int,
        pose1: np.ndarray,
        pose2: np.ndarray,
        ) -> SE3CompositeJunction:

        if self.replaced_id_contained(replaced_id):
            raise ValueError(f"replaced_id {replaced_id} is already used in another composite. Please choose a unique replaced_id for each composite.")    
        
        if replaced_id < 0 or replaced_id >= self.njuncs:
            raise ValueError(f"replaced_id {replaced_id} is out of bounds for number of junctions {self.njuncs}. Must be between 0 and {self.njuncs - 1}.")
        
        if not isinstance(key, (str, int)):
            raise TypeError(f"key must be a string or integer. Got {type(key)}: {key}")
        
        if self.key_contained(key):
            raise ValueError(f"key {key} is already used in another composite. Please choose a unique key for each composite.")
        
        X0 = self.static_junctions[junction_ids]
        comp = SE3CompositeJunction(
            X0,
            junction_topologies,
            junction_ids=junction_ids,
            key=key,
            pose1=pose1,
            pose2=pose2
        )
        comp.set_replaced_id(replaced_id)
        self.composites.append(comp)
        
        self._reset_ptrs()
        return comp

    def dynamic_composite_excess_junction(self):
        excess_junction = np.zeros((len(self.composites),4,4),dtype=np.float64)
        for i,comp in enumerate(self.composites):
            excess_junction[i] = comp._dynamic_composite_excess_junction()
        return excess_junction
    
    def dynamic_composite_excess_coordinates(self):
        excess_coordinates = np.zeros((len(self.composites),6),dtype=np.float64)
        for i,comp in enumerate(self.composites):
            excess_coordinates[i] = comp.dynamic_composite_excess_coordinates() 
        return excess_coordinates

    def transform_stiffness_matrix(self, stiffmat: csr_matrix | np.ndarray) -> csr_matrix | np.ndarray:
        """Transform stiffness matrix using congruence transformation: K' = T^-T @ K @ T^-1"""
        
        transform = self.transformation_matrix()
        
        if stiffmat.shape != transform.shape:
            raise ValueError(f"Shape mismatch: stiffmat {stiffmat.shape} vs transform {transform.shape}")
        
        if issparse(stiffmat):
            return self._sparse_congruence_transformation(stiffmat, transform)
        else:
            return self._dense_congruence_transformation(stiffmat, transform)


    @staticmethod
    def _sparse_congruence_transformation(A: csr_matrix, T: csr_matrix) -> csr_matrix:
        """Sparse congruence transformation using solve (numerically stable)."""

        # def sparsity(A):
        #     # Count non-zero entries
        #     num_nonzero = A.nnz
        #     # You can also get the total number of elements
        #     total_elements = A.shape[0] * A.shape[1]
        #     return 1 - (num_nonzero / total_elements)
        # print(f"Sparsity A: {sparsity(A):.2%}")
        # print(f"Sparsity T: {sparsity(T):.2%}")
        # print(f'shape: {A.shape}')

        X = spsolve(T, A.T, permc_spec='COLAMD')
        if not issparse(X):
            X = csr_matrix(X)
        else:
            X = X.tocsr()
        # Second solve: T.T @ Y = X.T
        A_transformed = spsolve(T.T.tocsc(), X.T, permc_spec='COLAMD')
        return csr_matrix(A_transformed) if not issparse(A_transformed) else A_transformed

    @staticmethod
    def _dense_congruence_transformation(stiffmat: np.ndarray, transform: csr_matrix | np.ndarray) -> np.ndarray:
        """Dense congruence transformation using solve."""
        T = transform.toarray() if issparse(transform) else transform
        X = solve(T.T, stiffmat.T)
        return solve(T.T, X.T)

    def transformation_matrix(self) -> np.ndarray:
        if self._transformation_matrix is None:
            self._transformation_matrix = self._build_transformation_matrix()
        return self._transformation_matrix
    
    def _build_transformation_matrix(self) -> np.ndarray:
        """Build the transformation matrix efficiently in the desired order.
        
        The matrix replaces certain junction coordinates with composite coordinates
        and reorders indices according to composite_order, all in a single pass.
        """
        
        transf_mat = np.zeros((self.njuncs * self.DIM_PER_JUNCTION,self.njuncs * self.DIM_PER_JUNCTION))
        
        old_to_new = self.old_to_new
        
        for rid in self.retained_ids:
            col_start = rid * self.DIM_PER_JUNCTION
            col_end   = col_start + self.DIM_PER_JUNCTION
            row_start = old_to_new[rid] * self.DIM_PER_JUNCTION
            row_end   = row_start + self.DIM_PER_JUNCTION
            # transf_mat[row_start:row_end,col_start:col_end] = eye(self.DIM_PER_JUNCTION, format='lil')
            transf_mat[row_start:row_end,col_start:col_end] = np.eye(self.DIM_PER_JUNCTION)
        
        for comp in self.composites:
            row_start  = old_to_new[comp.replaced_id] * self.DIM_PER_JUNCTION
            row_end    = row_start + self.DIM_PER_JUNCTION

            composite_transforms = comp.build_transforms()
            for junc_idx, junc_id in enumerate(comp.junction_ids):
                col_start = junc_id * self.DIM_PER_JUNCTION
                col_end   = col_start + self.DIM_PER_JUNCTION
                transf_mat[row_start:row_end,col_start:col_end] = composite_transforms[junc_idx]

        return transf_mat
    
    def corrected_transformation_matrix(self, excess_dynamic_coordinates: np.ndarray) -> np.ndarray:
        if self._corrected_transformation_matrix is None:
            self._corrected_transformation_matrix, self._corrected_constant_vector = self._build_corrected_transformation_matrix(excess_dynamic_coordinates)
        return self._corrected_transformation_matrix, self._corrected_constant_vector

    def _build_corrected_transformation_matrix(self, excess_dynamic_coordinates: np.ndarray) -> np.ndarray:
        """
            excess_dynamic_coordinates: These are the excess values of the original junctions, not the composites
        """
        if len(excess_dynamic_coordinates.shape) == 1:
            excess_dynamic_coordinates = excess_dynamic_coordinates.reshape(-1, self.DIM_PER_JUNCTION)
        
        transf_mat = np.zeros((self.njuncs * self.DIM_PER_JUNCTION,self.njuncs * self.DIM_PER_JUNCTION))
        const = np.zeros(len(self.composites)*self.DIM_PER_JUNCTION, dtype=np.float64)

        old_to_new = self.old_to_new
        
        for rid in self.retained_ids:
            col_start = rid * self.DIM_PER_JUNCTION
            col_end   = col_start + self.DIM_PER_JUNCTION
            row_start = old_to_new[rid] * self.DIM_PER_JUNCTION
            row_end   = row_start + self.DIM_PER_JUNCTION
            # transf_mat[row_start:row_end,col_start:col_end] = eye(self.DIM_PER_JUNCTION, format='lil')
            transf_mat[row_start:row_end,col_start:col_end] = np.eye(self.DIM_PER_JUNCTION)
        
        shift_const_id = 0
        if self.composite_order == COMP_TRANSFORM_BACK:
            shift_const_id = len(self.retained_ids) * self.DIM_PER_JUNCTION

        if self.iterative:
            self.corr_excess = np.zeros((len(self.composites), self.DIM_PER_JUNCTION), dtype=np.float64)
        # for comp in self.composites:
        for i,comp in enumerate(self.composites):
            row_start  = old_to_new[comp.replaced_id] * self.DIM_PER_JUNCTION
            row_end    = row_start + self.DIM_PER_JUNCTION

            if self.iterative:
                composite_transforms, constant_vector, self.corr_excess[i] = comp.build_transforms_iterative_correction(
                    excess_dynamic_coordinates[comp.junction_ids])

            else:
                composite_transforms, constant_vector = comp.build_corrected_transforms(
                    excess_dynamic_coordinates[comp.junction_ids])

            # composite_transforms, constant_vector = comp.build_corrected_transforms(excess_dynamic_coordinates[comp.junction_ids])
            const[row_start-shift_const_id:row_end-shift_const_id] = constant_vector

            for junc_idx, junc_id in enumerate(comp.junction_ids):
                col_start = junc_id * self.DIM_PER_JUNCTION
                col_end   = col_start + self.DIM_PER_JUNCTION
                transf_mat[row_start:row_end,col_start:col_end] = composite_transforms[junc_idx]

        return transf_mat,const

    @staticmethod
    def _normalize_composite_order(order_str: str) -> str:
        """Normalize composite order string to standard form.
        
        Parameters:
            order_str: User-provided order string (case-insensitive)
        
        Returns:
            Normalized order: 'COMP_TRANSFORM_BACK' or 'COMP_TRANSFORM_FRONT'
        
        Raises:
            ValueError: If order string is invalid
        """
        normalized = order_str.lower().strip()
        if normalized in COMP_TRANSFORM_BACKS:
            return COMP_TRANSFORM_BACK
        elif normalized in COMP_TRANSFORM_FRONTS:
            return COMP_TRANSFORM_FRONT
        else:
            raise ValueError(f"Invalid composite_order: '{order_str}'. "
                           f"Valid options: {',  '.join(COMP_TRANSFORM_BACKS + COMP_TRANSFORM_FRONTS)} (case-insensitive)")
    

    def _compute_index_mapping(self) -> np.ndarray:
        """Compute the reordering of junction indices and populate key-to-index mappings.
        
        This function builds both forward (_old_to_new) and inverse (_new_to_old) mappings
        in a single pass for efficiency.
        
        Returns:
            _old_to_new: Array mapping old indices to new indices
        """        
        self._old_to_new = np.zeros(self.njuncs, dtype=np.int32)
        self._new_to_old = np.zeros(self.njuncs, dtype=np.int32)
        
        # Clear/initialize key mappings
        self._composite_key_to_full_index = {}
        self._composite_key_to_composite_index = {}
        self._full_index_to_composite_key = {}
        self._composite_index_to_composite_key = {}
        
        # Build index mappings and key mappings in a single pass
        if self.composite_order == COMP_TRANSFORM_BACK:
            new_idx = 0
            # Retained (non-replaced) junctions first
            for old_idx in self.retained_ids:
                self._old_to_new[old_idx] = new_idx
                self._new_to_old[new_idx] = old_idx
                new_idx += 1
            # Composites at the end
            for composite_idx, comp in enumerate(self.composites):
                self._old_to_new[comp.replaced_id] = new_idx
                self._new_to_old[new_idx] = comp.replaced_id
                self._composite_key_to_full_index[comp.key] = new_idx
                self._composite_key_to_composite_index[comp.key] = composite_idx
                self._full_index_to_composite_key[new_idx] = comp.key
                self._composite_index_to_composite_key[composite_idx] = comp.key
                new_idx += 1
        elif self.composite_order == COMP_TRANSFORM_FRONT:
            new_idx = 0
            # Composites at the front
            for composite_idx, comp in enumerate(self.composites):
                self._old_to_new[comp.replaced_id] = new_idx
                self._new_to_old[new_idx] = comp.replaced_id
                self._composite_key_to_full_index[comp.key] = new_idx
                self._composite_key_to_composite_index[comp.key] = composite_idx
                self._full_index_to_composite_key[new_idx] = comp.key
                self._composite_index_to_composite_key[composite_idx] = comp.key
                new_idx += 1
            # Retained junctions at the end
            for old_idx in self.retained_ids:
                self._old_to_new[old_idx] = new_idx
                self._new_to_old[new_idx] = old_idx
                new_idx += 1
        else:
            raise ValueError(f"Invalid composite_order: {self.composite_order}. "
                           f"Must be {COMP_TRANSFORM_FRONT} or {COMP_TRANSFORM_BACK}.")
        
        return self._old_to_new
    

    def replaced_id_contained(self, replaced_id: int | str) -> bool:
        for comp in self.composites:
            if comp.replaced_id == replaced_id:
                return True
        return False
    
    def key_contained(self, key: int | str) -> bool:
        for comp in self.composites:
            if comp.key == key:
                return True
        return False
    
    @property
    def replaced_ids(self) -> list[int]:
        if self._replaced_ids is None:
            self._replaced_ids = [comp.replaced_id for comp in self.composites]
        return self._replaced_ids
    
    @property
    def frozenset_replaced_ids(self) -> frozenset[int]:
        if self._frozenset_replaced_ids is None:
            self._frozenset_replaced_ids = frozenset(self.replaced_ids)
        return self._frozenset_replaced_ids

    @property
    def retained_ids(self) -> list[int]:
        if self._retained_ids is None:
            self._retained_ids = [i for i in range(self.njuncs) if i not in self.replaced_ids]
        return self._retained_ids
    
    @property
    def old_to_new(self) -> np.ndarray:
        if self._old_to_new is None:
            self._compute_index_mapping()
        return self._old_to_new
    
    @property
    def new_to_old(self) -> np.ndarray:
        if self._new_to_old is None:
            self._compute_index_mapping()
        return self._new_to_old
    
    @property
    def composite_key_to_full_index(self) -> dict[int | str, int]:
        if self._composite_key_to_full_index is None:
            self._compute_index_mapping()
        return self._composite_key_to_full_index
    
    @property
    def composite_key_to_composite_index(self) -> dict[int | str, int]:
        if self._composite_key_to_composite_index is None:
            self._compute_index_mapping()
        return self._composite_key_to_composite_index
    
    @property
    def full_index_to_composite_key(self) -> dict[int, int | str]:
        if self._full_index_to_composite_key is None:
            self._compute_index_mapping()
        return self._full_index_to_composite_key
    
    @property
    def composite_index_to_composite_key(self) -> dict[int, int | str]:
        if self._composite_index_to_composite_key is None:
            self._compute_index_mapping()
        return self._composite_index_to_composite_key


    def _reset_ptrs(self):
        self._transformation_matrix = None
        self._corrected_transformation_matrix = None
        self._corrected_constant_vector = None
        self._replaced_ids = None
        self._frozenset_replaced_ids = None
        self._retained_ids = None
        self._new_to_old = None
        self._old_to_new = None
        self._composite_key_to_full_index = None
        self._composite_key_to_composite_index = None
        self._full_index_to_composite_key = None
        self._composite_index_to_composite_key = None


    # ============================================================================================#
    #  free energy computation of hard constraint model
    # ============================================================================================#  

    def hard_constraint(self, stiffmat: np.ndarray, correction: bool = True, mode: str = 'optimized'):
        """Compute the free energy contribution of the hard constraint imposed by the composite junctions, given a stiffness matrix for the full junction set.

        Parameters
        ----------
        stiffmat : ndarray
            Stiffness matrix for the full junction set (N×N).
        correction : bool
            Whether to apply the second-order correction pass.
        mode : str
            Method for computing the free energy. Options are:
            - 'explicit': Explicit inversion of the relevant submatrices (non-optimized, straightforward).
            - 'solve': Use linear solvers instead of explicit inversion (non-optimized, more stable).
            - 'optimized': Integrated optimized computation with structure analysis and efficient solvers (recommended for large systems).

        Returns
        -------
        dict with keys: F, F_fluctuation, F_enthalpy, F_jacob, F_freedna,
             dF, gs.
        """ 
        VALID_MODES = ['explicit', 'solve', 'optimized']

        if len(self.composites) == 0:
            if mode == 'explicit' or mode == 'solve':
                return self._unconstrained_free_energy(stiffmat)
            elif mode == 'optimized':
                return self._unconstrained_free_energy_optimized(stiffmat)
            else:
                raise ValueError(f"Invalid mode: {mode}. Must be in {', '.join(VALID_MODES)}.")
        
        if mode == 'explicit':
            return self._hc_explicit(stiffmat, correction=correction)
        elif mode == 'solve':
            return self._hc_solve(stiffmat, correction=correction)
        elif mode == 'optimized':
            return self._hc_optimized(stiffmat, correction=correction)
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be in {', '.join(VALID_MODES)}.")
    
    # ============================================================================================#
    #  free energy computation for unconstrained system (reference state)
    # ============================================================================================#  

    def _unconstrained_free_energy(self, stiffmat: np.ndarray) -> float:
        """Compute the free energy of the unconstrained system for reference."""
        _, ff_logdet = np.linalg.slogdet(stiffmat)
        ff_pi = -0.5*len(stiffmat) * np.log(2*np.pi)
        F_free = 0.5*ff_logdet + ff_pi  
        Fdict = {
            FE_ATTRIBUTE_KEY_F    : F_free,
            FE_ATTRIBUTE_KEY_FFL  : F_free,
            FE_ATTRIBUTE_KEY_FEN  : 0,
            FE_ATTRIBUTE_KEY_FJC  : 0,
            FE_ATTRIBUTE_KEY_FFR  : F_free,
            FE_ATTRIBUTE_KEY_DF   : 0,
            FE_ATTRIBUTE_KEY_GS   : np.zeros(len(stiffmat), dtype=np.float64)
        }
        return Fdict


    def _unconstrained_free_energy_optimized(self, stiffmat: np.ndarray | csr_matrix) -> dict:
        """Compute the free energy of the unconstrained system using optimized log-det.

        Mirrors the bandwidth-detection and Cholesky strategy of
        :meth:`_hc_optimized`:

        1. Sparse input is converted to a dense array once.
        2. A binary-search bandwidth scan decides between banded Cholesky
           (O(N · bw²)) and dense Cholesky (O(N³)).  ``check_finite=False``
           avoids redundant input validation inside scipy.
        3. Falls back to ``np.linalg.slogdet`` only if both Cholesky paths
           raise a ``LinAlgError`` (non-SPD matrix).
        """
        K = stiffmat.toarray() if issparse(stiffmat) else np.asarray(stiffmat, dtype=np.float64)
        N = len(K)

        # ---- bandwidth detection (binary search, mirrors _hc_optimized) ----
        bw_limit = N // 4
        if not np.any(K.diagonal(1)):
            # strictly diagonal — bw stays 0, use dense cho (N is tiny)
            bw = 0
        elif bw_limit <= 1 or np.any(K.diagonal(min(bw_limit, N - 1))):
            bw = bw_limit          # bandwidth >= N//4 → treat as dense
        else:
            lo, hi, bw = 1, bw_limit - 1, 1
            while lo <= hi:
                mid = (lo + hi) // 2
                if np.any(K.diagonal(mid)):
                    bw = mid
                    lo = mid + 1
                else:
                    hi = mid - 1

        # ---- log-det via banded or dense Cholesky --------------------------
        if bw > 0 and bw < N // 4:
            # Pack upper band into LAPACK banded storage
            ab = np.zeros((bw + 1, N), dtype=K.dtype)
            for d in range(bw + 1):
                ab[bw - d, d:] = K.diagonal(d)
            try:
                cb = sp.linalg.cholesky_banded(ab, lower=False,
                                               check_finite=False)
                ff_logdet = 2.0 * np.sum(np.log(cb[bw]))
            except np.linalg.LinAlgError:
                # Fallback to dense cho when banded cho fails (e.g. fill-in)
                cho = sp.linalg.cho_factor(K, check_finite=False)
                ff_logdet = 2.0 * np.sum(np.log(np.diag(cho[0])))
        else:
            try:
                cho = sp.linalg.cho_factor(K, check_finite=False)
                ff_logdet = 2.0 * np.sum(np.log(np.diag(cho[0])))
            except np.linalg.LinAlgError:
                _, ff_logdet = np.linalg.slogdet(K)

        ff_pi = -0.5 * N * np.log(2.0 * np.pi)
        F_free = 0.5 * ff_logdet + ff_pi
        Fdict = {
            FE_ATTRIBUTE_KEY_F    : F_free,
            FE_ATTRIBUTE_KEY_FFL  : F_free,
            FE_ATTRIBUTE_KEY_FEN  : 0,
            FE_ATTRIBUTE_KEY_FJC  : 0,
            FE_ATTRIBUTE_KEY_FFR  : F_free,
            FE_ATTRIBUTE_KEY_DF   : 0,
            FE_ATTRIBUTE_KEY_GS   : np.zeros(N, dtype=np.float64)
        }
        return Fdict
    
    # ============================================================================================#
    #  free energy computation of hard constraint model with explicit inversion (non-optimized)
    # ============================================================================================#  

    def _hc_explicit(self, stiffmat: np.ndarray, correction: bool = True):

        # Step 1: Get transformation matrix and initial block structure
        A = self.transformation_matrix()
        A_inv = np.linalg.inv(A)
        MXi = A_inv.T @ stiffmat @ A_inv

        N  = len(MXi)
        N_C = len(self.replaced_ids) * self.DIM_PER_JUNCTION
        N_R = N-N_C

        if self.composite_order == COMP_TRANSFORM_FRONT:
            raise ValueError(f"Not implemented for order '{self.composite_order}'")

        elif self.composite_order == COMP_TRANSFORM_BACK:
            M_R  = MXi[:N_R,:N_R]
            M_C  = MXi[N_R:,N_R:]
            M_RC = MXi[N_R:,:N_R]
        else:
            raise ValueError(f"Not implemented for order '{self.composite_order}'")

        # Step 2: Compute excess coordinates and coupling vector
        C = self.dynamic_composite_excess_coordinates().flatten()
        b = M_RC.T @ C
        
        if correction:

            for i in range(5):
                # Step 3: First-order ground state calculation
                M_R_inv = np.linalg.inv(M_R)
                alpha_fo = -M_R_inv @ b

                xi_dynamic_excess_fo = np.concatenate((alpha_fo, C))
                # A_inv = np.linalg.inv(A)
                gs_fo = A_inv @ xi_dynamic_excess_fo

                # Step 4: Compute corrected transformation
                A, P = self.corrected_transformation_matrix(gs_fo)
                if self.iterative:
                    C = self.corr_excess.flatten()

                # Step 5: Recompute with corrected transformation
                A_inv = np.linalg.inv(A)
                MXi = A_inv.T @ stiffmat @ A_inv

                if self.composite_order == COMP_TRANSFORM_FRONT:
                    raise ValueError(f"Not implemented for order '{self.composite_order}'")

                elif self.composite_order == COMP_TRANSFORM_BACK:
                    M_R  = MXi[:N_R,:N_R]
                    M_C  = MXi[N_R:,N_R:]
                    M_RC = MXi[N_R:,:N_R]
                else:
                    raise ValueError(f"Not implemented for order '{self.composite_order}'")

                # Step 6: Update excess coordinates and coupling vector
                C = C - P
                b = M_RC.T @ C

        # Step 7: Final ground state calculation
        M_R_inv = np.linalg.inv(M_R)
        alpha = -M_R_inv @ b
        xi_dynamic_excess = np.concatenate((alpha, C))
        # A_inv = np.linalg.inv(A)
        Y_dynamic_excess = A_inv @ xi_dynamic_excess

        # Step 8: Compute free energy components
        F_const_C =  0.5 * C.T @ M_C @ C
        F_const_b = -0.5 * b.T @ M_R_inv @ b

        # entropy term
        n = len(M_R)
        logdet_sign, logdet = np.linalg.slogdet(M_R)
        F_pi = -0.5*n * np.log(2*np.pi)
        # matrix term
        F_mat = 0.5*logdet
        F_fluctuation = F_pi + F_mat
        sign, F_jacob = np.linalg.slogdet(A)
        
        # free energy of unconstrained DNA
        ff_logdet_sign, ff_logdet = np.linalg.slogdet(stiffmat)
        ff_pi = -0.5*len(stiffmat) * np.log(2*np.pi)
        F_free = 0.5*ff_logdet + ff_pi
        
        # prepare output with all intermediate values
        Fdict = {
            FE_ATTRIBUTE_KEY_F: F_fluctuation + F_jacob + F_const_C + F_const_b,
            FE_ATTRIBUTE_KEY_FFL  : F_fluctuation + F_jacob,
            FE_ATTRIBUTE_KEY_FEN : F_const_C + F_const_b,
            FE_ATTRIBUTE_KEY_FJC    : F_jacob,
            FE_ATTRIBUTE_KEY_FFR  : F_free,
            FE_ATTRIBUTE_KEY_DF          : F_fluctuation + F_jacob + F_const_C + F_const_b - F_free , 
            FE_ATTRIBUTE_KEY_GS         : Y_dynamic_excess,
        }
        return Fdict

    # ============================================================================================#
    #  free energy computation of hard constraint model with linear solve (non-optimized)
    # ============================================================================================#  

    def _hc_solve(self,stiffmat: np.ndarray, correction: bool = True):
        
        A = self.transformation_matrix()
        MXi = solve(A.T, solve(A.T, stiffmat.T).T)

        N  = len(MXi)
        N_C = len(self.replaced_ids) * self.DIM_PER_JUNCTION
        N_R = N-N_C

        if self.composite_order == COMP_TRANSFORM_FRONT:
            raise ValueError(f"Not implemented for order '{self.composite_order}'")
            # MF = MXi[:NF,:NF]
            # MC = MXi[NF:,NF:]
            # MM = MXi[NF:,:NF]

        elif self.composite_order == COMP_TRANSFORM_BACK:
            M_R  = MXi[:N_R,:N_R]
            M_C  = MXi[N_R:,N_R:]
            M_RC = MXi[N_R:,:N_R]
        
        else:
            raise ValueError(f"Not implemented for order '{self.composite_order}'")

        C = self.dynamic_composite_excess_coordinates().flatten()
        b = M_RC.T @ C

        if correction:

            alpha = -solve(M_R, b)
            xi_dynamic_excess = np.concatenate((alpha,C))
            Y_dynamic_excess = solve(A, xi_dynamic_excess)

            A, P = self.corrected_transformation_matrix(Y_dynamic_excess)
            MXi = solve(A.T, solve(A.T, stiffmat.T).T)

            if self.composite_order == COMP_TRANSFORM_FRONT:
                raise ValueError(f"Not implemented for order '{self.composite_order}'")
                # MF = MXi[:NF,:NF]
                # MC = MXi[NF:,NF:]
                # MM = MXi[NF:,:NF]

            elif self.composite_order == COMP_TRANSFORM_BACK:
                M_R  = MXi[:N_R,:N_R]
                M_C  = MXi[N_R:,N_R:]
                M_RC = MXi[N_R:,:N_R]
            else:
                raise ValueError(f"Not implemented for order '{self.composite_order}'")
            
            C -= P
            b = M_RC.T @ C

        alpha = -solve(M_R, b)
        xi_dynamic_excess = np.concatenate((alpha,C))
        Y_dynamic_excess = solve(A, xi_dynamic_excess)

        F_const_C =  0.5 * C.T @ M_C @ C
        F_const_b = 0.5 * b.T @ alpha

        # entropy term
        n = len(M_R)
        logdet_sign, logdet = np.linalg.slogdet(M_R)
        F_pi = -0.5*n * np.log(2*np.pi)
        # matrix term
        F_mat = 0.5*logdet
        F_fluctuation = F_pi + F_mat
        sign, F_jacob = np.linalg.slogdet(A)
        # F_jacob = F_jacob * sign
        
        # free energy of unconstrained DNA
        ff_logdet_sign, ff_logdet = np.linalg.slogdet(stiffmat)
        ff_pi = -0.5*len(stiffmat) * np.log(2*np.pi)
        F_free = 0.5*ff_logdet + ff_pi
        
        # prepare output
        Fdict = {
            FE_ATTRIBUTE_KEY_F: F_fluctuation + F_jacob + F_const_C + F_const_b,
            FE_ATTRIBUTE_KEY_FFL  : F_fluctuation + F_jacob,
            FE_ATTRIBUTE_KEY_FEN : F_const_C + F_const_b,
            FE_ATTRIBUTE_KEY_FJC    : F_jacob,
            FE_ATTRIBUTE_KEY_FFR  : F_free,
            FE_ATTRIBUTE_KEY_DF          : F_fluctuation + F_jacob + F_const_C + F_const_b - F_free , 
            FE_ATTRIBUTE_KEY_GS         : Y_dynamic_excess,
        }
        return Fdict

    # ==================================================================
    #  Optimized free energy computation of hard constraint model
    # ==================================================================

    def _hc_optimized(self, stiffmat: np.ndarray | csr_matrix, correction: bool = True):
        """One-shot optimized free energy computation.

        Performs matrix structure analysis and the actual computation in a
        single integrated pass — no separate ``ComputePlan`` is created.
        All algorithm decisions (banded Cholesky, block-bidiagonal D solvers,
        block-diagonal K extraction, block B^T multiplication) are made
        inline and applied immediately.

        Parameters
        ----------
        stiffmat : ndarray or sparse matrix
            Stiffness matrix (N×N).
        correction : bool
            Whether to apply the second-order correction pass.

        Returns
        -------
        dict  with keys *F*, *F_fluctuation*, *F_enthalpy*, *F_jacob*,
              *F_freedna*, *dF*, *gs*.
        """
        DIM = 6
        njuncs = self.njuncs
        N = njuncs * DIM
        composites = self.composites
        n_comp = len(composites)
        N_C = n_comp * DIM
        n_ret = njuncs - n_comp
        N_R = n_ret * DIM

        if self.composite_order != COMP_TRANSFORM_BACK:
            raise ValueError(
                f"Not implemented for order '{self.composite_order}'")

        # ---- 1. Dense K ------------------------------------------------
        K = stiffmat.toarray() if issparse(stiffmat) else stiffmat

        # ---- 2. Structural mappings (computed once) --------------------
        retained_ids = self.retained_ids
        ret_map = {rid: k * DIM for k, rid in enumerate(retained_ids)}
        rep_map = {comp.replaced_id: k * DIM
                   for k, comp in enumerate(composites)}
        ret_arr = np.asarray(retained_ids, dtype=np.intp)
        rep_arr = np.array([c.replaced_id for c in composites], dtype=np.intp)

        # DOF index arrays (for gs reconstruction)
        ret_dofs = np.empty(N_R, dtype=np.intp)
        rep_dofs = np.empty(N_C, dtype=np.intp)
        for i in range(n_ret):
            s = ret_arr[i] * DIM
            ret_dofs[i * DIM:(i + 1) * DIM] = np.arange(s, s + DIM)
        for i in range(n_comp):
            s = rep_arr[i] * DIM
            rep_dofs[i * DIM:(i + 1) * DIM] = np.arange(s, s + DIM)

        # ---- 3. K structure analysis (smart bandwidth) -----------------
        # Quick block-diag check: diagonal at offset DIM
        k_blkdiag = not np.any(K.diagonal(DIM))
        if k_blkdiag:
            # Find exact bw for banded Cholesky (0..DIM-1)
            bw = 0
            for d in range(1, DIM):
                if np.any(K.diagonal(d)):
                    bw = d
                else:
                    break
        else:
            # Binary search for banded vs dense Cholesky decision
            bw_limit = N // 4
            if bw_limit <= DIM or np.any(K.diagonal(min(bw_limit, N - 1))):
                bw = bw_limit          # bandwidth ≥ N//4 → dense cho
            else:
                lo, hi, bw = DIM, bw_limit - 1, DIM
                while lo <= hi:
                    mid = (lo + hi) // 2
                    if np.any(K.diagonal(mid)):
                        bw = mid
                        lo = mid + 1
                    else:
                        hi = mid - 1

        # ---- 4. K block extraction -------------------------------------
        if k_blkdiag:
            KRR_blk = np.empty((n_ret, DIM, DIM))
            for i in range(n_ret):
                s = ret_arr[i] * DIM
                KRR_blk[i] = K[s:s + DIM, s:s + DIM]
            KCC_blk = np.empty((n_comp, DIM, DIM))
            for i in range(n_comp):
                s = rep_arr[i] * DIM
                KCC_blk[i] = K[s:s + DIM, s:s + DIM]
            krc_zero = True
            kcc_blkdiag = True
            KRR = KRC = KCC = None
        else:
            KRR = K[np.ix_(ret_dofs, ret_dofs)]
            KRC = K[np.ix_(ret_dofs, rep_dofs)]
            KCC = K[np.ix_(rep_dofs, rep_dofs)]
            krc_zero = False       # non-block-diag K ⇒ KRC generally non-zero
            kcc_blkdiag = False    # non-block-diag K ⇒ KCC not block-diag
            KCC_blk = None

        # ---- 5. ff_logdet ----------------------------------------------
        if bw > 0 and bw < N // 4:
            ab = np.zeros((bw + 1, N), dtype=K.dtype)
            for d in range(bw + 1):
                ab[bw - d, d:] = K.diagonal(d)
            cb = sp.linalg.cholesky_banded(ab, lower=False,
                                           check_finite=False)
            ff_logdet = 2.0 * np.sum(np.log(cb[bw]))
        else:
            cho = sp.linalg.cho_factor(K, check_finite=False)
            ff_logdet = 2.0 * np.sum(np.log(np.diag(cho[0])))

        # ---- 6. D structure (structural check) -------------------------
        replaced_set = frozenset(rep_arr)
        d_bidiag = True
        for ci in range(n_comp):
            for jid in composites[ci].junction_ids:
                if jid in replaced_set:
                    ck = rep_map[jid] // DIM
                    if ck != ci and ck != ci - 1:
                        d_bidiag = False
                        break
            if not d_bidiag:
                break

        # ---- 7. Pre-compute CSR structure for fast B sparse construction --
        DIM2 = DIM * DIM
        _row_off = np.arange(DIM, dtype=np.intp).repeat(DIM)
        _col_off = np.tile(np.arange(DIM, dtype=np.intp), DIM)
        _b_blocks = []
        for ci, comp in enumerate(composites):
            rs = ci * DIM
            for jid in comp.junction_ids:
                if jid in ret_map:
                    _b_blocks.append((rs, ret_map[jid]))
        _nb = len(_b_blocks)
        _coo_r = np.empty(_nb * DIM2, dtype=np.intp)
        _coo_c = np.empty(_nb * DIM2, dtype=np.intp)
        for i, (rs, cs) in enumerate(_b_blocks):
            s = i * DIM2
            _coo_r[s:s + DIM2] = rs + _row_off
            _coo_c[s:s + DIM2] = cs + _col_off
        _B_shape = (N_C, N_R)

        # Build a template CSR from the known block positions (one-time)
        _B_tmpl = csr_matrix(
            (np.ones(_nb * DIM2, dtype=np.float64), (_coo_r, _coo_c)),
            shape=_B_shape)
        _B_tmpl.sort_indices()
        _csr_indptr = _B_tmpl.indptr
        _csr_indices = _B_tmpl.indices
        # Pre-compute (row, col) pairs in CSR data order for fast extraction
        _csr_nnz = _B_tmpl.nnz
        _csr_rows = np.empty(_csr_nnz, dtype=np.intp)
        for i in range(N_C):
            _csr_rows[_csr_indptr[i]:_csr_indptr[i + 1]] = i

        # ================================================================
        #  _run_pass: build B/D → fwd-sub → congruence
        # ================================================================
        def _run_pass(corrected, gs_fo_2d=None):
            # --- Build B, D_diag/D_sub (or full D) ---
            if d_bidiag:
                if corrected:
                    B, Dd, Ds, P = self._build_BD_direct(
                        ret_map, rep_map, corrected=True,
                        excess_dynamic_coordinates=gs_fo_2d,
                        d_block_bidiag=True)
                else:
                    B, Dd, Ds = self._build_BD_direct(
                        ret_map, rep_map, corrected=False,
                        d_block_bidiag=True)
                    P = None
                # Forward substitution: E = D^{-1} B
                Dinv = np.linalg.inv(Dd)
                E = np.empty_like(B)
                E[:DIM] = Dinv[0] @ B[:DIM]
                for k in range(1, n_comp):
                    s = k * DIM
                    E[s:s + DIM] = Dinv[k] @ (
                        B[s:s + DIM] - Ds[k - 1] @ E[s - DIM:s])
                d_st = (Dd, Ds, Dinv)
            else:
                if corrected:
                    B, D_full, P = self._build_BD_direct(
                        ret_map, rep_map, corrected=True,
                        excess_dynamic_coordinates=gs_fo_2d)
                else:
                    B, D_full = self._build_BD_direct(
                        ret_map, rep_map, corrected=False)
                    P = None
                lu_D, piv_D = sp.linalg.lu_factor(D_full)
                E = sp.linalg.lu_solve((lu_D, piv_D), B)
                d_st = (lu_D, piv_D)

            # --- KCC@E + D^{-T} backward sub → M_RC ---
            if d_bidiag:
                Dd_l, Ds_l, Dinv_l = d_st
                M_RC = np.empty_like(E)
                if kcc_blkdiag:
                    # Fused: block-diag KCC@E inside backward sweep
                    E_r = E.reshape(n_comp, DIM, -1)
                    last = n_comp - 1
                    s = last * DIM
                    kcc_e = KCC_blk[last] @ E_r[last]
                    rhs = (-kcc_e if krc_zero
                           else KRC[:, s:s + DIM].T - kcc_e)
                    M_RC[s:s + DIM] = Dinv_l[last].T @ rhs
                    for k in range(n_comp - 2, -1, -1):
                        s = k * DIM
                        kcc_e = KCC_blk[k] @ E_r[k]
                        rhs = (-kcc_e if krc_zero
                               else KRC[:, s:s + DIM].T - kcc_e)
                        M_RC[s:s + DIM] = Dinv_l[k].T @ (
                            rhs - Ds_l[k].T @ M_RC[s + DIM:s + 2 * DIM])
                else:
                    # Pre-compute: dense KCC@E as one BLAS call
                    rhs_all = -(KCC @ E)
                    if not krc_zero:
                        rhs_all += KRC.T
                    last = n_comp - 1
                    s = last * DIM
                    M_RC[s:s + DIM] = Dinv_l[last].T @ rhs_all[s:s + DIM]
                    for k in range(n_comp - 2, -1, -1):
                        s = k * DIM
                        M_RC[s:s + DIM] = Dinv_l[k].T @ (
                            rhs_all[s:s + DIM]
                            - Ds_l[k].T @ M_RC[s + DIM:s + 2 * DIM])
            else:
                # General LU path
                if kcc_blkdiag:
                    KCC_E = np.empty_like(E)
                    E_r = E.reshape(n_comp, DIM, -1)
                    for k in range(n_comp):
                        s = k * DIM
                        KCC_E[s:s + DIM] = KCC_blk[k] @ E_r[k]
                else:
                    KCC_E = KCC @ E
                rhs = (-KCC_E if krc_zero else KRC.T - KCC_E)
                lu_D, piv_D = d_st
                M_RC = sp.linalg.lu_solve(
                    (lu_D, piv_D), rhs, trans=1)

            # --- B^T @ M_RC via pre-built CSR template ---
            B_sp = csr_matrix(
                (B[_csr_rows, _csr_indices], _csr_indices, _csr_indptr),
                shape=_B_shape)
            BT_MRC = np.asarray(B_sp.T @ M_RC)

            # --- M_R construction ---
            if krc_zero:
                np.negative(BT_MRC, out=BT_MRC)
                if k_blkdiag:
                    for k in range(n_ret):
                        s = k * DIM
                        BT_MRC[s:s + DIM, s:s + DIM] += KRR_blk[k]
                else:
                    BT_MRC += KRR
                M_R = BT_MRC
            else:
                M_R = KRR - (KRC @ E) - BT_MRC

            return E, M_R, M_RC, d_st, P

        # ==================== PASS 1 ====================================
        E, M_R, M_RC, d_state, _ = _run_pass(corrected=False)

        C = self.dynamic_composite_excess_coordinates().flatten()
        b = M_RC.T @ C

        if correction:
            cho_MR = sp.linalg.cho_factor(M_R, check_finite=False)
            alpha_fo = -sp.linalg.cho_solve(cho_MR, b,
                                             check_finite=False)

            # D^{-1} C (vector forward substitution)
            if d_bidiag:
                _, Ds_1, Dinv_1 = d_state
                Dinv_C = np.empty(N_C)
                Dinv_C[:DIM] = Dinv_1[0] @ C[:DIM]
                for k in range(1, n_comp):
                    s = k * DIM
                    Dinv_C[s:s + DIM] = Dinv_1[k] @ (
                        C[s:s + DIM] - Ds_1[k - 1] @ Dinv_C[s - DIM:s])
            else:
                lu_1, piv_1 = d_state
                Dinv_C = sp.linalg.lu_solve((lu_1, piv_1), C)

            gs_fo = np.empty(N)
            gs_fo[ret_dofs] = alpha_fo
            gs_fo[rep_dofs] = Dinv_C - E @ alpha_fo

            # ==================== PASS 2 ================================
            E, M_R, M_RC, d_state, P = _run_pass(
                corrected=True, gs_fo_2d=gs_fo.reshape(-1, DIM))
            C = C - P
            b = M_RC.T @ C

        # ==================== FINAL SOLVE ================================
        cho_MR = sp.linalg.cho_factor(M_R, check_finite=False)
        alpha = -sp.linalg.cho_solve(cho_MR, b, check_finite=False)

        if d_bidiag:
            _, Ds_f, Dinv_f = d_state
            Dinv_C = np.empty(N_C)
            Dinv_C[:DIM] = Dinv_f[0] @ C[:DIM]
            for k in range(1, n_comp):
                s = k * DIM
                Dinv_C[s:s + DIM] = Dinv_f[k] @ (
                    C[s:s + DIM] - Ds_f[k - 1] @ Dinv_C[s - DIM:s])
        else:
            lu_f, piv_f = d_state
            Dinv_C = sp.linalg.lu_solve((lu_f, piv_f), C)

        gs = np.empty(N)
        gs[ret_dofs] = alpha
        gs[rep_dofs] = Dinv_C - E @ alpha

        # ==================== FREE ENERGY ================================
        if kcc_blkdiag:
            fc = 0.0
            dc = Dinv_C.reshape(n_comp, DIM)
            for k in range(n_comp):
                fc += dc[k] @ (KCC_blk[k] @ dc[k])
            F_const_C = 0.5 * fc
        elif KCC is not None:
            F_const_C = 0.5 * Dinv_C @ (KCC @ Dinv_C)
        else:
            F_const_C = 0.5 * Dinv_C @ (
                K[np.ix_(rep_dofs, rep_dofs)] @ Dinv_C)

        F_const_b = 0.5 * b @ alpha

        logdet_MR = 2.0 * np.sum(np.log(np.diag(cho_MR[0])))
        F_pi = -0.5 * N_R * np.log(2.0 * np.pi)
        F_fluct = F_pi + 0.5 * logdet_MR

        if d_bidiag:
            Dd_f = d_state[0]
            F_jacob = np.sum(np.log(np.abs(np.linalg.det(Dd_f))))
        else:
            lu_f = d_state[0]
            F_jacob = np.sum(np.log(np.abs(np.diag(lu_f))))

        ff_pi = -0.5 * N * np.log(2.0 * np.pi)
        F_free = 0.5 * ff_logdet + ff_pi

        F = F_fluct + F_jacob + F_const_C + F_const_b
        return {
            FE_ATTRIBUTE_KEY_F             : F,
            FE_ATTRIBUTE_KEY_FFL : F_fluct + F_jacob,
            FE_ATTRIBUTE_KEY_FEN    : F_const_C + F_const_b,
            FE_ATTRIBUTE_KEY_FJC       : F_jacob,
            FE_ATTRIBUTE_KEY_FFR     : F_free,
            FE_ATTRIBUTE_KEY_DF             : F - F_free,
            FE_ATTRIBUTE_KEY_GS            : gs,
        }

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------

    def _build_BD_direct(self, ret_jid_to_Bcol, rep_jid_to_Dcol,
                         corrected=False, excess_dynamic_coordinates=None,
                         d_block_bidiag=False):
        """Build B, D blocks directly without constructing full N×N matrix.

        Parameters
        ----------
        d_block_bidiag : bool
            When *True*, return D as ``(D_diag, D_sub)`` block arrays instead
            of a dense N_C×N_C matrix.  ``D_diag`` has shape (n_comp, 6, 6)
            and ``D_sub`` has shape (n_comp-1, 6, 6).
        """
        DIM = self.DIM_PER_JUNCTION
        n_comp = len(self.composites)
        N_C = n_comp * DIM
        N_R = (self.njuncs - n_comp) * DIM

        B = np.zeros((N_C, N_R), dtype=np.float64)
        P = np.zeros(N_C, dtype=np.float64) if corrected else None

        if d_block_bidiag:
            D_diag = np.zeros((n_comp, DIM, DIM), dtype=np.float64)
            D_sub  = np.zeros((max(n_comp - 1, 0), DIM, DIM),
                              dtype=np.float64)
        else:
            D = np.zeros((N_C, N_C), dtype=np.float64)

        if corrected:
            if (excess_dynamic_coordinates is not None
                    and excess_dynamic_coordinates.ndim == 1):
                excess_dynamic_coordinates = excess_dynamic_coordinates.reshape(
                    -1, DIM)

        for comp_idx, comp in enumerate(self.composites):
            row_s = comp_idx * DIM
            row_e = row_s + DIM

            if corrected:
                if self.iterative:
                    transforms, constant, corr_excess = comp.build_transforms_iterative_correction(
                        excess_dynamic_coordinates[comp.junction_ids])
                else:
                    transforms, constant = comp.build_corrected_transforms(
                        excess_dynamic_coordinates[comp.junction_ids])
                P[row_s:row_e] = constant
            else:
                transforms = comp.build_transforms()

            for junc_idx, junc_id in enumerate(comp.junction_ids):
                if junc_id in ret_jid_to_Bcol:
                    col_s = ret_jid_to_Bcol[junc_id]
                    B[row_s:row_e, col_s:col_s+DIM] = transforms[junc_idx]
                elif junc_id in rep_jid_to_Dcol:
                    if d_block_bidiag:
                        col_k = rep_jid_to_Dcol[junc_id] // DIM
                        if col_k == comp_idx:
                            D_diag[comp_idx] = transforms[junc_idx]
                        else:
                            # sub-diagonal: row comp_idx, col comp_idx-1
                            D_sub[comp_idx - 1] = transforms[junc_idx]
                    else:
                        col_s = rep_jid_to_Dcol[junc_id]
                        D[row_s:row_e, col_s:col_s+DIM] = transforms[junc_idx]

        if d_block_bidiag:
            if corrected:
                return B, D_diag, D_sub, P
            return B, D_diag, D_sub

        if corrected:
            return B, D, P
        return B, D

    @staticmethod
    def _logdet_symm(K):
        """log|det(K)| for SPD K.  Banded Cholesky when K is banded."""
        N = K.shape[0]
        bw = 0
        for d in range(1, min(N, 300)):
            if np.any(K.diagonal(d) != 0):
                bw = d
            else:
                break
        if bw > 0 and bw < N // 4:
            ab = np.zeros((bw + 1, N), dtype=K.dtype)
            for k in range(bw + 1):
                ab[bw - k, k:] = K.diagonal(k)
            try:
                cb = sp.linalg.cholesky_banded(ab, lower=False)
                return 2.0 * np.sum(np.log(cb[bw]))
            except np.linalg.LinAlgError:
                pass
        try:
            cho = sp.linalg.cho_factor(K)
            return 2.0 * np.sum(np.log(np.diag(cho[0])))
        except np.linalg.LinAlgError:
            _, logdet = np.linalg.slogdet(K)
            return logdet



if __name__ == "__main__":
    
    # Create static coordinates for 3 junctions
    
    # # generate random static coordinates for testing
    # X0_1 = np.random.uniform(-0.5*np.pi, 0.5*np.pi, size=(6,))  # junction 1
    # X0_2 = np.random.uniform(-0.5*np.pi, 0.5*np.pi, size=(6,))  # junction 2
    # X0_3 = np.random.uniform(-0.5*np.pi, 0.5*np.pi, size=(6,))  # junction 3
    
    
    X0_1 = np.zeros(6)
    X0_2 = np.zeros(6)
    X0_3 = np.zeros(6)

    rot_rge = 0.2*np.pi
    X0_1[:3] = np.random.uniform(-rot_rge, rot_rge, size=(3,)) 
    X0_2[:3] = np.random.uniform(-rot_rge, rot_rge, size=(3,)) 
    X0_3[:3] = np.random.uniform(-rot_rge, rot_rge, size=(3,)) 
    
    trans_rge = 1.0
    X0_1[3:] = np.random.uniform(-trans_rge, trans_rge, size=(3,))
    X0_2[3:] = np.random.uniform(-trans_rge, trans_rge, size=(3,))
    X0_3[3:] = np.random.uniform(-trans_rge, trans_rge, size=(3,))
    
    Yd_1 = np.zeros(6)
    Yd_2 = np.zeros(6)
    Yd_3 = np.zeros(6)
    
    rot_rge = 0.1
    Yd_1[:3] = np.random.uniform(-rot_rge, rot_rge, size=(3,)) 
    Yd_2[:3] = np.random.uniform(-rot_rge, rot_rge, size=(3,)) 
    Yd_3[:3] = np.random.uniform(-rot_rge, rot_rge, size=(3,)) 
    
    trans_rge = 0.1
    Yd_1[3:] = np.random.uniform(-trans_rge, trans_rge, size=(3,))
    Yd_2[3:] = np.random.uniform(-trans_rge, trans_rge, size=(3,))
    Yd_3[3:] = np.random.uniform(-trans_rge, trans_rge, size=(3,))
    
 