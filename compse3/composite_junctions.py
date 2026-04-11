from __future__ import annotations

from enum import Enum
import sys
import numpy as np
# from scipy.sparse import csr_matrix, eye, lil_matrix
# from scipy.sparse import issparse

from .SO3 import so3
from .se3_junction_methods import A_lh, A_rev, A_rh, X2g, X2g_inv, X2glh, X2grh, X2glh_inv, X2grh_inv, g2X


################################################################################
### Topology Enum ##############################################################
################################################################################

class JunctionStepType(Enum):
    """Type of junction step traversal."""
    FULL = 'full'
    LEFT_HALF = 'lh'
    RIGHT_HALF = 'rh'


class TraversalDirection(Enum):
    """Direction of junction traversal."""
    FORWARD = 1
    BACKWARD = -1


class JunctionTopology(Enum):
    """Combined topology specification: step type × direction.
    
    Examples:
        - FULL_FWD: Full junction traversed forward (identity transformation)
        - FULL_BWD: Full junction traversed backward (inverse transformation)
        - LH_FWD: Left-half step traversed forward (0 → midpoint)
        - LH_BWD: Left-half step traversed backward (midpoint → 0)
        - RH_FWD: Right-half step traversed forward (midpoint → 1)
        - RH_BWD: Right-half step traversed backward (1 → midpoint)
    """
    FULL_FWD = (JunctionStepType.FULL, TraversalDirection.FORWARD)
    FULL_BWD = (JunctionStepType.FULL, TraversalDirection.BACKWARD)
    LH_FWD = (JunctionStepType.LEFT_HALF, TraversalDirection.FORWARD)
    LH_BWD = (JunctionStepType.LEFT_HALF, TraversalDirection.BACKWARD)
    RH_FWD = (JunctionStepType.RIGHT_HALF, TraversalDirection.FORWARD)
    RH_BWD = (JunctionStepType.RIGHT_HALF, TraversalDirection.BACKWARD)
    
    @property
    def step_type(self) -> JunctionStepType:
        """Get the step type component."""
        return self.value[0]
    
    @property
    def direction(self) -> TraversalDirection:
        """Get the traversal direction component."""
        return self.value[1]
    
    def is_full_step(self) -> bool:
        """Check if this is a full-step junction."""
        return self.step_type == JunctionStepType.FULL
    
    def is_half_step(self) -> bool:
        """Check if this is a half-step junction."""
        return self.step_type != JunctionStepType.FULL
    
    def is_left_half(self) -> bool:
        """Check if this is a left-half step junction."""
        return self.step_type == JunctionStepType.LEFT_HALF
    
    def is_right_half(self) -> bool:
        """Check if this is a right-half step junction."""
        return self.step_type == JunctionStepType.RIGHT_HALF
    
    def is_forward(self) -> bool:
        """Check if traversal is forward."""
        return self.direction == TraversalDirection.FORWARD
    
    def is_backward(self) -> bool:
        """Check if traversal is backward."""
        return self.direction == TraversalDirection.BACKWARD
    
    @staticmethod
    def from_string(topo_str: str) -> 'JunctionTopology':
        """Convert legacy string format to JunctionTopology enum.
        
        Parameters:
            topo_str: String in format '+g', '-g', '+lh', '-lh', '+rh', '-rh'
        
        Returns:
            Corresponding JunctionTopology enum value
        
        Raises:
            ValueError: If string is not a valid topology specification
        """
        mapping = {
            '+g': JunctionTopology.FULL_FWD,
            '-g': JunctionTopology.FULL_BWD,
            '+lh': JunctionTopology.LH_FWD,
            '-lh': JunctionTopology.LH_BWD,
            '+rh': JunctionTopology.RH_FWD,
            '-rh': JunctionTopology.RH_BWD,
            '+full': JunctionTopology.FULL_FWD,
            '-full': JunctionTopology.FULL_BWD,
            '+left_half': JunctionTopology.LH_FWD,
            '-left_half': JunctionTopology.LH_BWD,
            '+right_half': JunctionTopology.RH_FWD,
            '-right_half': JunctionTopology.RH_BWD,
        }
        if topo_str not in mapping:
            raise ValueError(f"Unknown topology string: '{topo_str}'. "
                           f"Valid values: {list(mapping.keys())}")
        return mapping[topo_str]
    
    @staticmethod
    def from_string_type_and_direction(type_str: str, dir: int | str) -> 'JunctionTopology':
        """Construct JunctionTopology from step type and traversal direction."""
        if isinstance(dir, str):
            dir = int(dir)
        if type_str in ['g', 'full']:
            if dir == 1:
                return JunctionTopology.FULL_FWD
            elif dir == -1:
                return JunctionTopology.FULL_BWD
        elif type_str in ['lh', 'left_half']:
            if dir == 1:
                return JunctionTopology.LH_FWD
            elif dir == -1:
                return JunctionTopology.LH_BWD
        elif type_str in ['rh', 'right_half']:
            if dir == 1:
                return JunctionTopology.RH_FWD
            elif dir == -1:
                return JunctionTopology.RH_BWD
        raise ValueError(f"Invalid type_str '{type_str}' or dir '{dir}'. "                         
                         f"Expected type_str in ['g', 'full', 'lh', 'left_half', 'rh', 'right_half'] and dir in [1, -1].")

    def to_string(self) -> str:
        """Convert to legacy string format for backward compatibility.
        
        Returns:
            String representation ('+g', '-g', '+lh', '-lh', '+rh', '-rh')
        """
        reverse_mapping = {
            JunctionTopology.FULL_FWD: '+g',
            JunctionTopology.FULL_BWD: '-g',
            JunctionTopology.LH_FWD: '+lh',
            JunctionTopology.LH_BWD: '-lh',
            JunctionTopology.RH_FWD: '+rh',
            JunctionTopology.RH_BWD: '-rh',
        }
        return reverse_mapping[self]
    

Topo = JunctionTopology  # Alias for convenience


################################################################################
### Composite Class ############################################################
################################################################################

class SE3CompositeJunction:
    
    JunctionTopology = JunctionTopology
    Topo = JunctionTopology

    def __init__(self, 
                 X0: list[np.ndarray] | np.ndarray,
                 junction_topologies: list[JunctionTopology] | list[str],
                 junction_ids: list[int] | np.ndarray | None = None,
                 key: str | int | None = None,
                 pose1: np.ndarray | None = None,
                 pose2: np.ndarray | None = None,
                 ):

        self.X0 = X0
        self.junction_ids = junction_ids
        self.key = key
        self._replaced_id = None
        
        # Convert strings to enums for backward compatibility
        self.junction_topologies = self._init_topology(junction_topologies)
        
        # Validate
        self._check_valid_topology(X0, self.junction_topologies, junction_ids)
        
        # build static components
        self._build_static_components()
        self._build_dynamic_conversions()
        self._build_accu_static_junctions()

        self.pose1 = pose1
        self.pose2 = pose2
        self._excess_junction = None
        self._excess_coordinates = None
        
    
    def set_replaced_id(self, replaced_id: int | str) -> None:
        self._replaced_id = replaced_id
    
    
    @property
    def replaced_id(self) -> int | str | None:
        return self._replaced_id
    
    
    @staticmethod
    def _init_topology(topologies: list[JunctionTopology] | list[str]) -> list[JunctionTopology]:
        normalized = []
        for topo in topologies:
            if isinstance(topo, JunctionTopology):
                normalized.append(topo)
            elif isinstance(topo, str):
                # Backward compatibility: convert string to enum
                normalized.append(JunctionTopology.from_string(topo))
            else:
                raise TypeError(f"Expected JunctionTopology enum or string, got {type(topo)}: {topo}")
        return normalized
    
    
    @staticmethod
    def _check_valid_topology(
        X0: list[np.ndarray] | np.ndarray,
        junction_topologies: list[JunctionTopology],
        junction_ids: list[int] | np.ndarray | None = None
        ) -> None:
        """Validate that topology specification is well-formed.
        
        Parameters:
            X0: Static coordinates of junctions
            junction_topologies: List of JunctionTopology enums
            junction_ids: Optional list of junction IDs
        
        Raises:
            ValueError: If X0 and junction_topologies have different lengths,
                       or if junction_ids is provided but has different length
        """
        if len(X0) != len(junction_topologies):
            raise ValueError(
                f"Length of X0 ({len(X0)}) must match "
                f"length of junction_topologies ({len(junction_topologies)})."
            )
        
        if junction_ids is not None and len(junction_ids) != len(junction_topologies):
            raise ValueError(
                f"Length of junction_ids ({len(junction_ids)}) must match "
                f"length of junction_topologies ({len(junction_topologies)})."
            )

        
    def _build_static_components(self) -> tuple[np.ndarray, np.ndarray]:

        self.static_junctions = np.zeros((len(self.X0), 4, 4), dtype=np.float64)
        self.static_junctions[:] = np.eye(4, dtype=np.float64)
        self.tail_static_junction = np.eye(4, dtype=np.float64)
    
        for i, (X0, topo) in enumerate(zip(self.X0, self.junction_topologies)):
            
            # Full steps
            if topo.is_full_step():  
                s = X2g(X0) if topo.is_forward() else X2g_inv(X0)            
            
            # Left half steps
            elif topo.is_left_half():
                s = X2glh(X0) if topo.is_forward() else X2glh_inv(X0)
                
            # Right half steps
            elif topo.is_right_half():
                s = X2grh(X0) if topo.is_forward() else X2grh_inv(X0)
            
            # Invalid topology (should not happen due to validation)
            else:
                raise ValueError(
                    f"Invalid topology {topo} for junction {self.key}." \
                    f"Expected JunctionTopology enum."
                )
            
            # assign static transforms to junctions according to direction
            if topo.is_forward():
                self.static_junctions[i] @= s
            else:
                if i < len(self.X0) - 1:
                    self.static_junctions[i+1] @= s
                else:
                    self.tail_static_junction @= s
        
        self.static_coordinates = np.zeros((len(self.X0), 6), dtype=np.float64)
        for i, s in enumerate(self.static_junctions):
            self.static_coordinates[i] = g2X(s)
        
        return self.static_junctions, self.tail_static_junction
    
    
    @property
    def static_composite_junction(self) -> np.ndarray:
        return self.accu_static[0]
        # if not hasattr(self, '_static_composite_junction'):
        #     saccu = np.eye(4, dtype=np.float64)
        #     for s in self.static_junctions:
        #         saccu = saccu @ s
        #     self._static_composite_junction = saccu
        # return self._static_composite_junction
    
    
    def dynamic_composite_excess_junction(self, 
                                          pose1: np.ndarray | None = None, 
                                          pose2: np.ndarray | None = None) -> np.ndarray:
        recal = False
        if pose1 is not None:
            self.pose1 = pose1
            recal = True
        if pose2 is not None:
            self.pose2 = pose2
            recal = True
        if recal or self._excess_junction is None:
            self._excess_junction = self._dynamic_composite_excess_junction(self.pose1,self.pose2)

        if self._excess_junction is None:
            raise ValueError(f'Either or both poses have not been provided yet.')
        return self._excess_junction
        
    def dynamic_composite_excess_coordinates(self, 
                                             pose1: np.ndarray | None = None, 
                                             pose2: np.ndarray | None = None) -> np.ndarray:
        recal = False
        if pose1 is not None:
            self.pose1 = pose1
            recal = True
        if pose2 is not None:
            self.pose2 = pose2
            recal = True
        if recal or self._excess_junction is None:
            self._excess_junction = self._dynamic_composite_excess_junction(self.pose1,self.pose2)
            self._excess_coordinates = g2X(self._excess_junction)
            return self._excess_coordinates

        if self._excess_junction is None:
            raise ValueError(f'Either or both poses have not been provided yet.')
        if self._excess_coordinates is None:
            self._excess_coordinates = g2X(self._excess_junction)
        return self._excess_coordinates


    def _dynamic_composite_excess_junction(self,
                                           pose1: np.ndarray, 
                                           pose2: np.ndarray) -> np.ndarray:
        if pose1.shape != (4,4):
            raise ValueError(f"pose1 must have shape [4, 4]. Got {pose1.shape}.")
        if pose2.shape != (4,4):
            raise ValueError(f"pose2 must have shape [4, 4]. Got {pose2.shape}.")
        return so3.se3_inverse(self.static_composite_junction) @ so3.se3_inverse(pose1) @ pose2 @ so3.se3_inverse(self.tail_static_junction)
    
    
    def _dynamic_composite_excess_coordinates(self, pose1: np.ndarray, pose2: np.ndarray) -> np.ndarray:
        excess_junction = self.dynamic_composite_excess_junction(pose1, pose2)
        return g2X(excess_junction)
    
    
    def _build_accu_static_junctions(self) -> np.ndarray:
        """Build accumulated static transforms for each junction."""
        self.accu_static = np.zeros((len(self.X0)+1, 4, 4), dtype=np.float64)
        self.accu_static[-1] = np.eye(4, dtype=np.float64)
        
        for l in range(len(self.X0)-1, -1, -1):
            self.accu_static[l] = self.static_junctions[l] @ self.accu_static[l+1]
        return self.accu_static 
    
    
    def _build_dynamic_conversions(self):
        self.dynamic_conversions = np.zeros((len(self.X0), 6, 6), dtype=np.float64)
        self.dynamic_conversions[:] = np.eye(6, dtype=np.float64)
        
        for l in range(len(self.X0)):
            topo = self.junction_topologies[l]
            if topo.is_backward():
                self.dynamic_conversions[l] @= A_rev()
                
            if topo.is_left_half():
                self.dynamic_conversions[l] @= A_lh(self.X0[l])
            
            if topo.is_right_half():
                self.dynamic_conversions[l] @= A_rh(self.X0[l]) 
        
        
    def build_transforms(self) -> np.ndarray:
        """Build the full set of transforms for this composite."""
        
        transforms = np.zeros((len(self.X0), 6, 6), dtype=np.float64)
        transforms[:] = np.eye(6, dtype=np.float64)
        
        for l in range(len(self.X0)):
            Saccu = self.accu_static[l+1][:3,:3]
            hatsaccu = so3.hat_map(self.accu_static[l+1][:3,3])
            A = np.zeros((6,6), dtype=np.float64)
            A[:3,:3] = Saccu.T
            A[3:,3:] = Saccu.T
            A[3:,:3] = -Saccu.T @ hatsaccu
            
            # dynamic_conversions takes care of converting the full forward-sense junctions into the correct half/full step and forward/backward sense
            transforms[l] @= A @ self.dynamic_conversions[l]
            
        return transforms
    
    
    def build_corrected_transforms(self, full_excess_dynamic_coordinates: np.ndarray) -> np.ndarray:
        # excess_dynamic_coordinates should be the dynamic coordinates calculated in the first iteration. 
        # These are the full junction coordinates in the forward sense. Conversion to the correct sense 
        # and half/full step is done internally in this function.
        
        if len(full_excess_dynamic_coordinates) != len(self.X0):
            raise ValueError(
                f"Length of full_excess_dynamic_coordinates ({len(full_excess_dynamic_coordinates)}) must match "
                f"number of junctions ({len(self.X0)})."
            )
        
        n = len(full_excess_dynamic_coordinates)
        i = 0
        j = n - 1
        I3 = np.eye(3, dtype=np.float64)
        
        # Convert the dynamic junction coordinates using dynamic_conversions
        excess_dynamic_coordinates = np.zeros(full_excess_dynamic_coordinates.shape, dtype=np.float64)
        for l in range(n):
            excess_dynamic_coordinates[l] = self.dynamic_conversions[l] @ full_excess_dynamic_coordinates[l]
        
        # init D_l^(c), R_l^(c), J_r for all junctions
        list_Phidc = excess_dynamic_coordinates[:,:3]
        list_Dlc = np.zeros((n, 3, 3), dtype=np.float64)
        list_Rlc = np.zeros((n, 3, 3), dtype=np.float64)
        list_Jr  = np.zeros((n, 3, 3), dtype=np.float64)
        for l in range(n):
            list_Dlc[l] = so3.euler2rotmat(list_Phidc[l])
            list_Rlc[l] = self.static_junctions[l,:3,:3] @ list_Dlc[l]
            list_Jr[l] = so3.right_jacobian(list_Phidc[l])
        
        # --- O(n) prefix products instead of O(n²) Rc_accu ---
        # Rc_prefix[k] = Rlc[0] @ Rlc[1] @ ... @ Rlc[k],  Rc_prefix[-1] = I
        # Then Rc_accu[l, k] = Rc_prefix[l-1].T @ Rc_prefix[k]  (for rotation matrices, inv = transpose)
        Rc_prefix = np.empty((n, 3, 3), dtype=np.float64)
        Rc_prefix[0] = list_Rlc[0]
        for k in range(1, n):
            Rc_prefix[k] = Rc_prefix[k-1] @ list_Rlc[k]

        # --- O(n) lambda_l via suffix sums ---
        # lambda_l[l] = Rc_prefix[l].T @ tail[l]
        # where tail[l] = sum(w_k for k=l+1..j)
        # and w_k = Rc_prefix[k-1] @ S[k,:3,:3] @ (D[k]-I) @ a[k+1,:3,3]
        lambda_l = np.zeros((n, 3), dtype=np.float64)
        tail = np.zeros(3, dtype=np.float64)
        for l in range(j - 1, -1, -1):
            k = l + 1
            Rp_km1 = Rc_prefix[k-1] if k >= 1 else I3
            inner = self.static_junctions[k,:3,:3] @ ((list_Dlc[k] - I3) @ self.accu_static[k+1,:3,3])
            tail = tail + Rp_km1 @ inner
            lambda_l[l] = Rc_prefix[l].T @ tail
        
        # compute corrected transforms for all i <= l <= j
        corrected_transforms = np.zeros((n, 6, 6), dtype=np.float64)
        
        Saccu_ij = self.accu_static[i][:3,:3]
        for l in range(n):
            Saccu_lp1 = self.accu_static[l+1][:3,:3]
            hatsaccu_lp1 = so3.hat_map(self.accu_static[l+1][:3,3])
            
            # Rc_accu[i,l] = Rc_prefix[l] (since i=0, Rc_prefix[-1]=I)
            Rp_il = Rc_prefix[l]
            # Rc_accu[i,l-1] = Rc_prefix[l-1] if l>0 else I
            Rp_ilm1 = Rc_prefix[l-1] if l > 0 else I3
            
            C_l = - Saccu_ij.T @ Rp_il @ (so3.hat_map(lambda_l[l]) + hatsaccu_lp1 ) @ list_Jr[l] 
            
            A = np.zeros((6,6), dtype=np.float64)
            A[:3,:3] = Saccu_lp1.T
            A[3:,3:] = Saccu_ij.T @ Rp_ilm1 @ self.static_junctions[l,:3,:3]
            A[3:,:3] = C_l
            
            corrected_transforms[l] = A @ self.dynamic_conversions[l]

        # compute constant part of transform
        sum_vec = np.zeros(3, dtype=np.float64)
        for l in range(i, j):
            Rp_il = Rc_prefix[l]
            sum1 = (I3 - list_Dlc[l].T) @ self.accu_static[l+1,:3,3]
            sum2 = (so3.hat_map(lambda_l[l]) + so3.hat_map(self.accu_static[l+1][:3,3])) @ list_Jr[l] @ list_Phidc[l]
            sum_vec += Rp_il @ (sum1 + sum2)
        
        constant_term = np.zeros(6, dtype=np.float64)
        constant_term[3:] = Saccu_ij.T @ sum_vec
        
        return corrected_transforms, constant_term
    


    def build_corrected_transforms_nonoptimized(self, full_excess_dynamic_coordinates: np.ndarray) -> np.ndarray:
        # excess_dynamic_coordinates should be the dynamic coordinates calculated in the first iteration. 
        # These are the full junction coordinates in the forward sense. Conversion to the correct sense 
        # and half/full step is done internally in this function.
        
        if len(full_excess_dynamic_coordinates) != len(self.X0):
            raise ValueError(
                f"Length of full_excess_dynamic_coordinates ({len(full_excess_dynamic_coordinates)}) must match "
                f"number of junctions ({len(self.X0)})."
            )
        
        i = 0
        j = len(full_excess_dynamic_coordinates)-1
        
        # Convert the dynamic junction coordinates to the correct sense and half/full step using the dynamic_conversions
        excess_dynamic_coordinates = np.zeros(full_excess_dynamic_coordinates.shape, dtype=np.float64)
        for l in range(len(full_excess_dynamic_coordinates)):
            excess_dynamic_coordinates[l] = self.dynamic_conversions[l] @ full_excess_dynamic_coordinates[l]
        
        # init D_l^(c) and R_l^(c) for all i <= l <= j
        list_Phidc = excess_dynamic_coordinates[:,:3]
        list_Dlc = np.zeros((len(excess_dynamic_coordinates), 3, 3), dtype=np.float64)
        list_Rlc = np.zeros((len(excess_dynamic_coordinates), 3, 3), dtype=np.float64)
        list_Jr  = np.zeros((len(excess_dynamic_coordinates), 3, 3), dtype=np.float64)
        for l in range(len(list_Phidc)):
            list_Dlc[l] = so3.euler2rotmat(list_Phidc[l])
            list_Rlc[l] = self.static_junctions[l,:3,:3] @ list_Dlc[l]
            list_Jr[l] = so3.right_jacobian(list_Phidc[l])
        
        # init R_[i,l]^(c) for all i <= l <= j
        # note that the last index is the one corresponding to l = i - 1 (i.e. the identity transform)
        # This can be conveniently accessed with index -1, but this means that -1 should never be interpreted as the last index (j)!!
        Rc_accu = np.zeros((len(excess_dynamic_coordinates),len(excess_dynamic_coordinates)+1, 3, 3), dtype=np.float64)
        for l in range(len(excess_dynamic_coordinates)):
            for k in range(len(excess_dynamic_coordinates)):
                if k < l:
                    Rc_accu[l,k] = np.eye(3, dtype=np.float64)
                elif k == l:
                    Rc_accu[l,k] = list_Rlc[l]
                else:
                    Rc_accu[l,k] = Rc_accu[l,k-1] @ list_Rlc[k]
            Rc_accu[l,-1] = np.eye(3, dtype=np.float64)
        
        # ########################################################################
        # # TESTING: REMOVE LATER
        # # this method is used to verify the correctness of the Rc_accu calculation by comparing it to a direct computation from the list_Rlc
        # def _Rc_accu_lk_test(l: int, k: int) -> np.ndarray:
        #     """Helper function to compute R_[l,k]^(c) for given l and k."""
        #     accu = np.eye(3, dtype=np.float64)
        #     for m in range(l, k+1):
        #         accu @= list_Rlc[m]
        #     return accu
        # ########################################################################
        
        # ########################################################################
        # # TESTING: REMOVE LATER
        # for l in range(len(self.X0)):
        #     for k in range(l, len(self.X0)):  # Only test valid ranges k >= l
        #         Rc_lk = _Rc_accu_lk_test(l,k)
        #         if not np.allclose(Rc_lk, Rc_accu[l,k], atol=1e-6):
        #             print(f"\n\nDiscrepancy in Rc_[{l},{k}]^(c) calculation!")
        #             print(f"Rc_[{l},{k}]^(c) (test):\n{Rc_lk}")
        #             print(f"Rc_[{l},{k}]^(c) (accu):\n{Rc_accu[l,k]}")
        #             print(f"Difference:\n{Rc_lk - Rc_accu[l,k]}")
        #             raise ValueError(f"Discrepancy in Rc_[{l},{k}]^(c) calculation exceeds tolerance!")
        #         else:
        #             print(f"Rc_[{l},{k}]^(c) calculation matches for l={l}, k={k}.")
            
        #     # Test the identity case for k = i-1 (stored at index -1)
        #     if not np.allclose(np.eye(3), Rc_accu[l,-1], atol=1e-6):
        #         print(f"Error: Rc_accu[{l},-1] should be identity!")
        #         raise ValueError(f"Rc_accu[{l},-1] is not identity matrix!")
        #     else:
        #         print(f"Rc_accu[{l},-1] correctly stores identity matrix.")
        # ########################################################################

        # init lambda_l for all i <= l <= j
        lambda_l = np.zeros((len(excess_dynamic_coordinates), 3), dtype=np.float64)
        for l in range(j+1):
            for k in range(l+1,j+1):
                lambda_l[l] += Rc_accu[l+1, k-1] @ self.static_junctions[k][:3,:3] @ (list_Dlc[k] - np.eye(3)) @ self.accu_static[k+1,:3,3] 
        
        # compute corrected transforms for all i <= l <= j
        corrected_transforms = np.zeros((len(excess_dynamic_coordinates), 6, 6), dtype=np.float64)
        corrected_transforms[:] = np.eye(6, dtype=np.float64)
        
        Saccu_ij = self.accu_static[i][:3,:3]
        for l in range(j+1):
            Saccu_lp1 = self.accu_static[l+1][:3,:3]
            hatsaccu_lp1 = so3.hat_map(self.accu_static[l+1][:3,3])
            C_l = - Saccu_ij.T @ Rc_accu[i,l] @ (so3.hat_map(lambda_l[l]) + hatsaccu_lp1 ) @ list_Jr[l] 
            
            A = np.zeros((6,6), dtype=np.float64)
            A[:3,:3] = Saccu_lp1.T
            A[3:,3:] = Saccu_ij.T @ Rc_accu[i,l-1] @ self.static_junctions[l,:3,:3]
            A[3:,:3] = C_l
            
            # dynamic_conversions takes care of converting the full forward-sense junctions into the correct half/full step and forward/backward sense
            corrected_transforms[l] @= A @ self.dynamic_conversions[l]

        # compute constant part of transform
        sum = np.zeros(3, dtype=np.float64)
        for l in range(i,j):
            sum1 = (np.eye(3,dtype=np.float64) - list_Dlc[l].T) @ self.accu_static[l+1,:3,3]
            sum2 = (so3.hat_map(lambda_l[l]) + so3.hat_map(self.accu_static[l+1][:3,3])) @ list_Jr[l] @ list_Phidc[l]
            sum += Rc_accu[i,l] @ (sum1 + sum2)
        
        constant_term = np.zeros(6, dtype=np.float64)
        constant_term[3:] = Saccu_ij.T @ sum
        
        return corrected_transforms, constant_term
        

    def build_transforms_iterative_correction(self, full_excess_dynamic_coordinates: np.ndarray) -> np.ndarray:
        """Build the full set of transforms for this composite."""
        
        if len(full_excess_dynamic_coordinates) != len(self.X0):
            raise ValueError(
                f"Length of full_excess_dynamic_coordinates ({len(full_excess_dynamic_coordinates)}) must match "
                f"number of junctions ({len(self.X0)})."
            )
        
        n = len(full_excess_dynamic_coordinates)
        # i = 0
        # j = n - 1
        # I3 = np.eye(3, dtype=np.float64)
        
        # Convert the dynamic junction coordinates using dynamic_conversions
        excess_dynamic_coordinates = np.zeros(full_excess_dynamic_coordinates.shape, dtype=np.float64)
        for l in range(n):
            excess_dynamic_coordinates[l] = self.dynamic_conversions[l] @ full_excess_dynamic_coordinates[l]
        
        # compute dsec's and ssec's for all i <= l <= j
        list_ssecs = np.zeros((len(self.X0), 4, 4), dtype=np.float64)
        list_dsecs = np.zeros((len(self.X0), 4, 4), dtype=np.float64)
        list_Jr    = np.zeros((len(self.X0), 3, 3), dtype=np.float64)
        list_Phidc = excess_dynamic_coordinates[:,:3]
        for l in range(len(self.X0)):
            rot_Ydc       = np.zeros(6, dtype=np.float64)
            rot_Ydc[:3]   = list_Phidc[l]
            list_dsecs[l] = so3.X2g(rot_Ydc)
            list_ssecs[l] = self.static_junctions[l] @ list_dsecs[l]
            list_Jr[l]    = so3.right_jacobian(list_Phidc[l])

        # compute accumulated static transforms for each junction (from l to j)
        list_accu_static_comp = np.zeros((len(self.X0)+1, 4, 4), dtype=np.float64)
        list_accu_static_comp[-1] = np.eye(4, dtype=np.float64)
        for l in range(len(self.X0)-1, -1, -1):
            list_accu_static_comp[l] = list_ssecs[l] @ list_accu_static_comp[l+1]


        transforms = np.zeros((len(self.X0), 6, 6), dtype=np.float64)
        transforms[:] = np.eye(6, dtype=np.float64)

        const_vec = np.zeros(6, dtype=np.float64)
        
        for l in range(len(self.X0)):
            Saccu_c = list_accu_static_comp[l+1][:3,:3]
            hatsaccu_c = so3.hat_map(list_accu_static_comp[l+1][:3,3])

            A = np.zeros((6,6), dtype=np.float64)
            A[:3,:3] = Saccu_c.T @ list_Jr[l]
            A[3:,3:] = Saccu_c.T @ list_dsecs[l,:3,:3].T
            A[3:,:3] = -Saccu_c.T @ hatsaccu_c @ list_Jr[l]
            
            # dynamic_conversions takes care of converting the full forward-sense junctions into the correct half/full step and forward/backward sense
            transforms[l] @= A @ self.dynamic_conversions[l]

            const_vec[:3] -= A[:3,:3] @ list_Phidc[l]
            const_vec[3:] -= A[3:,:3] @ list_Phidc[l]
            
        return transforms, const_vec
    