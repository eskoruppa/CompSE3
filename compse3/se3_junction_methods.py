"""Utilities to convert between 6D SE(3) coordinates and homogeneous transforms.

Conventions
-----------
- 6D vectors `X` are laid out as `[omega, v]` where `omega` is a rotation vector
    (axis-angle, radians) and `v` is a translation vector.
- Rotation-vector ↔ rotation-matrix conversions use `so3.euler2rotmat` /
    `so3.rotmat2euler` (the package's rotation-vector / Rodrigues convention).

Functions implement both direct SE(3) conversions and left/right-midpoint
representations (`glh`, `grh`) such that `g = glh @ grh`.
"""
from __future__ import annotations

import numpy as np
from .SO3 import so3
from .SO3.so3.pyConDec.pycondec import cond_jit

@cond_jit(nopython=True,cache=True)
def X2g(X: np.ndarray) -> np.ndarray:
    """Convert a 6D SE(3) coordinate vector to a 4x4 homogeneous matrix.

    Parameters
    ----------
    X : ndarray, shape (6,)
        6D vector `[omega, v]` where `omega` is the rotation vector
        (axis-angle, radians) and `v` is the translation.

    Returns
    -------
    g : ndarray, shape (4, 4)
        Homogeneous transformation matrix in SE(3). Rotation is placed in
        the top-left 3x3 block and translation in the top-right 3x1 column.
    """
    g = np.zeros((4, 4), dtype=np.float64)
    R = so3.euler2rotmat(X[:3])
    w = X[3:]
    g[:3, :3] = R
    g[:3, 3] = w
    g[3, 3] = 1.0
    return g

@cond_jit(nopython=True,cache=True)
def g2X(g: np.ndarray) -> np.ndarray:
    """Convert a 4x4 homogeneous transform to a 6D coordinate vector.

    Parameters
    ----------
    g : ndarray, shape (4, 4)
        Homogeneous transformation matrix.

    Returns
    -------
    X : ndarray, shape (6,)
        6D vector `[omega, v]` where `omega` is the rotation vector
        (axis-angle, radians) recovered from the rotation block and `v` is
        the translation (top-right column).
    """
    X = np.zeros((6,), dtype=np.float64)
    R = g[:3, :3]
    w = g[:3, 3]
    X[:3] = so3.rotmat2euler(R)
    X[3:] = w
    return X

@cond_jit(nopython=True,cache=True)
def X2glh(X: np.ndarray) -> np.ndarray:
    """Compute the left-midpoint homogeneous transform for a 6D vector.

    This returns `g_lh = exp(0.5 * xi)` in matrix form where `xi = [omega, v]`.
    For this left-midpoint representation the translation in `g_lh` is simply
    half the original translation.

    Parameters
    ----------
    X : ndarray, shape (6,)
        SE(3) coordinate vector `[omega, v]`.

    Returns
    -------
    glh : ndarray, shape (4,4)
        Left-midpoint homogeneous transform.
    """
    glh = np.zeros((4, 4), dtype=np.float64)
    glh[:3, :3] = so3.euler2rotmat(0.5 * X[:3])
    glh[:3, 3] = 0.5 * X[3:]
    glh[3, 3] = 1.0
    return glh

@cond_jit(nopython=True,cache=True)
def X2grh(X: np.ndarray) -> np.ndarray:
    """Compute the right-midpoint homogeneous transform for a 6D vector.

    The right-midpoint `g_rh` is chosen so that the full transform decomposes as
    `g = g_lh @ g_rh`. The rotation part is the half-rotation and the
    translation is expressed in the right (local) frame:
    `t_rh = 0.5 * R_half.T @ v`.

    Parameters
    ----------
    X : ndarray, shape (6,)
        SE(3) coordinate vector `[omega, v]`.

    Returns
    -------
    grh : ndarray, shape (4,4)
        Right-midpoint homogeneous transform.
    """
    grh = np.zeros((4, 4), dtype=np.float64)
    sqrtR = so3.euler2rotmat(0.5 * X[:3])
    grh[:3, :3] = sqrtR
    grh[:3, 3] = 0.5 * sqrtR.T @ X[3:]
    grh[3, 3] = 1.0
    return grh

@cond_jit(nopython=True,cache=True)
def glh2X(glh: np.ndarray) -> np.ndarray:
    """Recover the 6D SE(3) vector from a left-midpoint homogeneous matrix.

    Parameters
    ----------
    glh : ndarray, shape (4,4)
        Left-midpoint homogeneous transform (g_lh = exp(0.5*xi)).

    Returns
    -------
    X : ndarray, shape (6,)
        Reconstructed SE(3) vector `[omega, v]`.
    """
    X = np.zeros((6,), dtype=np.float64)
    R_lh = glh[:3, :3]
    w_lh = glh[:3, 3]
    X[:3] = 2 * so3.rotmat2euler(R_lh)
    X[3:] = 2 * w_lh
    return X

@cond_jit(nopython=True,cache=True)
def grh2X(grh: np.ndarray) -> np.ndarray:
    """Recover the 6D SE(3) vector from a right-midpoint homogeneous matrix.

    Parameters
    ----------
    grh : ndarray, shape (4,4)
        Right-midpoint homogeneous transform.

    Returns
    -------
    X : ndarray, shape (6,)
        Reconstructed SE(3) vector `[omega, v]`.

    Notes
    -----
    Because the right-midpoint stores the translation in the local frame,
    the global translation is recovered as `v = 2 * R_rh @ t_rh`.
    """
    X = np.zeros((6,), dtype=np.float64)
    R_rh = grh[:3, :3]
    w_rh = grh[:3, 3]
    X[:3] = 2 * so3.rotmat2euler(R_rh)
    X[3:] = 2 * R_rh @ w_rh
    return X
    
@cond_jit(nopython=True,cache=True)
def g2glh(g: np.ndarray) -> np.ndarray:
    """Compute left-midpoint `glh` directly from homogeneous matrix `g`.

    This is a thin wrapper: `g2glh(g) == X2glh(g2X(g))`.
    """
    return X2glh(g2X(g))

@cond_jit(nopython=True,cache=True)
def g2grh(g: np.ndarray) -> np.ndarray:
    """Compute right-midpoint `grh` directly from homogeneous matrix `g`.

    This is a thin wrapper: `g2grh(g) == X2grh(g2X(g))`.
    """
    return X2grh(g2X(g))

@cond_jit(nopython=True,cache=True)
def glh2g(glh: np.ndarray) -> np.ndarray:
    """Reconstruct full homogeneous matrix `g` from left-midpoint `glh`.

    This is a thin wrapper around `X2g(glh2X(glh))`.
    """
    return X2g(glh2X(glh))

@cond_jit(nopython=True,cache=True)
def grh2g(grh: np.ndarray) -> np.ndarray:
    """Reconstruct full homogeneous matrix `g` from right-midpoint `grh`.

    This is a thin wrapper around `X2g(grh2X(grh))`.
    """
    return X2g(grh2X(grh))

@cond_jit(nopython=True,cache=True)
def X2g_inv(X: np.ndarray) -> np.ndarray:
    g = np.zeros((4, 4), dtype=np.float64)
    R = so3.euler2rotmat(X[:3])
    w = X[3:]
    g[:3, :3] = R.T
    g[:3, 3] = -R.T @ w
    g[3, 3] = 1.0
    return g

@cond_jit(nopython=True,cache=True)
def X2glh_inv(X: np.ndarray) -> np.ndarray:
    glh = np.zeros((4, 4), dtype=np.float64)
    sqrtR = so3.euler2rotmat(0.5 * X[:3])
    glh[:3, :3] = sqrtR.T
    glh[:3, 3] = -0.5 * sqrtR.T @ X[3:]
    glh[3, 3] = 1.0
    return glh

@cond_jit(nopython=True,cache=True)
def X2grh_inv(X: np.ndarray) -> np.ndarray:
    grh = np.zeros((4, 4), dtype=np.float64)
    sqrtR = so3.euler2rotmat(0.5 * X[:3])
    grh[:3, :3] = sqrtR.T
    grh[:3, 3] = -0.5 * sqrtR.T @ sqrtR.T @ X[3:]
    grh[3, 3] = 1.0
    return grh

@cond_jit(nopython=True,cache=True)
def A_rev():
    return -np.eye(6)

@cond_jit(nopython=True,cache=True)
def A_lh(X0: np.ndarray) -> np.ndarray:
    """Compute the transformation matrix that converts dynamic junction coordinates corresponding to full junctions to dynamic left-hand half step junction coordinates.

    Parameters
    ----------
    X0 : ndarray, shape (6,)
        SE(3) coordinate vector at the reference point.

    Returns
    -------
    A : ndarray, shape (6, 6)
        Transformation matrix linearized around X0.
    """
    A = np.zeros((6, 6), dtype=np.float64)

    Jr_phihalf = so3.right_jacobian(0.5 * X0[:3])
    Jri_phi    = so3.inverse_right_jacobian(X0[:3])
    
    A[:3,:3] = Jr_phihalf @ Jri_phi
    A[3:,3:] = so3.euler2rotmat(0.5*X0[:3])
    return 0.5*A

@cond_jit(nopython=True,cache=True)
def A_rh(X0: np.ndarray) -> np.ndarray:
    """Compute the transformation matrix that converts dynamic junction coordinates corresponding to full junctions to dynamic right-hand half step junction coordinates.

    Parameters
    ----------
    X0 : ndarray, shape (6,)
        SE(3) coordinate vector at the reference point.

    Returns
    -------
    A : ndarray, shape (6, 6)
        Transformation matrix linearized around X0.
    """
    A = np.zeros((6, 6), dtype=np.float64)

    Jr_phihalf = so3.right_jacobian(0.5 * X0[:3])
    Jri_phi    = so3.inverse_right_jacobian(X0[:3])
    jac_prod = Jr_phihalf @ Jri_phi
    sqrtS = so3.euler2rotmat(0.5*X0[:3])
    S = sqrtS @ sqrtS
    shat = so3.hat_map(X0[3:])
 
    A[:3,:3] = jac_prod
    A[3:,3:] = np.eye(3)
    A[3:,:3] = 0.5 * S.T @ shat @ sqrtS @ jac_prod
    return 0.5*A    





if __name__ == "__main__":
    
    X = np.random.uniform(-np.pi,np.pi, size=(6,))
    nOm = np.linalg.norm(X[:3])
    if nOm > np.pi:
        X[:3] = X[:3] * (1.0/nOm)
    
    
    gt = so3.se3_euler2rotmat(X)
    
    
    g = X2g(X)
    if not np.allclose(g, gt):
        print('X2g failed')
        print("g:\n", g)
        print("gt:\n", gt)
    else:
        print('X2g passed')
    
    X1 = g2X(g)
    if not np.allclose(X, X1):
        print('g2X failed')
        print("X:\n", X)
        print("X1:\n", X1)
    else:        
        print('g2X passed')
        
    glh = X2glh(X)
    grh = X2grh(X)
    
    g2 = glh @ grh
    
    if not np.allclose(g, g2):
        print('X2glh,X2grh failed')
        print("g:\n", g)
        print("g2:\n", g2)
    else:
        print('X2glh,X2grh passed')
        
    
    glh2 = g2glh(g)
    grh2 = g2grh(g)   
    
    if not np.allclose(glh, glh2):
        print('g2glh failed')
        print("glh:\n", glh)
        print("glh2:\n", glh2)
    else:
        print('g2glh passed')
        
    if not np.allclose(grh, grh2):
        print('g2grh failed')
        print("grh:\n", grh)
        print("grh2:\n", grh2)
    else:
        print('g2grh passed')
        
    g3 = glh2g(glh)
    g4 = grh2g(grh)
    
    if not np.allclose(g, g3):
        print('glh2g failed')
        print("g:\n", g)
        print("g3:\n", g3)
    else:
        print('glh2g passed')
        
    if not np.allclose(g, g4):
        print('grh2g failed')
        print("g:\n", g)
        print("g4:\n", g4)
    else:
        print('grh2g passed')
        
        
    A_rh(X)