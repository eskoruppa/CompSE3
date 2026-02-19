"""
CompSE3
=======

A python tool for constructing composite junctions in SE(3).

Allows for composites including midstep-junctions (left-hand and right-hand),
placed in any location—not just the termini— and reverse sense traversal of
junctions within the composite. Transformations from full junction to half-step
and forward-sense to backward-sense of dynamic coordinates is automatically
handled. The generated transformation (change of basis) matrix acts on the
coordinate vector of the full underlying junctions.

Public API
----------
Classes
~~~~~~~
SE3CompositeJunction
    Defines a single composite junction (ordered chain of elementary junctions
    each with a specified topology).
SE3CompositeTransform
    Builds the coordinate-transformation matrix for a full set of junctions that
    includes one or more composite junctions, and evaluates the hard-constraint
    free energy.
JunctionTopology (alias: Topo)
    Enum combining step-type (full / left-half / right-half) and traversal
    direction (forward / backward).
JunctionStepType
    Enum for the step type component of a topology.
TraversalDirection
    Enum for the traversal direction component of a topology.

SE(3) coordinate utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~
X2g, g2X
    Full SE(3) coordinate ↔ 4×4 homogeneous matrix.
X2glh, X2grh
    Left- / right-midpoint homogeneous matrices from a 6D coordinate vector.
X2g_inv, X2glh_inv, X2grh_inv
    Inverse transforms.
glh2X, grh2X
    Recover 6D coordinate from midpoint matrix.
g2glh, g2grh, glh2g, grh2g
    Convenience converters between matrix representations.
A_lh, A_rh, A_rev
    Linearised dynamic-coordinate conversion matrices.
"""

from .composite_junctions import (
    SE3CompositeJunction,
    JunctionTopology,
    JunctionStepType,
    TraversalDirection,
    Topo,
)

from .composite_transformation import SE3CompositeTransform

from .se3_junction_methods import (
    X2g,
    g2X,
    X2glh,
    X2grh,
    X2g_inv,
    X2glh_inv,
    X2grh_inv,
    glh2X,
    grh2X,
    g2glh,
    g2grh,
    glh2g,
    grh2g,
    A_lh,
    A_rh,
    A_rev,
)

__all__ = [
    # Composite classes
    "SE3CompositeJunction",
    "SE3CompositeTransform",
    # Topology enums
    "JunctionTopology",
    "Topo",
    "JunctionStepType",
    "TraversalDirection",
    # SE(3) utilities
    "X2g",
    "g2X",
    "X2glh",
    "X2grh",
    "X2g_inv",
    "X2glh_inv",
    "X2grh_inv",
    "glh2X",
    "grh2X",
    "g2glh",
    "g2grh",
    "glh2g",
    "grh2g",
    "A_lh",
    "A_rh",
    "A_rev",
]
