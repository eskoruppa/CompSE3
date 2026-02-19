# COMPSE3

A python tool for contructing composite junctions in $\mathrm{SE}(3)$.

Allows for composites including midstep-junctions (left-hand and right-hand), placed in any location—not just the termini— and reverse sense traversal of junctions within the composite. Transformations from full junction to half-step and forward-sense to backward-sense of dynamic coordinates is automatically handled. The generated transformation (chance of basis) matrix acts on the coordinate vector of the full underlying junctions.  