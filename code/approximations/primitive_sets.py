"""Primitive sets.

Each primitive set is defined as a tuple of tuples,
where each nested tuple defines a primitive function
and its corresponding arity.
"""
from functools import partial

from primitives import approximate as f_a
from primitives import standard as f

########################################################################
# Standard primitive sets.
########################################################################

# Standard transcendental primitive set.
standard = (
    (f.add, 2),
    (f.sub, 2),
    (f.mul, 2),
    (f.div, 2),
    (f.sin, 1),
    (f.cos, 1),
    (f.exp, 1),
    (f.log, 1),
    (f.sqrt, 1),
    (f.tanh, 1),
)

########################################################################
# MAD primitive sets.
########################################################################

# Fast math approximations.
# 
# Performance is defined based on the maximum number of
# multiply-add (MADD) operations needed by the primitives
# in the function set.
#
# The functions primarily causing a performance/accuracy 
# bottleneck are `sin`, `cos`, and `tanh`.
#
# The functions `add`, `sub`, and `mul` are not approximated 
# and are equivalent to the corresponding standard implementation, 
# since the standard implementation is trivial for hardware.


# MAD-16.
#
# Maximum number of MADD operations for this set: 16
mad_16 = (
    (f.add, 2),
    (f.sub, 2),
    (f.mul, 2),
    (partial(f_a.div, accuracy_level=4), 2),
    (partial(f_a.sin, accuracy_level=1), 1),
    (partial(f_a.cos, accuracy_level=1), 1),
    (partial(f_a.exp, accuracy_level=5), 1),
    (partial(f_a.log, accuracy_level=5), 1),
    (partial(f_a.sqrt, accuracy_level=4), 1),
    (partial(f_a.tanh, accuracy_level=3), 1),
)

# MAD-10.
#
# Maximum number of MADD operations for this set: 9
mad_10 = (
    (f.add, 2),
    (f.sub, 2),
    (f.mul, 2),
    (partial(f_a.div, accuracy_level=4), 2),
    (partial(f_a.sin, accuracy_level=1), 1),
    (partial(f_a.cos, accuracy_level=1), 1),
    (partial(f_a.exp, accuracy_level=3), 1),
    (partial(f_a.log, accuracy_level=3), 1),
    (partial(f_a.sqrt, accuracy_level=2), 1),
    (partial(f_a.tanh, accuracy_level=2), 1),
)

# MAD-04.
#
# Maximum number of MADD operations for this set: 4
mad_04 = (
    (f.add, 2),
    (f.sub, 2),
    (f.mul, 2),
    (partial(f_a.div, accuracy_level=1), 2),
    (partial(f_a.sin, accuracy_level=0), 1),
    (partial(f_a.cos, accuracy_level=0), 1),
    (partial(f_a.exp, accuracy_level=1), 1),
    (partial(f_a.log, accuracy_level=1), 1),
    (partial(f_a.sqrt, accuracy_level=1), 1),
    (partial(f_a.tanh, accuracy_level=0), 1),
)