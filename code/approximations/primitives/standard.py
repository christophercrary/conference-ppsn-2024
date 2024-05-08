"""Standard function primitives.

Standard implementations of function primitives, all of which are 
differentiable. 

Some protections are put in place for certain input conditions
depending on the function, but for simplicity and/or efficiency, 
not all invalid input conditions are always considered.

All primitives are defined to operate on iterables (e.g., NumPy 
arrays), where the function is meant to be applied to each element 
of the iterable independently.

Derivative calculations can be computed by way of the `differential` 
parameter. If `differential` is `None`, then the function is computed 
normally. Otherwise, for an arity-`m` function, if `differential == i`
for `0 <= i <= m - 1`, then the derivative of the function with respect 
to the `i`-th input is computed.
"""
import numpy as np

def add(X0, X1, primal=None, differential=None):
    """Return result of addition."""
    if len(X0) == 1 and len(X1) > 1:
        X0 = np.broadcast_to(X0, (len(X1),))
    elif len(X1) == 1 and len(X0) > 1:
        X1 = np.broadcast_to(X1, (len(X0),))
    res = np.empty((len(X0),), dtype=X0.dtype)
    if differential is None:
        res[:] = X0 + X1
    elif differential == 0 or differential == 1:
        res[:] = 1
    else:
        raise ValueError(f'differential={differential} is an invalid setting.')
    return res

def cos(X, primal=None, differential=None):
    """Return result of cosine."""
    res = np.empty((len(X),), dtype=X.dtype)
    if differential is None:
        res[:] = np.cos(X)
    elif differential == 0:
        res[:] = -np.sin(X)
    else:
        raise ValueError(f'differential={differential} is an invalid setting.')
    return res

def div(X0, X1, primal=None, differential=None):
    """Return result of division."""
    if len(X0) == 1 and len(X1) > 1:
        X0 = np.broadcast_to(X0, (len(X1),))
    elif len(X1) == 1 and len(X0) > 1:
        X1 = np.broadcast_to(X1, (len(X0),))
    res = np.empty((len(X0),), dtype=X0.dtype)
    if differential is None:
        res[:] = X0 / X1
    elif differential == 0:
        res[:] = 1 / X1
    elif differential == 1:
        # res[:] = -X0 / (X1 ** 2)
        res[:] = -primal * (1 / X1)
    else:
        raise ValueError(f'differential={differential} is an invalid setting.')
    return res

def exp(X, primal=None, differential=None): 
    """Return result of exponentiation, base `e`."""
    res = np.empty((len(X),), dtype=X.dtype)
    if differential is None:
        res[:] = np.exp(X)
    elif differential == 0:
        res[:] = primal
    else:
        raise ValueError(f'differential={differential} is an invalid setting.')
    return res

def inv(X, primal=None, differential=None):
    """Return result of inverse (i.e., reciprocal)."""
    res = np.empty((len(X),), dtype=X.dtype)
    if differential is None:
        res[:] = 1 / X
    elif differential == 0:
        # res[:] = -1 / X ** 2
        res[:] = -(primal ** 2)
    else:
        raise ValueError(f'differential={differential} is an invalid setting.')
    return res

def isqrt(X, primal=None, differential=None):
    """Return result of inverse square root."""
    res = np.empty((len(X),), dtype=X.dtype)
    if differential is None:
        res[:] = 1 / np.sqrt(X)
    elif differential == 0:
        # res[:] = -1 / (2 * np.sqrt(X ** 3))
        res[:] = primal * -(1 / (2 * X))
    else:
        raise ValueError(f'differential={differential} is an invalid setting.')
    return res

def log(X, primal=None, differential=None):
    """Return result of logarithm, base `e`."""
    res = np.empty((len(X),), dtype=X.dtype)
    if differential is None:
        res[:] = np.log(X)
    elif differential == 0:
        res[:] = 1 / X
    else:
        raise ValueError(f'differential={differential} is an invalid setting.')
    return res

def mul(X0, X1, primal=None, differential=None):
    """Return result of multiplication."""
    if len(X0) == 1 and len(X1) > 1:
        X0 = np.broadcast_to(X0, (len(X1),))
    elif len(X1) == 1 and len(X0) > 1:
        X1 = np.broadcast_to(X1, (len(X0),))
    res = np.empty((len(X0),), dtype=X0.dtype)
    if differential is None:
        res[:] = X0 * X1
    elif differential == 0:
        res[:] = X1
    elif differential == 1:
        res[:] = X0
    else:
        raise ValueError(f'differential={differential} is an invalid setting.')
    return res

def sin(X, primal=None, differential=None):
    """Return result of sine."""
    res = np.empty((len(X),), dtype=X.dtype)
    if differential is None:
        res[:] = np.sin(X)
    elif differential == 0:
        res[:] = np.cos(X)
    else:
        raise ValueError(f'differential={differential} is an invalid setting.')
    return res

def sqrt(X, primal=None, differential=None):
    """Return result of square root."""
    res = np.empty((len(X),), dtype=X.dtype)
    if differential is None:
        res[:] = np.sqrt(X)
    elif differential == 0:
        # res[:] = 1 / (2 * np.sqrt(X))
        res[:] = 1 / (2 * primal)
    else:
        raise ValueError(f'differential={differential} is an invalid setting.')
    return res
    
def sub(X0, X1, primal=None, differential=None):
    """Return result of subtraction."""
    if len(X0) == 1 and len(X1) > 1:
        X0 = np.broadcast_to(X0, (len(X1),))
    elif len(X1) == 1 and len(X0) > 1:
        X1 = np.broadcast_to(X1, (len(X0),))
    res = np.empty((len(X0),), dtype=X0.dtype)
    if differential is None:
        res[:] = X0 - X1
    elif differential == 0:
        res[:] = 1
    elif differential == 1:
        res[:] = -1
    else:
        raise ValueError(f'differential={differential} is an invalid setting.')
    return res

def tanh(X, primal=None, differential=None):
    """Return result of hyperbolic tangent."""
    res = np.empty((len(X),), dtype=X.dtype)
    if differential is None:
        res[:] = np.tanh(X)
    elif differential == 0:
        # res[:] = (1 - np.tanh(X) ** 2)
        res[:] = (1 - primal ** 2)
    else:
        raise ValueError(f'differential={differential} is an invalid setting.')
    return res