"""Approximate program function primitives.

Approximations of certain standard functions, all of which are 
differentiable. These approximations were implemented to explore 
if more efficient computation can lend itself to similar results 
in an evolutionary run. In Python, these functions may not be
faster than the standard NumPy implementations, although 
realizations in lower-level languages like C/C++ or in hardware 
can be highly efficient.

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

Lastly, note that we assume that multiplying and dividing floating-point
values by two can be done by simply incrementing or decrementing the 
the relevant floating-point exponent; ultimately, this expectation is
incorporated into the estimates given for the number of multiply-add 
(MADD) operations specified for each function. Some approximations 
may rely on other optimizations to avoid additional MADD operations.
"""
import numpy as np

def cos(X, primal=None, differential=None, accuracy_level=0):
    """Return result of cosine.
    
    Almost all logic for sine/cosine overlaps.

    See the following links for more details: 
    1. http://tinyurl.com/2u8nvb94
    2. http://tinyurl.com/tv7byxmk

    Number of multiply-adds (MADDs):
        - Accuracy level 0: 4 MADDs
        - Accuracy level 1: 10 MADDs

    Note that we can implement `res * (1 - abs(res))`
    with one MADD, rather than two, by implementing
    `res - res ** 2` with some additional logic for
    the sign of `res * abs(res)`.
    """
    res = np.empty((len(X),), dtype=X.dtype)
    if differential is None:
        if accuracy_level == 0:
            # The referenced approximation is for `-1/4 * sin(pi * x)`,
            # so we first reduce the frequency and change the phase
            # to approximate `1/4 * cos(x)`.
            res[:] = X / np.pi + 1.5
            # The referenced approximation expects inputs in the range 
            # `[-1, 1]`, so we must next perform range reduction.
            # Note that `-1 < res - np.fix(res) < 1`.
            res[:] = res * 0.5
            mask = (res > 0)
            res[mask] = 2 * (res[mask] - np.fix(res[mask])) - 1
            res[~mask] = 2 * (res[~mask] - np.fix(res[~mask])) + 1
            # Now, we approximate `1/4 * cos(x)` and scale by four,
            # so that we ultimately approximate `cos(x)`.
            res[:] = 4 * res * (1 - abs(res))
        elif accuracy_level == 1:
            # Perform approximation given by the VDT library:
            # http://tinyurl.com/yc287zcy.
            #
            # First, reduce input to [0, pi/4].
            DP1F = 0.78515625
            DP2F = 2.4187564849853515625e-4
            DP3F = 3.77489497744594108e-8
            x = np.abs(X)
            quad = (x * (4 / np.pi)).astype(np.int32)
            quad = (quad + 1) & (~1)
            y = quad.astype(np.float32)
            x = ((x - y * DP1F) - y * DP2F) - y * DP3F
            # Sign of cosine.
            quad = quad - 2
            sign_c = quad & 4
            # Sign of polynomial.
            sign_p = quad & 2
            # Compute sine or cosine depending on the polynomial sign.
            z = x ** 2
            mask = (sign_p == 0)
            # Compute sine for values in which the mask is true.
            res[mask] = (((
                (-1.9515295891e-4 * z[mask] + 8.3321608736e-3) 
                * z[mask] - 1.6666654611e-1) * z[mask] * x[mask]) 
                + x[mask])
            # Compute cosine for values in which the mask is false.
            res[~mask] = ((
                (2.443315711809948e-5 * z[~mask] - 1.388731625493765e-3) 
                * z[~mask] + 4.166664568298827e-2) * z[~mask] 
                * z[~mask] - 0.5 * z[~mask] + 1.0)
            # Flip signs, if necessary.
            mask = (sign_c == 0)
            res[mask] = res[mask] * -1
        # Handle certain input conditions.
        mask = (np.isnan(X)) | (np.isinf(X)) | (abs(X) > 33567376.0)
        # mask = (np.isnan(X)) | (np.isinf(X))
        res[mask] = np.nan
    elif differential == 0:
        res[:] = -sin(X, accuracy_level=accuracy_level)
    else:
        raise ValueError(f'differential={differential} is an invalid setting.')
    return res

def div(X0, X1, primal=None, differential=None, accuracy_level=0):
    """Return result of division.

    Number of multiply-adds (MADDs):
        - Accuracy level i (i >= 0): 2 * i + 1 MADDs
    """
    if len(X0) == 1 and len(X1) > 1:
        X0 = np.broadcast_to(X0, (len(X1),))
    elif len(X1) == 1 and len(X0) > 1:
        X1 = np.broadcast_to(X1, (len(X0),))
    res = np.empty((len(X0),), dtype=X0.dtype)
    if differential is None:
        res[:] = X0 * inv(X1, accuracy_level=accuracy_level)
        # Handle certain input conditions.
        mask = (np.isnan(X0) | np.isnan(X1) | ((X0 == 0) & (X1 == 0)) |
                (np.isinf(X0) & (abs(X1) > 1.602756e38)))
        res[mask] = np.nan
        mask = (
            ((X0 != 0) & (X0 < 0) & (X1 == 0) & (~np.signbit(X1))) 
            | ((X0 != 0) & (X0 > 0) & (X1 == 0) & (np.signbit(X1))))
        res[mask] = -np.inf
        mask = (
            ((X0 != 0) & (X0 < 0) & (X1 == 0) & (np.signbit(X1))) 
            | ((X0 != 0) & (X0 > 0) & (X1 == 0) & (~np.signbit(X1))))
        res[mask] = +np.inf
        mask = (X0 == 0) & (X1 != 0) & (~np.isnan(X1))
        res[mask] = 0
    elif differential == 0:
        res[:] = inv(X1, accuracy_level=accuracy_level)
    elif differential == 1:
        # res[:] = div(-X0, X1**2, accuracy_level=accuracy_level)
        res[:] = -primal * inv(X1, accuracy_level=accuracy_level)
    else:
        raise ValueError(f'differential={differential} is an invalid setting.')
    return res

def exp(X, primal=None, differential=None, accuracy_level=0):
    """Return result of exponentiation, base `e`.
    
    See the following links for more details:
    1. https://bit.ly/3NmkzWu
    2. https://bit.ly/3ChoAoC
    3. http://tinyurl.com/3cvwajck
    4. https://tinyurl.com/2vbdcvuc
    
    Number of multiply-adds (MADDs):
        - Accuracy level 0: 1 MADDs
        - Accuracy level 1: 4 MADDs
        - Accuracy level 2: 7 MADDs
        - Accuracy level 3: 8 MADDs
        - Accuracy level 4: 9 MADDs
        - Accuracy level 5: 15 MADDs
    """
    res = np.empty((len(X),), dtype=X.dtype)
    if differential is None:
        if accuracy_level == 0:
            # Degree-1 polynomial. (See the links given above.)
            res[:] = X * 12102203.161561485 + 1065054451
            res = res.astype(np.int32)
            res.dtype = np.float32
        elif accuracy_level >= 1 and accuracy_level <= 4:
            # We compute `exp(x)` by way of the equivalent formulation
            # `2 ** (x / log(2)) \approx 2 ** (x * 1.44269504)`.
            # 
            # Effectively, we split `t = x * 1.44269504` into an 
            # integer `i` and fraction `f` such that `t = i + f` 
            # and `0 <= f <= 1`. From this, we can compute 
            # `2 ** (x * 1.44269504) = (2 ** f) * (2 ** i)` by first
            # approximating `2 ** f` with a polynomial, and then by 
            # scaling with `2 ** i`, the latter of which can be 
            # performed simply by adding `i` to the exponent of the 
            # result of `2 ** f`.
            #
            # To compute `i` (and then `f`) as described above, one 
            # could compute `floor(t)`, which is equal to `trunc(t)` 
            # when `t > 0` or when `t` is a negative *integer*, and 
            # equal to `trunc(t) - 1` otherwise. We avoid directly 
            # computing `floor` by instead using an extension of
            # Schraudolph's algorithm: http://tinyurl.com/3cvwajck.
            # This allows for more efficient logic in hardware.
            #
            # `INV_LOG2_SHIFTED = (1 << 23) / log(2)`.
            INV_LOG2_SHIFTED = 12102203.0
            # `EXP2_NEG_23 = 2 ** -23`.
            EXP2_NEG_23 = 1.1920929e-7
            # Compute scaled input value.
            t = (INV_LOG2_SHIFTED * X).astype(np.int32)
            # Compute bit-shifted `i` value, `j`.
            # Specifically, `j = (int)(floor (x/log(2))) << 23`.
            # (Note that `t` is in two's-complement form, which
            # allows for the bitwise AND operation to compute
            # the correct floor operation when `i` is negative.)
            j = (t & 0xFF800000).astype(np.int32)
            t = (t - j).astype(np.float32)
            # `f = (x/log(2)) - floor(x/log(2))`.
            f = EXP2_NEG_23 * t
            res.dtype = np.int32
            res[:] = j
            # Compute `2 ** f`, where `0 <= f <= 1`.
            #
            # We choose the polynomial approximation
            # based on the specified accuracy level.
            if accuracy_level == 1:
                # Degree-2 polynomial.
                xf = (0.3371894346 * f + 0.657636276) * f + 1.00172476
            elif accuracy_level == 2:
                # Degree-5 polynomial.
                xf = (((((0.00189268149 * f + 0.00895538940) 
                        * f + 0.0558525427) * f + 0.240145453) 
                        * f + 0.693153934) * f + 0.999999917)
            elif accuracy_level == 3:
                # Degree-6 polynomial.
                xf = ((((((0.000221577741 * f + 0.00122991652) 
                        * f + 0.00969518396) * f + 0.0554745401) 
                        * f + 0.240231977) * f + 0.693146803)
                        * f + 1.00000000)
            elif accuracy_level == 4:
                # Degree-7 polynomial.
                xf = (((((((0.0000217349529 * f + 0.000142668753) 
                        * f + 0.00134347152) * f + 0.00961318205) 
                        * f + 0.0555054119) * f + 0.240226344)
                        * f + 0.693147187) * f + 1.00000000)
            # Scale `xf` by `2 ** i`.
            xf.dtype = np.int32
            res[:] += xf
            res.dtype = np.float32
        elif accuracy_level == 5:
            # Perform approximation given by the VDT library:
            # https://tinyurl.com/2vbdcvuc.
            LOG2_E = 1.44269504088896341
            x = X.astype(np.float32)
            z = np.floor(LOG2_E * x + 0.5)
            x[:] = x - 0.693359375 * z
            x[:] = x + 2.12194440e-4 * z
            n = z.astype(np.int32)
            n = (n + 127) << 23
            n.dtype = np.float32
            x2 = x ** 2
            res[:] = (((((1.9875691500e-4 * x + 1.3981999507e-3) 
                * x + 8.3334519073e-3) * x + 4.1665795894e-2)
                * x + 1.6666665459e-1) * x + 5.0000001201e-1)
            res[:] = res * x2 + x
            res[:] = res + 1
            res[:] = res * n
    elif differential == 0:
        res[:] = primal
    else:
        raise ValueError(f'differential={differential} is an invalid setting.')
    # Handle certain input conditions.
    mask = (X == 0)
    res[mask] = 1
    mask = (np.isnan(X))
    res[mask] = np.nan
    mask = (np.isneginf(X)) | (X < -88)
    res[mask] = 0
    mask = (np.isposinf(X)) | (X > 88.72283905206835)
    res[mask] = np.inf
    return res

def inv(X, primal=None, differential=None, accuracy_level=0):
    """Return result of inverse (i.e., reciprocal).
    
    See the following link for more details:
    https://bit.ly/42qbEHG.

    Number of multiply-adds (MADDs):
        - Accuracy level i (i >= 0): 2 * i MADDs
    """
    res = np.empty((len(X),), dtype=X.dtype)
    if differential is None:
        # Keep track of negative inputs.
        mask = (X < 0)
        # Make all inputs positive for the time being.
        res_ = X.astype(np.float32)
        res_[mask] = res_[mask] * -1
        # Approximation of `x ** (-1)`.
        res.dtype = np.int32
        res_.dtype = np.int32
        res[:] = (0x7EF127EA - res_).astype(np.int32)
        # Convert approximation back to floating-point.
        res.dtype = np.float32
        res_.dtype = np.float32
        # Further improve accuracy with Newton-Raphson iterations.
        for _ in range(accuracy_level):
            res[:] = res * (2 - res * res_)
        # Re-invert inputs that were originally negative.
        res[mask] = res[mask] * -1
        # Handle certain input conditions.
        mask = (X == 0) & (np.signbit(X))
        res[mask] = -np.inf
        mask = (X == 0) & (~np.signbit(X))
        res[mask] = np.inf
        # Handle certain input conditions.
        mask = (np.isnan(X))
        res[mask] = np.nan
        mask = (abs(X) > 1.602756e38)
        res[mask] = 0
    elif differential == 0:
        # res[:] = inv(-(X ** 2), accuracy_level=accuracy_level)
        res[:] = -(primal ** 2)
        # Handle certain input conditions.
        mask = (X == 0) & (np.signbit(X))
        res[mask] = np.inf
        mask = (X == 0) & (~np.signbit(X))
        res[mask] = -np.inf
        mask = (abs(X) > 1.266e19)
        res[mask] = 0
    else:
        raise ValueError(f'differential={differential} is an invalid setting.')
    # Handle certain input conditions.
    mask = (np.isnan(X))
    res[mask] = np.nan
    mask = (np.isinf(X))
    res[mask] = 0
    return res

def isqrt(X, primal=None, differential=None, accuracy_level=0):
    """Return result of inverse square root.
    
    See the following link for more details:
    https://en.wikipedia.org/wiki/Fast_inverse_square_root.
    
    Number of multiply-adds (MADDs):
        - Accuracy level i (i >= 0): 3 * i MADDs
    """
    res = np.empty((len(X),), dtype=X.dtype)
    if differential is None:
        # Half of `X`.
        X_ = X * 0.5
        # Extract the single-precision IEEE-754 encoded values of 
        # `X`, and convert the encoded value into integers.
        res[:] = X
        res.dtype = np.int32
        # Approximation of `x ** (-0.5)`.
        res[:] = 0x5F3759DF - (res >> 1)
        # Convert approximation back to floating-point.
        res.dtype = np.float32
        # Further improve accuracy with Newton-Raphson iterations.
        for _ in range(accuracy_level):
            res[:] = res * (1.5 - (X_ * (res ** 2)))
        # Handle certain input conditions.
        mask = (X == 0) & (~np.signbit(X))
        res[mask] = np.inf
    elif differential == 0:
        # res[:] = inv(-2 * sqrt(X ** 3))
        res[:] = primal * -inv((2 * X), accuracy_level=accuracy_level)
        # Handle certain input conditions.
        mask = (X == 0) & (~np.signbit(X))
        res[mask] = -np.inf
    else:
        raise ValueError(f'differential={differential} is an invalid setting.')
    # Handle certain input conditions.
    mask = (np.isnan(X)) | (X < 0)
    res[mask] = np.nan
    mask = (np.isposinf(X))
    res[mask] = 0
    return res

def log(X, primal=None, differential=None, accuracy_level=0):
    """Return result of logarithm, base `e`.

    See the following links for more details:
    1. https://bit.ly/3NmkzWu
    2. https://bit.ly/3ChoAoC
    3. http://tinyurl.com/yva9sv59
    4. http://tinyurl.com/2s4zecuf
    5. https://tinyurl.com/4aex3p2k
    
    Number of multiply-adds (MADDs):
        - Accuracy level 0: 1 MADDs
        - Accuracy level 1: 4 MADDs
        - Accuracy level 2: 7 MADDs
        - Accuracy level 3: 8 MADDs
        - Accuracy level 4: 9 MADDs
        - Accuracy level 5: 16 MADDs
    """
    res = np.empty((len(X),), dtype=X.dtype)
    if differential is None:
        if accuracy_level == 0:
            # Degree-1 polynomial. (See the links given above.)
            res_ = X.astype(np.float32)
            res_.dtype = np.int32
            res[:] = (res_ - 1065353217) * 8.262958405176314e-8
        elif accuracy_level >= 1 and accuracy_level <= 4:
            # The input is of the form `x = m * (2 ** e)`, where
            # `e` is the unbiased exponent value of the input, and
            # where `m` is `1.0 + m_` for mantissa `0 <= m_ < 1`.
            #
            # We compute `log(x)` by way of the equivalent
            # formulation `log(2) * e + log(m)`.
            LOG2 = 0.6931471806
            # Extract the exponent value of the input.
            x = X.astype(np.float32)
            x.dtype = np.uint32
            e = ((x >> 23).astype(np.int32) - 127).astype(np.float32)
            # Compute the first term of the given approximation.
            res[:] = LOG2 * e
            # Extract the `m` term, as defined above, by OR-ing
            # the floating-point representation of `1.0` with
            # the mantissa of the input.
            m = (0x7F << 23) | (x & 0x7FFFFF)
            m.dtype = np.float32
            # Compute `log(m)`, where `1 <= m < 2`.
            #
            # We choose the polynomial approximation
            # based on the specified accuracy level.
            if accuracy_level == 1:
                # Degree-2 polynomial.
                res += (-0.239030721 * m + 1.40339138) * m - 1.16093668
            elif accuracy_level == 2:
                # Degree-5 polynomial.
                res += (((((0.0308913737 * m - 0.287210575) 
                        * m + 1.12631109) * m - 2.45526047) 
                        * m + 3.52527699) * m - 1.94000044)
            elif accuracy_level == 3:
                # Degree-6 polynomial.
                res += ((((((-0.0170793339 * m + 0.184865198) 
                        * m - 0.859215646) * m + 2.24670209) 
                        * m - 3.67519809) * m + 4.22523802)
                        * m - 2.10531206)
            elif accuracy_level == 4:
                # Degree-7 polynomial.
                # Error gets worse... perhaps more rounding error?
                res += (((((((0.0102893135 * m - 0.125467594) 
                        * m + 0.669001960) * m - 2.04733924) 
                        * m + 3.97620707) * m - 5.16804080)
                        * m + 4.93256353) * m - 2.24721410)
        elif accuracy_level == 5:
            # Perform approximation given by the VDT library:
            # https://tinyurl.com/4aex3p2k.
            #
            # Separate the input `x` into a mantissa `m` and an exponent
            # `e` such that `0.5 <= abs(m) < 1.0`, and x = m * (2 ** e).
            # (An input of zero is a special condition for log.)
            # From this, we compute `log(x)` using the equality 
            # `log(x) = log(m) + log_e(2) * e`, where we use
            # a polynomial approximation to compute `log(m)`.
            m = X.astype(np.float32)
            m.dtype = np.int32
            # Initial value of exponent.
            e = ((m >> 23) - 127).astype(np.int32)
            # Desired representation of mantissa.
            # (We force the exponent to be equal to 
            # a value of 0.5 and copy the original mantissa.)
            m[:] = (m & 0x807FFFFF) | 0x3F000000
            m.dtype = np.float32
            # We alter the exponent/mantissa value to meet 
            # the definition given above.
            SQRTHF = 0.707106781186547524
            mask = (m > SQRTHF)
            e[mask] += 1
            e = e.astype(np.float32)
            m[mask] = m[mask] * 1 - 1
            m[~mask] = m[~mask] * 2 - 1
            # Now, we approximate `log(m)` with a polynomial.
            m2 = m ** 2
            res[:] = ((((((((7.0376836292e-2 * m + -1.1514610310e-1) 
                * m + 1.1676998740e-1) * m + -1.2420140846e-1)
                * m + 1.4249322787e-1) * m + -1.6668057665e-1)
                * m + 2.0000714765e-1) * m + -2.4999993993e-1)
                * m + 3.3333331174e-1)
            res[:] = res * m2
            res[:] = res * m
            res[:] = -2.12194440e-4 * e + res
            res[:] = -0.5 * m2 + res
            res[:] = res + m
            # Add the `log_e(2) * e` term.
            res[:] = res + 0.693359375 * e
        # Handle certain input conditions.
        mask = (X == 1)
        res[mask] = 0
        mask = (np.isnan(X)) | (X < 0)
        res[mask] = np.nan
        mask = (np.isposinf(X))
        res[mask] = +np.inf
        mask = (X == 0)
        res[mask] = -np.inf
    elif differential == 0:
        res[:] = inv(X, accuracy_level=accuracy_level)
    else:
        raise ValueError(f'differential={differential} is an invalid setting.')
    return res

def sin(X, primal=None, differential=None, accuracy_level=0):
    """Return result of sine.
    
    Almost all logic for sine/cosine overlaps.

    See the following link for more details: 
    http://tinyurl.com/2u8nvb94.

    Number of multiply-adds (MADDs):
        - Accuracy level 0: 4 MADDs
        - Accuracy level 1: 10 MADDs

    Note that we can multiply and divide by two by simply
    shifting the bits of the floating-point exponent.
    In addition, we implement `res * (1 - abs(res))`
    with one MADD, rather than two, by implementing
    `res - res ** 2` with some additional logic for
    the sign of `res * abs(res)`.
    """
    res = np.empty((len(X),), dtype=X.dtype)
    if differential is None:
        if accuracy_level == 0:
            # The referenced approximation is for `-1/4 * sin(pi * x)`,
            # so we first reduce the frequency and change the phase
            # to approximate `1/4 * sin(x)`.
            res[:] = X / np.pi + 1
            # The referenced approximation expects inputs in the range 
            # `[-1, 1]`, so we must next perform range reduction.
            # Note that `-1 < res - np.fix(res) < 1`.
            res[:] = res * 0.5
            mask = (res > 0)
            res[mask] = 2 * (res[mask] - np.fix(res[mask])) - 1
            res[~mask] = 2 * (res[~mask] - np.fix(res[~mask])) + 1
            # Now, we approximate `1/4 * sin(x)` and scale by four,
            # so that we ultimately approximate `sin(x)`.
            res[:] = 4 * res * (1 - abs(res))
        elif accuracy_level == 1:
            # Reduce input to [0, pi/4].
            DP1F = np.array([0.78515625,], dtype=np.float32)
            DP2F = np.array([2.4187564849853515625e-4,], dtype=np.float32)
            DP3F = np.array([3.77489497744594108e-8,], dtype=np.float32)
            ONEOPIO4F = np.array([4/np.pi,], dtype=np.float32)
            x = np.abs(X).astype(np.float32)
            quad = (x * ONEOPIO4F).astype(np.int32)
            quad = (quad + 1) & (~1)
            y = quad.astype(np.float32)
            x = ((x - y * DP1F) - y * DP2F) - y * DP3F
            # Sign of sine.
            sign_s = quad & 4
            # Sign of polynomial.
            quad = quad - 2
            sign_p = quad & 2
            # Compute sine or cosine depending on the polynomial sign.
            z = x ** 2
            mask = (sign_p == 0)
            # Compute cosine for values in which mask is true.
            res[mask] = ((
                (2.443315711809948e-5 * z[mask] - 1.388731625493765e-3) 
                * z[mask] + 4.166664568298827e-2) * z[mask] 
                * z[mask] - 0.5 * z[mask] + 1.0)
            # Compute sine for values in which mask is false.
            res[~mask] = (((
                (-1.9515295891e-4 * z[~mask] + 8.3321608736e-3) 
                * z[~mask] - 1.6666654611e-1) * z[~mask] * x[~mask]) 
                + x[~mask])
            # Flip signs, if necessary.
            mask = (sign_s != 0)
            res[mask] = res[mask] * -1
            mask = (X < 0)
            res[mask] = res[mask] * -1
        # Handle certain input conditions.
        mask = (np.isnan(X)) | (np.isinf(X)) | (abs(X) > 31875756.0)
        # mask = (np.isnan(X)) | (np.isinf(X))
        res[mask] = np.nan
    elif differential == 0:
        res[:] = cos(X, accuracy_level=accuracy_level)
    else:
        raise ValueError(f'differential={differential} is an invalid setting.')
    return res

def sqrt(X, primal=None, differential=None, accuracy_level=0):
    """Return result of square root.
    
    Number of multiply-adds (MADDs):
        - Accuracy level i (i >= 0): 3 * i + 1 MADDs
    """
    res = np.empty((len(X),), dtype=X.dtype)
    if differential is None:
        res[:] = X * isqrt(X, accuracy_level=accuracy_level)
        # Handle certain input conditions.
        mask = (np.isposinf(X))
        res[mask] = np.inf
        mask = (X == 0)
        res[mask] = 0
    elif differential == 0:
        res[:] = inv((2 * primal), accuracy_level=accuracy_level)
        # res[:] = 0.5 * isqrt(X, accuracy_level=accuracy_level)
    else:
        raise ValueError(f'differential={differential} is an invalid setting.')
    return res

def tanh(X, primal=None, differential=None, accuracy_level=0):
    """Return result of hyperbolic tangent.
    
    Number of multiply-adds (MADDs):
        - Accuracy level 0: 3 MADDs
        - Accuracy level 1: 5 MADDs
        - Accuracy level 2: 9 MADDs
        - Accuracy level 3: 16 MADDs
    """
    res = np.empty((len(X),), dtype=X.dtype)
    if differential is None:
        if accuracy_level == 0 or accuracy_level == 1:
            # Special polynomial approximation with 
            # more limited domain.
            mask = (X > -3) & (X < 3)
            res[mask] = (
                X[mask] * (8/3 * inv(
                    X[mask] ** 2 + 3, accuracy_level=accuracy_level) + 1/9))
            # Handle certain input conditions.
            mask = (np.isneginf(X)) | (X < -3)
            res[mask] = -1
            mask = (np.isposinf(X)) | (X > +3)
            res[mask] = +1
        else:
            # Mask for non-saturated inputs.
            mask = (X > -32) & (X < 32)
            if accuracy_level == 2:
                res[mask] = (
                    1 - 2 * (inv(exp(
                        2 * X[mask], accuracy_level=0) + 1, accuracy_level=3)))
            elif accuracy_level == 3:
                res[mask] = (
                    1 - 2 * (inv(exp(
                        2 * X[mask], accuracy_level=3) + 1, accuracy_level=3)))
            # Handle certain input conditions.
            mask = (np.isneginf(X)) | (X < -32)
            res[mask] = -1
            mask = (np.isposinf(X)) | (X > +32)
            res[mask] = +1
    elif differential == 0:
        # res[:] = (1 - tanh(X, accuracy_level=accuracy_level) ** 2)
        res[:] = (1 - primal ** 2)
    else:
        raise ValueError(f'differential={differential} is an invalid setting.')
    # Handle certain input conditions.
    mask = (np.isnan(X))
    res[mask] = np.nan
    return res