"""
Various different codings for mandelbrot calculations to explore performance
"""
try:
    import numba, numpy as np
    withnumba=True
except:
    withnumba = False

def get_calc_versions(func):
    """
    returns different optimisations on te given function in a list
    
    func: a straight python function

    returns array:
        0: the python version
        1: numba njit version
        2: numba njit version with parallel = True
        
    Both 1 & 2 can be None if that version not available
    """
    versions = [func, None, None]
    try:
        versions[1] = numba.njit(func)
    except:
        pass
    try:
        versions[2] = numba.njit(func, parallel=True)
    except:
        pass
    return versions

def m_calc(results, xbase, xtop, ybase, ytop, x_size, y_size, i_max):
    """
    basic loop  in straight python with not much optimisation
    
    Can be directly or can be optimized with numba jit
    """
    x_inc = (xtop-xbase) / x_size
    x_vals = [xbase + xi*x_inc for xi in range(x_size)]
    y_inc = (ytop-ybase) / y_size
    for y in numba.prange(y_size):
        yv = ybase + y*y_inc 
        for ix, xv in enumerate(x_vals):
            z = xv + yv * 1j
            c = z
            for i in range(i_max):
                if abs(z) > 2.0: break
                z = z * z + c
            results[ix,y] = i

def m_nsqu_calc(results, xbase, xtop, ybase, ytop, x_size, y_size, i_max):
    """
    check size of z without needing square root - useless in straight python.
    
    But the numba jit version is MUCH faster
    """
    x_inc = (xtop-xbase) / x_size
    x_vals = [xbase + xi*x_inc for xi in range(x_size)]
    y_inc = (ytop-ybase) / y_size
    for y in numba.prange(y_size):
        yv = ybase + y*y_inc 
        for ix, xv in enumerate(x_vals):
            z = xv + yv * 1j
            c = z
            for i in range(i_max):
                if z.imag*z.imag+z.real*z.real > 4: break
                z = z * z + c
            results[ix,y] = i

def i_numpy_calc32(results, x_range, y_range, i_max):
    """
    basic loop modded to use numpy data types to force length of floating point variables
    """
    for xi in numba.prange(x_range.shape[0]):
        for yi in range(y_range.shape[0]):
            z = np.csingle(x_range[xi] + y_range[yi]*1j)
            c = z
            for i in range(i_max):
                if z.imag*z.imag+z.real*z.real > 4: break
                z = z * z + c
            results[xi,yi] = i

_ji_numpy_calc32=None
_jpi_numpy_calc32=None
try:
    _ji_numpy_calc32 = numba.njit(i_numpy_calc32)
    _jpi_numpy_calc32 = numba.njit(i_numpy_calc32, parallel=True)
except:
    pass

def make_ranges(xbase, xtop, ybase, ytop, x_size, y_size, dtype):
    lims = np.array([xbase, xtop], dtype=dtype)
    x_range = np.linspace(lims[0], lims[1], x_size, endpoint = False)
    lims = np.array([ybase, ytop], dtype=dtype)
    y_range = np.linspace(lims[0], lims[1], y_size, endpoint = False)
    return x_range, y_range

def numpy_calc32(results, xbase, xtop, ybase, ytop, x_size, y_size, i_max):
    x_range, y_range = make_ranges(xbase, xtop, ybase, ytop, x_size, y_size, 'single')
    return i_numpy_calc32(results, x_range, y_range, i_max)

def numpy_j_calc32(results, xbase, xtop, ybase, ytop, x_size, y_size, i_max):
    x_range, y_range = make_ranges(xbase, xtop, ybase, ytop, x_size, y_size, 'single')
    return _ji_numpy_calc32(results, x_range, y_range, i_max)

def numpy_jp_calc32(results, xbase, xtop, ybase, ytop, x_size, y_size, i_max):
    x_range, y_range = make_ranges(xbase, xtop, ybase, ytop, x_size, y_size, 'single')
    return _jpi_numpy_calc32(results, x_range, y_range, i_max)

def i_numpy_calc64(results, x_range, y_range, i_max):
    """
    basic loop modded to use numpy data types to force length of floating point variables
    """
    for xi in numba.prange(x_range.shape[0]):
        for yi in range(y_range.shape[0]):
            z = np.cdouble(x_range[xi] + y_range[yi]*1j)
            c = z
            for i in range(i_max):
                if z.imag*z.imag+z.real*z.real > 4: break
                z = z * z + c
            results[xi,yi] = i

_ji_numpy_calc64=None
_jpi_numpy_calc64=None
try:
    _ji_numpy_calc64 = numba.njit(i_numpy_calc64)
    _jpi_numpy_calc64 = numba.njit(i_numpy_calc64, parallel=True)
except:
    pass

def numpy_calc64(results, xbase, xtop, ybase, ytop, x_size, y_size, i_max):
    x_range, y_range = make_ranges(xbase, xtop, ybase, ytop, x_size, y_size, 'double')
    return i_numpy_calc64(results, x_range, y_range, i_max)

def numpy_j_calc64(results, xbase, xtop, ybase, ytop, x_size, y_size, i_max):
    x_range, y_range = make_ranges(xbase, xtop, ybase, ytop, x_size, y_size, 'double')
    return _ji_numpy_calc64(results, x_range, y_range, i_max)

def numpy_jp_calc64(results, xbase, xtop, ybase, ytop, x_size, y_size, i_max):
    x_range, y_range = make_ranges(xbase, xtop, ybase, ytop, x_size, y_size, 'double')
    return _jpi_numpy_calc64(results, x_range, y_range, i_max)

def i_numpy_calc128(results, x_range, y_range, i_max):
    """
    basic loop modded to use numpy data types to force length of floating point variables
    """
    for xi in numba.prange(x_range.shape[0]):
        for yi in range(y_range.shape[0]):
            z = np.clongdouble(x_range[xi] + y_range[yi]*1j)
            c = z
            for i in range(i_max):
                if z.imag*z.imag+z.real*z.real > 4: break
                z = z * z + c
            results[xi,yi] = i

_ji_numpy_calc128=None
_jpi_numpy_calc128=None
#try:                   # numba not supporting float128 (yet?)
#    _ji_numpy_calc128 = numba.njit(i_numpy_calc128)
#    _jpi_numpy_calc128 = numba.njit(i_numpy_calc128, parallel=True)
#except:
#    pass

def numpy_calc128(results, xbase, xtop, ybase, ytop, x_size, y_size, i_max):
    x_range, y_range = make_ranges(xbase, xtop, ybase, ytop, x_size, y_size, 'longdouble')
    return i_numpy_calc128(results, x_range, y_range, i_max)

if _ji_numpy_calc128:
    def numpy_j_calc128(results, xbase, xtop, ybase, ytop, x_size, y_size, i_max):
        x_range, y_range = make_ranges(xbase, xtop, ybase, ytop, x_size, y_size, 'longdouble')
        return _ji_numpy_calc64(results, x_range, y_range, i_max)
else:
    numpy_j_calc128=None

if _jpi_numpy_calc128:
    def numpy_jp_calc128(results, xbase, xtop, ybase, ytop, x_size, y_size, i_max):
        x_range, y_range = make_ranges(xbase, xtop, ybase, ytop, x_size, y_size, 'longdouble')
        return _jpi_numpy_calc64(results, x_range, y_range, i_max)
else:
    numpy_jp_calc128=None


@numba.vectorize([numba.uint32(numba.float64,  numba.float64, numba.int64)])
def xcoremand(x_val,  y_val,   i_max):
    c = z = x_val + y_val*1j
    for i in range(i_max):
        if z.imag*z.imag+z.real*z.real > 4: break
        z = z * z + c
    return i

def _i_x_calc(results, x_range, y_range, i_max):
    for ix in numba.prange(x_range.shape[0]):
        results[ix] = xcoremand(x_range[ix], y_range, i_max)

_ij_x_calc = numba.jit(_i_x_calc)
_ijp_x_calc = numba.jit(_i_x_calc, parallel=True)

def xc_calc(results, xbase, xtop, ybase, ytop, x_size, y_size, i_max):
    x_range, y_range = make_ranges(xbase, xtop, ybase, ytop, x_size, y_size, 'double')
    return _i_x_calc(results, x_range, y_range, i_max)

if _ij_x_calc:
    def xc_j_calc(results, xbase, xtop, ybase, ytop, x_size, y_size, i_max):
        x_range, y_range = make_ranges(xbase, xtop, ybase, ytop, x_size, y_size, 'double')
        return _ij_x_calc(results, x_range, y_range, i_max)
else:
    xc_j_calc = None

if _ijp_x_calc:
    def xc_jp_calc(results, xbase, xtop, ybase, ytop, x_size, y_size, i_max):
        x_range, y_range = make_ranges(xbase, xtop, ybase, ytop, x_size, y_size, 'double')
        return _ijp_x_calc(results, x_range, y_range, i_max)
else:
    xc_jp_calc = None