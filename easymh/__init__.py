from collections import namedtuple
import numpy as np


__all__ = ["extract_options", "make_cube", "inf_cube", "incube",
           "uniform", "gaussian", 
           "individual", "collective", "rotative", "mh"]


Id = lambda x, *vargs, **kvargs: x
MESSAGE = "Starting point outside of the domain."
Roaming = namedtuple('Roaming', 'x walker objective')


def extract_options(options, prefix):
    """extract_options(dict(law=0, law_a=1, law_b=2, foo=3, foo_c=4), 'law') == {'a': 1, 'b': 2}"""
    return {k.replace(prefix+'_', ""):options[k] for k in options if k.find(prefix+'_')==0}


def make_cube(d, start=-np.inf, stop=np.inf):
    cube = np.zeros((d, 2))
    cube[:, 0] = start
    cube[:, 1] = stop
    return cube


def inf_cube(d):
    return make_cube(d)


def incube(x, cube):  # whether in the open cube
    x = np.array(x)
    cube = np.array(cube)
    return np.alltrue(cube[:, 0] < x) and np.alltrue(x < cube[:, 1])


def uniform(x, width=1, *vargs, seed=None, **kvargs):
    x = np.array(x)
    d = x.size
    if np.isscalar(width):
        width = width * np.ones(d)
    if seed is not None:
        np.random.seed(seed)
        
    return x + width * (np.random.rand(d) - 0.5)


def gaussian(x, sigma=1, *vargs, seed=None, **kvargs):
    x = np.array(x)
    d = x.size
    if np.isscalar(sigma):
        sigma = sigma * np.ones(d)
    if seed is not None:
        np.random.seed(seed)
    
    return x + sigma * (np.random.randn(d))


def individual(x, cube=None, law=uniform, *vargs, **options):
    x = np.array(x)
    d = x.size

    if cube is None:
        cube = inf_cube(d)
    cube = np.array(cube)
    
    if not incube(x, cube):
        raise Exception(MESSAGE)
        
    y = law(x, **extract_options(options, 'law'))
    index = (y < cube[:, 0]) | (y > cube[:, 1])
    y[index] = x[index]
    return y, None


def collective(x, cube=None, cov=1, *vargs, seed=None, **kvargs):  # multivariate-normal distribution
    x = np.array(x)
    d = x.size

    if cube is None:
        cube = inf_cube(d)
    cube = np.array(cube)
    if not incube(x, cube):
        raise Exception(MESSAGE)
    
    if np.isscalar(cov):
        cov = cov * np.ones(d)
    cov = np.array(cov)
    if cov.ndim == 1:
        cov = np.diag(cov)

    if seed is not None:
        np.random.seed(seed)    

    y = x + np.random.multivariate_normal(np.zeros(d), cov)
    y = y if incube(y, cube) else x
    return y, None


def rotative(x, t=0, state=None, cube=None, law=uniform, *vargs, seed=None, **options):  
    x = np.array(x, dtype='float64')   # caution: must specify the datatype, in-place opertion below
    d = x.size
        
    if cube is None:
        cube = inf_cube(d)
    cube = np.array(cube)
    
    if not incube(x, cube):
        raise Exception(MESSAGE)
    
    if seed is not None and t==0:
        np.random.seed(seed)

    r = t % d
    y = law([x[r]], **extract_options(options, 'law'))
    if cube[r, 0] < y < cube[r, 1]:
        x[r] = y
    return x, state


def mh(x, proba, cube=None, move='individual', ascdes=(Id, Id), picked=range(100, 1000, 1), seed=None, **options):
    x = np.array(x)
    d = len(x)
        
    if cube is None:
        cube = inf_cube(d)
    cube = np.array(cube)
    
    if not incube(x, cube):
        raise Exception(MESSAGE)

    dispatcher = dict(individual=individual, collective=collective, rotative=rotative)
    move = dispatcher.get(move, move)
        
    rng = np.random.RandomState(seed)
    
    N = picked[-1]
    walker = np.zeros((N+1, d))
    objective = np.zeros(N+1)
    walker[0, :] = x
    objective[0] = proba(x)
    
    _x = ascdes[0](x)
    px = proba(x)
    _cube = np.apply_along_axis(ascdes[0], 0, cube)
    
    state = None
    for t in range(N):
        _y, state2 = move(_x, t=t, state=state, cube=_cube, **extract_options(options, 'move'))
        y = ascdes[1](_y)
        py = proba(y)
        if rng.rand() < py / px:
            _x, x, px = _y, y, py
            state = state2
        
        walker[t+1, :] = x
        objective[t+1] = px

    return Roaming(np.mean(walker[picked, :], axis=0), walker, objective)
