import numpy as np


__all__ = ["mh"]


Id = lambda x: x
MESSAGE = "Starting point outside of the domain."


def indomain(x, domain):
    return np.alltrue(domain[:, 0] < x) and np.alltrue(x < domain[:, 1])


def move_elementwise(x, law, domain=None, width=1, seed=None, **kvargs):
    x = np.array(x)
    d = len(x)
    
    if domain is None:
        domain = np.ones((d, 2))
        domain[:, 0] = -np.inf * domain[:, 0]
        domain[:, 1] = np.inf * domain[:, 1]
    domain = np.array(domain)
    
    if np.isscalar(width):
        width = width * np.ones(d)
        
    if seed is not None:
        np.random.seed(seed)
    
    if not indomain(x, domain):
        raise Exception(MESSAGE)
    
    if law == 'u':
        y = x + width * (np.random.rand(d) - 0.5)
    elif law == 'n':
        y = x + width * np.random.randn(d)
        
    index = (y < domain[:, 0]) | (y > domain[:, 1])
    y[index] = x[index]
    
    return y


def move_cohert(x, domain=None, cov=1, seed=None, **kvargs):
    x = np.array(x)
    d = len(x)
    
    if domain is None:
        domain = np.ones((d, 2))
        domain[:, 0] = -np.inf * domain[:, 0]
        domain[:, 1] = np.inf * domain[:, 1]
    domain = np.array(domain)
            
    if np.isscalar(cov):
        cov = np.diag(cov*np.ones(d))

    if seed is not None:
        np.random.seed(seed)
    
    if not indomain(x, domain):
        raise Exception(MESSAGE)
        
    y = x + np.random.multivariate_normal(np.zeros(d), cov)

    if not indomain(y, domain):
        y = x
    
    return y


def mh(x, proba, domain=None, N=500, B=200, move='n', ascdes=(Id, Id), seed=None, **kvargs):
    x = np.array(x)
    d = len(x)
    
    if domain is None:
        domain = np.ones((d, 2))
        domain[:, 0] = -np.inf * domain[:, 0]
        domain[:, 1] = np.inf * domain[:, 1]
    
    if not indomain(x, domain):
        raise Exception(MESSAGE)
    
    if seed is not None:
        np.random.seed(seed)
        
    walker = np.zeros((N+1, d))
    walker[0, :] = x
    
    px = proba(x)

    _x = ascdes[0](x)
    _domain = ascdes[0](domain)
    
    for i in range(N):
        if move in ['u', 'n']:
            _y = move_elementwise(_x, move, _domain, **kvargs)
        elif move == 'c':
            _y = move_cohert(_x, _domain, **kvargs)
        else:
            _y = move(_x, _domain, **kvargs)
        
        y = ascdes[1](_y)
        py = proba(y)
        if np.random.rand() < py / px:
            _x, x, px = _y, y, py
        
        walker[i+1, :] = x
        
    return np.mean(walker[B:, :], axis=0), walker
