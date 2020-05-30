import numpy as np
from easymh import *


def test_extract_options():
    assert extract_options(dict(law=0, law_a=1, law_b=2, foo=3, foo_c=4), 'law') == {'a': 1, 'b': 2}
    
    
def test_incube():
    assert incube(2, [[1, 3]])
    assert incube([2], [[1, 3]])
    assert not incube([1,3], [[0, 2], [2, 3]])
    assert incube([1,3], [[0, 2], [2, 4]]) 
    
    
def test_uniform():
    assert np.isclose(uniform(0, seed=0), 0.0488135)
    assert np.isclose(uniform([0], seed=0), 0.0488135)
    assert np.allclose(uniform([0, 0], width=1, seed=0), [0.0488135 , 0.21518937])
    assert np.allclose(uniform([0, 0], [1, 10], seed=0), [0.0488135 , 2.15189366])


def test_gaussian():
    assert np.allclose(gaussian([0, 0], sigma=1, seed=0), [1.76405235, 0.40015721])
    assert np.allclose(gaussian([0, 0], sigma=10, seed=0), [17.64052346,  4.00157208])


def test_individual():
    assert np.allclose(individual([0, 0], cube=make_cube(2), law=uniform, law_width=5, law_seed=0)[0], [0.24406752, 1.07594683])
    assert np.allclose(individual([0, 0], cube=make_cube(2, -1, 1), law=uniform, law_width=5, law_seed=0)[0], [0.24406752, 0.])
    assert np.allclose(individual([0, 0], cube=make_cube(2), law=gaussian, law_sigma=5, law_seed=0)[0], [8.82026173, 2.00078604])


def test_collective():
    assert np.allclose(collective([0, 0], cube=make_cube(2), cov=1, seed=0)[0], [1.76405235, 0.40015721])
    assert np.allclose(collective([0, 0], cube=make_cube(2, -1, 1), cov=1, seed=0)[0], [0, 0])
    
    
def test_rotative():
    assert np.allclose(rotative([0, 0], t=2, cube=make_cube(2), law=uniform, law_seed=0)[0], [0.0488135, 0])


import plotly.graph_objects as go 
import plotly.express as px 
import plotly.io as pio
from plotly.subplots import make_subplots


def incircle(x, r=1):
    x = np.array(x)
    return 1 if np.sum(x**2)<=r**2 else 0


def subgaussian(x, sigma=1):
    x = np.array(x)
    return np.exp(-sigma*np.sum(x**2))


def whirl(x, t, state=None, **kvargs):
    x = np.array(x, dtype='float64')
    if state is None:
        r = 1
    else:
        r = state
    x[0] += r * np.cos(t/10)
    x[1] += r * np.sin(t/10)
    return x, r+1


a = mh([0,0], incircle, picked=range(0, 10000), move=collective, move_cov=.1)
fig = px.scatter(x=a.walker[:, 0], y=a.walker[:, 1], opacity=0.5)
fig.update_xaxes(scaleanchor="y", scaleratio=1)
fig.show()


a = mh([0,0], subgaussian, picked=range(0, 10000), move=collective, move_cov=1)
fig = px.scatter(x=a.walker[:, 0], y=a.walker[:, 1], opacity=0.2)
fig.update_xaxes(scaleanchor="y", scaleratio=1)
fig.show()


a = mh([0, 0], lambda x:1, move=collective, picked=range(10000))
fig = px.line(x=a.walker[:, 0], y=a.walker[:, 1])
fig.update_xaxes(scaleanchor="y", scaleratio=1)
fig.show()


a = mh([0, 0], lambda x:1, move=whirl)
fig = px.line(x=a.walker[:, 0], y=a.walker[:, 1])
fig.update_xaxes(scaleanchor="y", scaleratio=1)
fig.show()
