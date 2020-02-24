import numpy as np
from easymh import move_elementwise, move_cohert, mh

zeros = np.zeros(3)
domain = np.array([[-1, 1], [-1, 1], [-1, 1]])


def test_move_elementwise():
    assert np.allclose(move_elementwise(zeros, 'u', domain, 3, 0), np.array([0.14644051, 0.6455681 , 0.30829013]))
    assert np.allclose(move_elementwise(zeros, 'u', domain, 3, 1), np.array([-0.24893399,  0.66097348,  0.]))
    assert np.allclose(move_elementwise(zeros, 'n', domain, 1, 0), np.array([0., 0.40015721, 0.97873798]))
    assert np.allclose(move_elementwise(zeros, 'n', domain, 1, 1), np.array([0., -0.61175641, -0.52817175]))
    assert np.allclose(move_elementwise([0], 'u', None, 1, 0), np.array([0.0488135]))
    

def test_move_cohert():
    assert np.allclose(move_cohert(zeros, domain, 1, 4), np.array([ 0.05056171,  0.49995133, -0.99590893]))
    
    
def test_mh():
    res, _ = mh(np.zeros(2), lambda x: np.exp(-np.dot(x, x)), N=10000)
    assert np.abs(res[0]) < 0.1 and np.abs(res[1]) < 0.1
    mh(np.zeros(1), lambda x: 1, move=lambda x, d, **kv: x+1)
    
