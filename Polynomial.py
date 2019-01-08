from some_libs import *
import numpy.polynomial.chebyshev as cheb
import numpy.polynomial.legendre as legend
# from pychebfun import *
from numpy.testing import assert_equal,assert_almost_equal,assert_raises

def cheb_fitcurve( x,y,order ):
    x = cheb.chebpts2(len(x))
    order = 64
    coef = legend.legfit(x, y, order);    assert_equal(len(coef), order+1)
    y1 = legend.legval(x, coef)
    err_1 = np.linalg.norm(y1-y) / np.linalg.norm(y)

    coef = cheb.chebfit(x, y, order);    assert_equal(len(coef), order + 1)
    thrsh = abs(coef[0]/1000)
    for i in range(len(coef)):
        if abs(coef[i])<thrsh:
            coef = coef[0:i+1]
            break

    y2 = cheb.chebval(x, coef)
    err_2 = np.linalg.norm(y2 - y) / np.linalg.norm(y)

    plt.plot(x, y2, '.')
    plt.plot(x, y, '-')
    plt.title("nPt={} order={} err_cheby={:.6g} err_legend={:.6g}".format(len(x),order,err_2,err_1))
    plt.show()
    assert_almost_equal(cheb.chebval(x, coef), y)
    #
    return coef


def test_chebfit( ):
    def f(x):
        return x * (x - 1) * (x - 2)

    # Test exceptions
    assert_raises(ValueError, cheb.chebfit, [1], [1], -1)
    assert_raises(TypeError, cheb.chebfit, [[1]], [1], 0)
    assert_raises(TypeError, cheb.chebfit, [], [1], 0)
    assert_raises(TypeError, cheb.chebfit, [1], [[[1]]], 0)
    assert_raises(TypeError, cheb.chebfit, [1, 2], [1], 0)
    assert_raises(TypeError, cheb.chebfit, [1], [1, 2], 0)
    assert_raises(TypeError, cheb.chebfit, [1], [1], 0, w=[[1]])
    assert_raises(TypeError, cheb.chebfit, [1], [1], 0, w=[1, 1])

    # Test fit
    x = np.linspace(0, 2)
    y = f(x)
    #
    coef3 = cheb.chebfit(x, y, 3)
    assert_equal(len(coef3), 4)
    assert_almost_equal(cheb.chebval(x, coef3), y)
    #
    coef4 = cheb.chebfit(x, y, 4)
    assert_equal(len(coef4), 5)
    assert_almost_equal(cheb.chebval(x, coef4), y)
    #
    coef2d = cheb.chebfit(x, np.array([y, y]).T, 3)
    assert_almost_equal(coef2d, np.array([coef3, coef3]).T)
    # test weighting
    w = np.zeros_like(x)
    yw = y.copy()
    w[1::2] = 1
    y[0::2] = 0
    wcoef3 = cheb.chebfit(x, yw, 3, w=w)
    assert_almost_equal(wcoef3, coef3)
    #
    wcoef2d = cheb.chebfit(x, np.array([yw, yw]).T, 3, w=w)
    assert_almost_equal(wcoef2d, np.array([coef3, coef3]).T)
    # test scaling with complex values x points whose square
    # is zero when summed.
    x = [1, 1j, -1, -1j]
    assert_almost_equal(cheb.chebfit(x, x, 1), [0, 1])

if __name__ == '__main__':
    test_chebfit()