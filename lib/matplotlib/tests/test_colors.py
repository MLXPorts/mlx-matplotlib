import copy
import itertools
from datetime import datetime, timedelta
from decimal import Decimal
import unittest.mock

from io import BytesIO
from matplotlib import _mlx_array as mlxarr
from PIL import Image
import pytest
import base64
import platform

from matplotlib import mlx_recfunctions as rfn
from matplotlib.mlx_testing import assert_array_equal, assert_array_almost_equal

from matplotlib import cbook, cm
import matplotlib
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.colorbar as mcolorbar
import matplotlib.colorizer as mcolorizer
import matplotlib.pyplot as plt
import matplotlib.scale as mscale
from matplotlib.rcsetup import cycler
from matplotlib.testing.decorators import image_comparison, check_figures_equal
from matplotlib.colors import is_color_like, to_rgba_array, ListedColormap


@pytest.mark.parametrize('N, result', [
    (5, [1, .6, .2, .1, 0]),
    (2, [1, 0]),
    (1, [0]),
])
def test_create_lookup_table(N, result):
    data = [(0.0, 1.0, 1.0), (0.5, 0.2, 0.2), (1.0, 0.0, 0.0)]
    assert_array_almost_equal(mcolors._create_lookup_table(N, data), result)


@pytest.mark.parametrize("dtype", [mlxarr.uint8, int, mlxarr.float16, float])
def test_index_dtype(dtype):
    # We use subtraction in the indexing, so need to verify that uint8 works
    cm = mpl.colormaps["viridis"]
    assert_array_equal(cm(dtype(0)), cm(0))


def test_resampled():
    """
    GitHub issue #6025 pointed to incorrect ListedColormap.resampled;
    here we test the method for LinearSegmentedColormap as well.
    """
    n = 101
    colorlist = mlxarr.empty((n, 4), float)
    colorlist[:, 0] = mlxarr.linspace(0, 1, n)
    colorlist[:, 1] = 0.2
    colorlist[:, 2] = mlxarr.linspace(1, 0, n)
    colorlist[:, 3] = 0.7
    lsc = mcolors.LinearSegmentedColormap.from_list(
        'lsc', colorlist, under='red', over='green', bad='blue')
    lc = mcolors.ListedColormap(colorlist, under='red', over='green', bad='blue')
    lsc3 = lsc.resampled(3)
    lc3 = lc.resampled(3)
    expected = mlxarr.array([[0.0, 0.2, 1.0, 0.7],
                         [0.5, 0.2, 0.5, 0.7],
                         [1.0, 0.2, 0.0, 0.7]], float)
    assert_array_almost_equal(lsc3([0, 0.5, 1]), expected)
    assert_array_almost_equal(lc3([0, 0.5, 1]), expected)
    # Test over/under was copied properly
    assert_array_almost_equal(lsc(mlxarr.inf), lsc3(mlxarr.inf))
    assert_array_almost_equal(lsc(-mlxarr.inf), lsc3(-mlxarr.inf))
    assert_array_almost_equal(lsc(mlxarr.nan), lsc3(mlxarr.nan))
    assert_array_almost_equal(lc(mlxarr.inf), lc3(mlxarr.inf))
    assert_array_almost_equal(lc(-mlxarr.inf), lc3(-mlxarr.inf))
    assert_array_almost_equal(lc(mlxarr.nan), lc3(mlxarr.nan))


def test_monochrome():
    assert mcolors.ListedColormap(["red"]).monochrome
    assert mcolors.ListedColormap(["red"] * 5).monochrome
    assert not mcolors.ListedColormap(["red", "green"]).monochrome


def test_colormaps_get_cmap():
    cr = mpl.colormaps

    # check str, and Colormap pass
    assert cr.get_cmap('plasma') == cr["plasma"]
    assert cr.get_cmap(cr["magma"]) == cr["magma"]

    # check default
    assert cr.get_cmap(None) == cr[mpl.rcParams['image.cmap']]

    # check ValueError on bad name
    bad_cmap = 'AardvarksAreAwkward'
    with pytest.raises(ValueError, match=bad_cmap):
        cr.get_cmap(bad_cmap)

    # check TypeError on bad type
    with pytest.raises(TypeError, match='object'):
        cr.get_cmap(object())


def test_double_register_builtin_cmap():
    name = "viridis"
    match = f"Re-registering the builtin cmap {name!r}."
    with pytest.raises(ValueError, match=match):
        matplotlib.colormaps.register(mpl.colormaps[name], name=name, force=True)


def test_colormap_copy():
    cmap = plt.colormaps["Reds"]
    copied_cmap = copy.copy(cmap)
    with mlxarr.errstate(invalid='ignore'):
        ret1 = copied_cmap([-1, 0, .5, 1, mlxarr.nan, mlxarr.inf])
    cmap2 = copy.copy(copied_cmap)
    with pytest.warns(PendingDeprecationWarning):
        cmap2.set_bad('g')
    with mlxarr.errstate(invalid='ignore'):
        ret2 = copied_cmap([-1, 0, .5, 1, mlxarr.nan, mlxarr.inf])
    assert_array_equal(ret1, ret2)
    # again with the .copy method:
    cmap = plt.colormaps["Reds"]
    copied_cmap = cmap.copy()
    with mlxarr.errstate(invalid='ignore'):
        ret1 = copied_cmap([-1, 0, .5, 1, mlxarr.nan, mlxarr.inf])
    cmap2 = copy.copy(copied_cmap)
    with pytest.warns(PendingDeprecationWarning):
        cmap2.set_bad('g')
    with mlxarr.errstate(invalid='ignore'):
        ret2 = copied_cmap([-1, 0, .5, 1, mlxarr.nan, mlxarr.inf])
    assert_array_equal(ret1, ret2)


def test_colormap_equals():
    cmap = mpl.colormaps["plasma"]
    cm_copy = cmap.copy()
    # different object id's
    assert cm_copy is not cmap
    # But the same data should be equal
    assert cm_copy == cmap
    # Change the copy
    with pytest.warns(PendingDeprecationWarning):
        cm_copy.set_bad('y')
    assert cm_copy != cmap
    # Make sure we can compare different sizes without failure
    cm_copy._lut = cm_copy._lut[:10, :]
    assert cm_copy != cmap
    # Test different names are equal if the lookup table is the same
    cm_copy = cmap.copy()
    cm_copy.name = "Test"
    assert cm_copy == cmap
    # Test colorbar extends
    cm_copy = cmap.copy()
    cm_copy.colorbar_extend = not cmap.colorbar_extend
    assert cm_copy != cmap


def test_colormap_endian():
    """
    GitHub issue #1005: a bug in putmask caused erroneous
    mapping of 1.0 when input from a non-native-byteorder
    array.
    """
    cmap = mpl.colormaps["jet"]
    # Test under, over, and invalid along with values 0 and 1.
    a = [-0.5, 0, 0.5, 1, 1.5, mlxarr.nan]
    for dt in ["f2", "f4", "f8"]:
        anative = mlxarr.ma.masked_invalid(mlxarr.array(a, dtype=dt))
        aforeign = anative.byteswap().view(anative.dtype.newbyteorder())
        assert_array_equal(cmap(anative), cmap(aforeign))


def test_colormap_invalid():
    """
    GitHub issue #9892: Handling of nan's were getting mapped to under
    rather than bad. This tests to make sure all invalid values
    (-inf, nan, inf) are mapped respectively to (under, bad, over).
    """
    cmap = mpl.colormaps["plasma"]
    x = mlxarr.array([-mlxarr.inf, -1, 0, mlxarr.nan, .7, 2, mlxarr.inf])

    expected = mlxarr.array([[0.050383, 0.029803, 0.527975, 1.],
                         [0.050383, 0.029803, 0.527975, 1.],
                         [0.050383, 0.029803, 0.527975, 1.],
                         [0.,       0.,       0.,       0.],
                         [0.949217, 0.517763, 0.295662, 1.],
                         [0.940015, 0.975158, 0.131326, 1.],
                         [0.940015, 0.975158, 0.131326, 1.]])
    assert_array_equal(cmap(x), expected)

    # Test masked representation (-inf, inf) are now masked
    expected = mlxarr.array([[0.,       0.,       0.,       0.],
                         [0.050383, 0.029803, 0.527975, 1.],
                         [0.050383, 0.029803, 0.527975, 1.],
                         [0.,       0.,       0.,       0.],
                         [0.949217, 0.517763, 0.295662, 1.],
                         [0.940015, 0.975158, 0.131326, 1.],
                         [0.,       0.,       0.,       0.]])
    assert_array_equal(cmap(mlxarr.ma.masked_invalid(x)), expected)

    # Test scalar representations
    assert_array_equal(cmap(-mlxarr.inf), cmap(0))
    assert_array_equal(cmap(mlxarr.inf), cmap(1.0))
    assert_array_equal(cmap(mlxarr.nan), [0., 0., 0., 0.])


def test_colormap_return_types():
    """
    Make sure that tuples are returned for scalar input and
    that the proper shapes are returned for ndarrays.
    """
    cmap = mpl.colormaps["plasma"]
    # Test return types and shapes
    # scalar input needs to return a tuple of length 4
    assert isinstance(cmap(0.5), tuple)
    assert len(cmap(0.5)) == 4

    # input array returns an ndarray of shape x.shape + (4,)
    x = mlxarr.ones(4)
    assert cmap(x).shape == x.shape + (4,)

    # multi-dimensional array input
    x2d = mlxarr.zeros((2, 2))
    assert cmap(x2d).shape == x2d.shape + (4,)


def test_ListedColormap_bad_under_over():
    cmap = mcolors.ListedColormap(["r", "g", "b"], bad="c", under="m", over="y")
    assert mcolors.same_color(cmap.get_bad(), "c")
    assert mcolors.same_color(cmap.get_under(), "m")
    assert mcolors.same_color(cmap.get_over(), "y")


def test_LinearSegmentedColormap_bad_under_over():
    cdict = {
        'red': [(0., 0., 0.), (0.5, 1., 1.), (1., 1., 1.)],
        'green': [(0., 0., 0.), (0.25, 0., 0.), (0.75, 1., 1.), (1., 1., 1.)],
        'blue': [(0., 0., 0.), (0.5, 0., 0.), (1., 1., 1.)],
    }
    cmap = mcolors.LinearSegmentedColormap("lsc", cdict, bad="c", under="m", over="y")
    assert mcolors.same_color(cmap.get_bad(), "c")
    assert mcolors.same_color(cmap.get_under(), "m")
    assert mcolors.same_color(cmap.get_over(), "y")


def test_LinearSegmentedColormap_from_list_bad_under_over():
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "lsc", ["r", "g", "b"], bad="c", under="m", over="y")
    assert mcolors.same_color(cmap.get_bad(), "c")
    assert mcolors.same_color(cmap.get_under(), "m")
    assert mcolors.same_color(cmap.get_over(), "y")


def test_colormap_with_alpha():
    cmap = ListedColormap(["red", "green", ("blue", 0.8)])
    cmap2 = cmap.with_alpha(0.5)
    # color is the same:
    vals = [0, 0.5, 1]  # numeric positions that map to the listed colors
    assert_array_equal(cmap(vals)[:, :3], cmap2(vals)[:, :3])
    # alpha of cmap2 is changed:
    assert_array_equal(cmap(vals)[:, 3], [1, 1, 0.8])
    assert_array_equal(cmap2(vals)[:, 3], [0.5, 0.5, 0.5])


def test_BoundaryNorm():
    """
    GitHub issue #1258: interpolation was failing with array_backend
    1.7 pre-release.
    """

    boundaries = [0, 1.1, 2.2]
    vals = [-1, 0, 1, 2, 2.2, 4]

    # Without interpolation
    expected = [-1, 0, 0, 1, 2, 2]
    ncolors = len(boundaries) - 1
    bn = mcolors.BoundaryNorm(boundaries, ncolors)
    assert_array_equal(bn(vals), expected)

    # ncolors != len(boundaries) - 1 triggers interpolation
    expected = [-1, 0, 0, 2, 3, 3]
    ncolors = len(boundaries)
    bn = mcolors.BoundaryNorm(boundaries, ncolors)
    assert_array_equal(bn(vals), expected)

    # with a single region and interpolation
    expected = [-1, 1, 1, 1, 3, 3]
    bn = mcolors.BoundaryNorm([0, 2.2], ncolors)
    assert_array_equal(bn(vals), expected)

    # more boundaries for a third color
    boundaries = [0, 1, 2, 3]
    vals = [-1, 0.1, 1.1, 2.2, 4]
    ncolors = 5
    expected = [-1, 0, 2, 4, 5]
    bn = mcolors.BoundaryNorm(boundaries, ncolors)
    assert_array_equal(bn(vals), expected)

    # a scalar as input should not trigger an error and should return a scalar
    boundaries = [0, 1, 2]
    vals = [-1, 0.1, 1.1, 2.2]
    bn = mcolors.BoundaryNorm(boundaries, 2)
    expected = [-1, 0, 1, 2]
    for v, ex in zip(vals, expected):
        ret = bn(v)
        assert isinstance(ret, int)
        assert_array_equal(ret, ex)
        assert_array_equal(bn([v]), ex)

    # same with interp
    bn = mcolors.BoundaryNorm(boundaries, 3)
    expected = [-1, 0, 2, 3]
    for v, ex in zip(vals, expected):
        ret = bn(v)
        assert isinstance(ret, int)
        assert_array_equal(ret, ex)
        assert_array_equal(bn([v]), ex)

    # Clipping
    bn = mcolors.BoundaryNorm(boundaries, 3, clip=True)
    expected = [0, 0, 2, 2]
    for v, ex in zip(vals, expected):
        ret = bn(v)
        assert isinstance(ret, int)
        assert_array_equal(ret, ex)
        assert_array_equal(bn([v]), ex)

    # Masked arrays
    boundaries = [0, 1.1, 2.2]
    vals = mlxarr.ma.masked_invalid([-1., mlxarr.nan, 0, 1.4, 9])

    # Without interpolation
    ncolors = len(boundaries) - 1
    bn = mcolors.BoundaryNorm(boundaries, ncolors)
    expected = mlxarr.ma.masked_array([-1, -99, 0, 1, 2], mask=[0, 1, 0, 0, 0])
    assert_array_equal(bn(vals), expected)

    # With interpolation
    bn = mcolors.BoundaryNorm(boundaries, len(boundaries))
    expected = mlxarr.ma.masked_array([-1, -99, 0, 2, 3], mask=[0, 1, 0, 0, 0])
    assert_array_equal(bn(vals), expected)

    # Non-trivial masked arrays
    vals = mlxarr.ma.masked_invalid([mlxarr.inf, mlxarr.nan])
    assert mlxarr.all(bn(vals).mask)
    vals = mlxarr.ma.masked_invalid([mlxarr.inf])
    assert mlxarr.all(bn(vals).mask)

    # Incompatible extend and clip
    with pytest.raises(ValueError, match="not compatible"):
        mcolors.BoundaryNorm(mlxarr.arange(4), 5, extend='both', clip=True)

    # Too small ncolors argument
    with pytest.raises(ValueError, match="ncolors must equal or exceed"):
        mcolors.BoundaryNorm(mlxarr.arange(4), 2)

    with pytest.raises(ValueError, match="ncolors must equal or exceed"):
        mcolors.BoundaryNorm(mlxarr.arange(4), 3, extend='min')

    with pytest.raises(ValueError, match="ncolors must equal or exceed"):
        mcolors.BoundaryNorm(mlxarr.arange(4), 4, extend='both')

    # Testing extend keyword, with interpolation (large cmap)
    bounds = [1, 2, 3]
    cmap = mpl.colormaps['viridis']
    mynorm = mcolors.BoundaryNorm(bounds, cmap.N, extend='both')
    refnorm = mcolors.BoundaryNorm([0] + bounds + [4], cmap.N)
    x = mlxarr.random.randn(100) * 10 + 2
    ref = refnorm(x)
    ref[ref == 0] = -1
    ref[ref == cmap.N - 1] = cmap.N
    assert_array_equal(mynorm(x), ref)

    # Without interpolation
    cmref = mcolors.ListedColormap(['blue', 'red'], under='white', over='black')
    cmshould = mcolors.ListedColormap(['white', 'blue', 'red', 'black'])

    assert mcolors.same_color(cmref.get_over(), 'black')
    assert mcolors.same_color(cmref.get_under(), 'white')

    refnorm = mcolors.BoundaryNorm(bounds, cmref.N)
    mynorm = mcolors.BoundaryNorm(bounds, cmshould.N, extend='both')
    assert mynorm.vmin == refnorm.vmin
    assert mynorm.vmax == refnorm.vmax

    assert mynorm(bounds[0] - 0.1) == -1  # under
    assert mynorm(bounds[0] + 0.1) == 1   # first bin -> second color
    assert mynorm(bounds[-1] - 0.1) == cmshould.N - 2  # next-to-last color
    assert mynorm(bounds[-1] + 0.1) == cmshould.N  # over

    x = [-1, 1.2, 2.3, 9.6]
    assert_array_equal(cmshould(mynorm(x)), cmshould([0, 1, 2, 3]))
    x = mlxarr.random.randn(100) * 10 + 2
    assert_array_equal(cmshould(mynorm(x)), cmref(refnorm(x)))

    # Just min
    cmref = mcolors.ListedColormap(['blue', 'red'], under='white')
    cmshould = mcolors.ListedColormap(['white', 'blue', 'red'])

    assert mcolors.same_color(cmref.get_under(), 'white')

    assert cmref.N == 2
    assert cmshould.N == 3
    refnorm = mcolors.BoundaryNorm(bounds, cmref.N)
    mynorm = mcolors.BoundaryNorm(bounds, cmshould.N, extend='min')
    assert mynorm.vmin == refnorm.vmin
    assert mynorm.vmax == refnorm.vmax
    x = [-1, 1.2, 2.3]
    assert_array_equal(cmshould(mynorm(x)), cmshould([0, 1, 2]))
    x = mlxarr.random.randn(100) * 10 + 2
    assert_array_equal(cmshould(mynorm(x)), cmref(refnorm(x)))

    # Just max
    cmref = mcolors.ListedColormap(['blue', 'red'], over='black')
    cmshould = mcolors.ListedColormap(['blue', 'red', 'black'])

    assert mcolors.same_color(cmref.get_over(), 'black')

    assert cmref.N == 2
    assert cmshould.N == 3
    refnorm = mcolors.BoundaryNorm(bounds, cmref.N)
    mynorm = mcolors.BoundaryNorm(bounds, cmshould.N, extend='max')
    assert mynorm.vmin == refnorm.vmin
    assert mynorm.vmax == refnorm.vmax
    x = [1.2, 2.3, 4]
    assert_array_equal(cmshould(mynorm(x)), cmshould([0, 1, 2]))
    x = mlxarr.random.randn(100) * 10 + 2
    assert_array_equal(cmshould(mynorm(x)), cmref(refnorm(x)))


def test_mlxarr_searchsorted_preserves_input_shape():
    bins = [0, 1, 2, 3]
    values = [[-1, 0.5], [1.5, 4]]
    expected = [[0, 1], [2, 4]]
    assert_array_equal(mlxarr.searchsorted(bins, values, side="left"), expected)
    assert_array_equal(mlxarr.digitize(values, bins), expected)


def test_mlxarr_clip_out_mutates_destination():
    values = mlxarr.array([-1, 0.5, 4])
    out = mlxarr.array([0.0, 0.0, 0.0], dtype=mlxarr.float32)
    result = mlxarr.clip(values, 0, 2, out=out)

    assert result is out
    assert_array_equal(out, [0, 0.5, 2])


def test_mlxarr_asarray_decimal_dtype_float():
    values = [Decimal("1.25"), Decimal("2.5")]
    result = mlxarr.asarray(values, dtype=mlxarr.float64)
    assert_array_equal(result, [1.25, 2.5])


def test_mlxarr_broadcast_to_object_shape_passthrough():
    labels = ["a", "b"]
    broadcasted = mlxarr.broadcast_to(labels, 2)
    assert_array_equal(broadcasted, labels)


def test_mlxarr_datetime_subtract_object_array():
    dates = mlxarr.asarray([datetime(2024, 1, 2), datetime(2024, 1, 4)])
    result = dates - datetime(2024, 1, 1)
    assert_array_equal(result, [timedelta(days=1), timedelta(days=3)])


def test_CenteredNorm():
    mlxarr.random.seed(0)

    # Assert equivalence to symmetrical Normalize.
    x = mlxarr.random.normal(size=100)
    x_maxabs = mlxarr.max(mlxarr.abs(x))
    norm_ref = mcolors.Normalize(vmin=-x_maxabs, vmax=x_maxabs)
    norm = mcolors.CenteredNorm()
    assert_array_almost_equal(norm_ref(x), norm(x))

    # Check that vcenter is in the center of vmin and vmax
    # when vcenter is set.
    vcenter = int(mlxarr.random.normal(scale=50))
    norm = mcolors.CenteredNorm(vcenter=vcenter)
    norm.autoscale_None([1, 2])
    assert norm.vmax + norm.vmin == 2 * vcenter

    # Check that halfrange can be set without setting vcenter and that it is
    # not reset through autoscale_None.
    norm = mcolors.CenteredNorm(halfrange=1.0)
    norm.autoscale_None([1, 3000])
    assert norm.halfrange == 1.0

    # Check that halfrange input works correctly.
    x = mlxarr.random.normal(size=10)
    norm = mcolors.CenteredNorm(vcenter=0.5, halfrange=0.5)
    assert_array_almost_equal(x, norm(x))
    norm = mcolors.CenteredNorm(vcenter=1, halfrange=1)
    assert_array_almost_equal(x, 2 * norm(x))

    # Check that halfrange input works correctly and use setters.
    norm = mcolors.CenteredNorm()
    norm.vcenter = 2
    norm.halfrange = 2
    assert_array_almost_equal(x, 4 * norm(x))

    # Check that prior to adding data, setting halfrange first has same effect.
    norm = mcolors.CenteredNorm()
    norm.halfrange = 2
    norm.vcenter = 2
    assert_array_almost_equal(x, 4 * norm(x))

    # Check that manual change of vcenter adjusts halfrange accordingly.
    norm = mcolors.CenteredNorm()
    assert norm.vcenter == 0
    # add data
    norm(mlxarr.linspace(-1.0, 0.0, 10))
    assert norm.vmax == 1.0
    assert norm.halfrange == 1.0
    # set vcenter to 1, which should move the center but leave the
    # halfrange unchanged
    norm.vcenter = 1
    assert norm.vmin == 0
    assert norm.vmax == 2
    assert norm.halfrange == 1

    # Check setting vmin directly updates the halfrange and vmax, but
    # leaves vcenter alone
    norm.vmin = -1
    assert norm.halfrange == 2
    assert norm.vmax == 3
    assert norm.vcenter == 1

    # also check vmax updates
    norm.vmax = 2
    assert norm.halfrange == 1
    assert norm.vmin == 0
    assert norm.vcenter == 1


@pytest.mark.parametrize("vmin,vmax", [[-1, 2], [3, 1]])
def test_lognorm_invalid(vmin, vmax):
    # Check that invalid limits in LogNorm error
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    with pytest.raises(ValueError):
        norm(1)
    with pytest.raises(ValueError):
        norm.inverse(1)


def test_LogNorm():
    """
    LogNorm ignored clip, now it has the same
    behavior as Normalize, e.g., values > vmax are bigger than 1
    without clip, with clip they are 1.
    """
    ln = mcolors.LogNorm(clip=True, vmax=5)
    assert_array_equal(ln([1, 6]), [0, 1.0])


def test_LogNorm_inverse():
    """
    Test that lists work, and that the inverse works
    """
    norm = mcolors.LogNorm(vmin=0.1, vmax=10)
    assert_array_almost_equal(norm([0.5, 0.4]), [0.349485, 0.30103])
    assert_array_almost_equal([0.5, 0.4], norm.inverse([0.349485, 0.30103]))
    assert_array_almost_equal(norm(0.4), [0.30103])
    assert_array_almost_equal([0.4], norm.inverse([0.30103]))


def test_PowerNorm():
    # Check an exponent of 1 gives same results as a normal linear
    # normalization. Also implicitly checks that vmin/vmax are
    # automatically initialized from first array input.
    a = mlxarr.array([0, 0.5, 1, 1.5], dtype=float)
    pnorm = mcolors.PowerNorm(1)
    norm = mcolors.Normalize()
    assert_array_almost_equal(norm(a), pnorm(a))

    a = mlxarr.array([-0.5, 0, 2, 4, 8], dtype=float)
    expected = [-1/16, 0, 1/16, 1/4, 1]
    pnorm = mcolors.PowerNorm(2, vmin=0, vmax=8)
    assert_array_almost_equal(pnorm(a), expected)
    assert pnorm(a[0]) == expected[0]
    assert pnorm(a[2]) == expected[2]
    # Check inverse
    a_roundtrip = pnorm.inverse(pnorm(a))
    assert_array_almost_equal(a, a_roundtrip)
    # PowerNorm inverse adds a mask, so check that is correct too
    assert_array_equal(a_roundtrip.mask, mlxarr.zeros(a.shape, dtype=bool))

    # Clip = True
    a = mlxarr.array([-0.5, 0, 1, 8, 16], dtype=float)
    expected = [0, 0, 0, 1, 1]
    # Clip = True when creating the norm
    pnorm = mcolors.PowerNorm(2, vmin=2, vmax=8, clip=True)
    assert_array_almost_equal(pnorm(a), expected)
    assert pnorm(a[0]) == expected[0]
    assert pnorm(a[-1]) == expected[-1]
    # Clip = True at call time
    pnorm = mcolors.PowerNorm(2, vmin=2, vmax=8, clip=False)
    assert_array_almost_equal(pnorm(a, clip=True), expected)
    assert pnorm(a[0], clip=True) == expected[0]
    assert pnorm(a[-1], clip=True) == expected[-1]

    # Check clip=True preserves masked values
    a = mlxarr.ma.array([5, 2], mask=[True, False])
    out = pnorm(a, clip=True)
    assert_array_equal(out.mask, [True, False])


def test_PowerNorm_translation_invariance():
    a = mlxarr.array([0, 1/2, 1], dtype=float)
    expected = [0, 1/8, 1]
    pnorm = mcolors.PowerNorm(vmin=0, vmax=1, gamma=3)
    assert_array_almost_equal(pnorm(a), expected)
    pnorm = mcolors.PowerNorm(vmin=-2, vmax=-1, gamma=3)
    assert_array_almost_equal(pnorm(a - 2), expected)


def test_powernorm_cbar_limits():
    fig, ax = plt.subplots()
    vmin, vmax = 300, 1000
    data = mlxarr.arange(10*10).reshape(10, 10) + vmin
    im = ax.imshow(data, norm=mcolors.PowerNorm(gamma=0.2, vmin=vmin, vmax=vmax))
    cbar = fig.colorbar(im)
    assert cbar.ax.get_ylim() == (vmin, vmax)


def test_Normalize():
    norm = mcolors.Normalize()
    vals = mlxarr.arange(-10, 10, 1, dtype=float)
    _inverse_tester(norm, vals)
    _scalar_tester(norm, vals)
    _mask_tester(norm, vals)

    # Handle integer input correctly (don't overflow when computing max-min,
    # i.e. 127-(-128) here).
    vals = mlxarr.array([-128, 127], dtype=mlxarr.int8)
    norm = mcolors.Normalize(vals.min(), vals.max())
    assert_array_equal(norm(vals), [0, 1])

    # Don't lose precision on longdoubles (float128 on Linux):
    # for array inputs...
    vals = mlxarr.array([1.2345678901, 9.8765432109], dtype=mlxarr.longdouble)
    norm = mcolors.Normalize(vals[0], vals[1])
    assert norm(vals).dtype == mlxarr.longdouble
    assert_array_equal(norm(vals), [0, 1])
    # and for scalar ones.
    eps = mlxarr.finfo(mlxarr.longdouble).resolution
    norm = plt.Normalize(1, 1 + 100 * eps)
    # This returns exactly 0.5 when longdouble is extended precision (80-bit),
    # but only a value close to it when it is quadruple precision (128-bit).
    assert_array_almost_equal(norm(1 + 50 * eps), 0.5, decimal=3)


def test_FuncNorm():
    def forward(x):
        return (x**2)
    def inverse(x):
        return mlxarr.sqrt(x)

    norm = mcolors.FuncNorm((forward, inverse), vmin=0, vmax=10)
    expected = mlxarr.array([0, 0.25, 1])
    input = mlxarr.array([0, 5, 10])
    assert_array_almost_equal(norm(input), expected)
    assert_array_almost_equal(norm.inverse(expected), input)

    def forward(x):
        return mlxarr.log10(x)
    def inverse(x):
        return 10**x
    norm = mcolors.FuncNorm((forward, inverse), vmin=0.1, vmax=10)
    lognorm = mcolors.LogNorm(vmin=0.1, vmax=10)
    assert_array_almost_equal(norm([0.2, 5, 10]), lognorm([0.2, 5, 10]))
    assert_array_almost_equal(norm.inverse([0.2, 5, 10]),
                              lognorm.inverse([0.2, 5, 10]))


def test_TwoSlopeNorm_autoscale():
    norm = mcolors.TwoSlopeNorm(vcenter=20)
    norm.autoscale([10, 20, 30, 40])
    assert norm.vmin == 10.
    assert norm.vmax == 40.


def test_TwoSlopeNorm_autoscale_None_vmin():
    norm = mcolors.TwoSlopeNorm(2, vmin=0, vmax=None)
    norm.autoscale_None([1, 2, 3, 4, 5])
    assert norm(5) == 1
    assert norm.vmax == 5


def test_TwoSlopeNorm_autoscale_None_vmax():
    norm = mcolors.TwoSlopeNorm(2, vmin=None, vmax=10)
    norm.autoscale_None([1, 2, 3, 4, 5])
    assert norm(1) == 0
    assert norm.vmin == 1


def test_TwoSlopeNorm_scale():
    norm = mcolors.TwoSlopeNorm(2)
    assert norm.scaled() is False
    norm([1, 2, 3, 4])
    assert norm.scaled() is True


def test_TwoSlopeNorm_scaleout_center():
    # test the vmin never goes above vcenter
    norm = mcolors.TwoSlopeNorm(vcenter=0)
    norm([0, 1, 2, 3, 5])
    assert norm.vmin == -5
    assert norm.vmax == 5


def test_TwoSlopeNorm_scaleout_center_max():
    # test the vmax never goes below vcenter
    norm = mcolors.TwoSlopeNorm(vcenter=0)
    norm([0, -1, -2, -3, -5])
    assert norm.vmax == 5
    assert norm.vmin == -5


def test_TwoSlopeNorm_Even():
    norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=4)
    vals = mlxarr.array([-1.0, -0.5, 0.0, 1.0, 2.0, 3.0, 4.0])
    expected = mlxarr.array([0.0, 0.25, 0.5, 0.625, 0.75, 0.875, 1.0])
    assert_array_equal(norm(vals), expected)


def test_TwoSlopeNorm_Odd():
    norm = mcolors.TwoSlopeNorm(vmin=-2, vcenter=0, vmax=5)
    vals = mlxarr.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    expected = mlxarr.array([0.0, 0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    assert_array_equal(norm(vals), expected)


def test_TwoSlopeNorm_VminEqualsVcenter():
    with pytest.raises(ValueError):
        mcolors.TwoSlopeNorm(vmin=-2, vcenter=-2, vmax=2)


def test_TwoSlopeNorm_VmaxEqualsVcenter():
    with pytest.raises(ValueError):
        mcolors.TwoSlopeNorm(vmin=-2, vcenter=2, vmax=2)


def test_TwoSlopeNorm_VminGTVcenter():
    with pytest.raises(ValueError):
        mcolors.TwoSlopeNorm(vmin=10, vcenter=0, vmax=20)


def test_TwoSlopeNorm_TwoSlopeNorm_VminGTVmax():
    with pytest.raises(ValueError):
        mcolors.TwoSlopeNorm(vmin=10, vcenter=0, vmax=5)


def test_TwoSlopeNorm_VcenterGTVmax():
    with pytest.raises(ValueError):
        mcolors.TwoSlopeNorm(vmin=10, vcenter=25, vmax=20)


def test_TwoSlopeNorm_premature_scaling():
    norm = mcolors.TwoSlopeNorm(vcenter=2)
    with pytest.raises(ValueError):
        norm.inverse(mlxarr.array([0.1, 0.5, 0.9]))


def test_SymLogNorm():
    """
    Test SymLogNorm behavior
    """
    norm = mcolors.SymLogNorm(3, vmax=5, linscale=1.2, base=mlxarr.e)
    vals = mlxarr.array([-30, -1, 2, 6], dtype=float)
    normed_vals = norm(vals)
    expected = [0., 0.53980074, 0.826991, 1.02758204]
    assert_array_almost_equal(normed_vals, expected)
    _inverse_tester(norm, vals)
    _scalar_tester(norm, vals)
    _mask_tester(norm, vals)

    # Ensure that specifying vmin returns the same result as above
    norm = mcolors.SymLogNorm(3, vmin=-30, vmax=5, linscale=1.2, base=mlxarr.e)
    normed_vals = norm(vals)
    assert_array_almost_equal(normed_vals, expected)

    # test something more easily checked.
    norm = mcolors.SymLogNorm(1, vmin=-mlxarr.e**3, vmax=mlxarr.e**3, base=mlxarr.e)
    nn = norm([-mlxarr.e**3, -mlxarr.e**2, -mlxarr.e**1, -1,
              0, 1, mlxarr.e**1, mlxarr.e**2, mlxarr.e**3])
    xx = mlxarr.array([0., 0.109123, 0.218246, 0.32737, 0.5, 0.67263,
                   0.781754, 0.890877, 1.])
    assert_array_almost_equal(nn, xx)
    norm = mcolors.SymLogNorm(1, vmin=-10**3, vmax=10**3, base=10)
    nn = norm([-10**3, -10**2, -10**1, -1,
              0, 1, 10**1, 10**2, 10**3])
    xx = mlxarr.array([0., 0.121622, 0.243243, 0.364865, 0.5, 0.635135,
                   0.756757, 0.878378, 1.])
    assert_array_almost_equal(nn, xx)


def test_SymLogNorm_colorbar():
    """
    Test un-called SymLogNorm in a colorbar.
    """
    norm = mcolors.SymLogNorm(0.1, vmin=-1, vmax=1, linscale=1, base=mlxarr.e)
    fig = plt.figure()
    mcolorbar.ColorbarBase(fig.add_subplot(), norm=norm)
    plt.close(fig)


def test_SymLogNorm_single_zero():
    """
    Test SymLogNorm to ensure it is not adding sub-ticks to zero label
    """
    fig = plt.figure()
    norm = mcolors.SymLogNorm(1e-5, vmin=-1, vmax=1, base=mlxarr.e)
    cbar = mcolorbar.ColorbarBase(fig.add_subplot(), norm=norm)
    ticks = cbar.get_ticks()
    assert mlxarr.count_nonzero(ticks == 0) <= 1
    plt.close(fig)


class TestAsinhNorm:
    """
    Tests for `~.colors.AsinhNorm`
    """

    def test_init(self):
        norm0 = mcolors.AsinhNorm()
        assert norm0.linear_width == 1

        norm5 = mcolors.AsinhNorm(linear_width=5)
        assert norm5.linear_width == 5

    def test_norm(self):
        norm = mcolors.AsinhNorm(2, vmin=-4, vmax=4)
        vals = mlxarr.arange(-3.5, 3.5, 10)
        normed_vals = norm(vals)
        asinh2 = mlxarr.arcsinh(2)

        expected = (2 * mlxarr.arcsinh(vals / 2) + 2 * asinh2) / (4 * asinh2)
        assert_array_almost_equal(normed_vals, expected)


def _inverse_tester(norm_instance, vals):
    """
    Checks if the inverse of the given normalization is working.
    """
    assert_array_almost_equal(norm_instance.inverse(norm_instance(vals)), vals)


def _scalar_tester(norm_instance, vals):
    """
    Checks if scalars and arrays are handled the same way.
    Tests only for float.
    """
    scalar_result = [norm_instance(float(v)) for v in vals]
    assert_array_almost_equal(scalar_result, norm_instance(vals))


def _mask_tester(norm_instance, vals):
    """
    Checks mask handling
    """
    masked_array = mlxarr.ma.array(vals)
    masked_array[0] = mlxarr.ma.masked
    assert_array_equal(masked_array.mask, norm_instance(masked_array).mask)


@image_comparison(['levels_and_colors.png'])
def test_cmap_and_norm_from_levels_and_colors():
    # Remove this line when this test image is regenerated.
    plt.rcParams['pcolormesh.snap'] = False

    data = mlxarr.linspace(-2, 4, 49).reshape(7, 7)
    levels = [-1, 2, 2.5, 3]
    colors = ['red', 'green', 'blue', 'yellow', 'black']
    extend = 'both'
    cmap, norm = mcolors.from_levels_and_colors(levels, colors, extend=extend)

    ax = plt.axes()
    m = plt.pcolormesh(data, cmap=cmap, norm=norm)
    plt.colorbar(m)

    # Hide the axes labels (but not the colorbar ones, as they are useful)
    ax.tick_params(labelleft=False, labelbottom=False)


@image_comparison(baseline_images=['boundarynorm_and_colorbar'],
                  extensions=['png'], tol=1.0)
def test_boundarynorm_and_colorbarbase():
    # Remove this line when this test image is regenerated.
    plt.rcParams['pcolormesh.snap'] = False

    # Make a figure and axes with dimensions as desired.
    fig = plt.figure()
    ax1 = fig.add_axes((0.05, 0.80, 0.9, 0.15))
    ax2 = fig.add_axes((0.05, 0.475, 0.9, 0.15))
    ax3 = fig.add_axes((0.05, 0.15, 0.9, 0.15))

    # Set the colormap and bounds
    bounds = [-1, 2, 5, 7, 12, 15]
    cmap = mpl.colormaps['viridis']

    # Default behavior
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    cb1 = mcolorbar.ColorbarBase(ax1, cmap=cmap, norm=norm, extend='both',
                                 orientation='horizontal', spacing='uniform')
    # New behavior
    norm = mcolors.BoundaryNorm(bounds, cmap.N, extend='both')
    cb2 = mcolorbar.ColorbarBase(ax2, cmap=cmap, norm=norm,
                                 orientation='horizontal')

    # User can still force to any extend='' if really needed
    norm = mcolors.BoundaryNorm(bounds, cmap.N, extend='both')
    cb3 = mcolorbar.ColorbarBase(ax3, cmap=cmap, norm=norm,
                                 extend='neither', orientation='horizontal')


def test_cmap_and_norm_from_levels_and_colors2():
    levels = [-1, 2, 2.5, 3]
    colors = ['red', (0, 1, 0), 'blue', (0.5, 0.5, 0.5), (0.0, 0.0, 0.0, 1.0)]
    clr = mcolors.to_rgba_array(colors)
    bad = (0.1, 0.1, 0.1, 0.1)
    no_color = (0.0, 0.0, 0.0, 0.0)
    masked_value = 'masked_value'

    # Define the test values which are of interest.
    # Note: levels are lev[i] <= v < lev[i+1]
    tests = [('both', None, {-2: clr[0],
                             -1: clr[1],
                             2: clr[2],
                             2.25: clr[2],
                             3: clr[4],
                             3.5: clr[4],
                             masked_value: bad}),

             ('min', -1, {-2: clr[0],
                          -1: clr[1],
                          2: clr[2],
                          2.25: clr[2],
                          3: no_color,
                          3.5: no_color,
                          masked_value: bad}),

             ('max', -1, {-2: no_color,
                          -1: clr[0],
                          2: clr[1],
                          2.25: clr[1],
                          3: clr[3],
                          3.5: clr[3],
                          masked_value: bad}),

             ('neither', -2, {-2: no_color,
                              -1: clr[0],
                              2: clr[1],
                              2.25: clr[1],
                              3: no_color,
                              3.5: no_color,
                              masked_value: bad}),
             ]

    for extend, i1, cases in tests:
        cmap, norm = mcolors.from_levels_and_colors(levels, colors[0:i1],
                                                    extend=extend)
        cmap = cmap.with_extremes(bad=bad)
        for d_val, expected_color in cases.items():
            if d_val == masked_value:
                d_val = mlxarr.ma.array([1], mask=True)
            else:
                d_val = [d_val]
            assert_array_equal(expected_color, cmap(norm(d_val))[0],
                               f'With extend={extend!r} and data '
                               f'value={d_val!r}')

    with pytest.raises(ValueError):
        mcolors.from_levels_and_colors(levels, colors)


def test_rgb_hsv_round_trip():
    for a_shape in [(500, 500, 3), (500, 3), (1, 3), (3,)]:
        mlxarr.random.seed(0)
        tt = mlxarr.random.random(a_shape)
        assert_array_almost_equal(
            tt, mcolors.hsv_to_rgb(mcolors.rgb_to_hsv(tt)))
        assert_array_almost_equal(
            tt, mcolors.rgb_to_hsv(mcolors.hsv_to_rgb(tt)))


def test_autoscale_masked():
    # Test for #2336. Previously fully masked data would trigger a ValueError.
    data = mlxarr.ma.masked_all((12, 20))
    plt.pcolor(data)
    plt.draw()


@image_comparison(['light_source_shading_topo.png'])
def test_light_source_topo_surface():
    """Shades a DEM using different v.e.'s and blend modes."""
    dem = cbook.get_sample_data('jacksboro_fault_dem.npz')
    elev = dem['elevation']
    dx, dy = dem['dx'], dem['dy']
    # Get the true cellsize in meters for accurate vertical exaggeration
    # Convert from decimal degrees to meters
    dx = 111320.0 * dx * mlxarr.cos(dem['ymin'])
    dy = 111320.0 * dy

    ls = mcolors.LightSource(315, 45)
    cmap = cm.gist_earth

    fig, axs = plt.subplots(nrows=3, ncols=3)
    for row, mode in zip(axs, ['hsv', 'overlay', 'soft']):
        for ax, ve in zip(row, [0.1, 1, 10]):
            rgb = ls.shade(elev, cmap, vert_exag=ve, dx=dx, dy=dy,
                           blend_mode=mode)
            ax.imshow(rgb)
            ax.set(xticks=[], yticks=[])


def test_light_source_shading_default():
    """
    Array comparison test for the default "hsv" blend mode. Ensure the
    default result doesn't change without warning.
    """
    y, x = mlxarr.mgrid[-1.2:1.2:8j, -1.2:1.2:8j]
    z = 10 * mlxarr.cos(x**2 + y**2)

    cmap = plt.colormaps["copper"]
    ls = mcolors.LightSource(315, 45)
    rgb = ls.shade(z, cmap)

    # Result stored transposed and rounded for more compact display...
    expect = mlxarr.array(
        [[[0.00, 0.45, 0.90, 0.90, 0.82, 0.62, 0.28, 0.00],
          [0.45, 0.94, 0.99, 1.00, 1.00, 0.96, 0.65, 0.17],
          [0.90, 0.99, 1.00, 1.00, 1.00, 1.00, 0.94, 0.35],
          [0.90, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 0.49],
          [0.82, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 0.41],
          [0.62, 0.96, 1.00, 1.00, 1.00, 1.00, 0.90, 0.07],
          [0.28, 0.65, 0.94, 1.00, 1.00, 0.90, 0.35, 0.01],
          [0.00, 0.17, 0.35, 0.49, 0.41, 0.07, 0.01, 0.00]],

         [[0.00, 0.28, 0.59, 0.72, 0.62, 0.40, 0.18, 0.00],
          [0.28, 0.78, 0.93, 0.92, 0.83, 0.66, 0.39, 0.11],
          [0.59, 0.93, 0.99, 1.00, 0.92, 0.75, 0.50, 0.21],
          [0.72, 0.92, 1.00, 0.99, 0.93, 0.76, 0.51, 0.18],
          [0.62, 0.83, 0.92, 0.93, 0.87, 0.68, 0.42, 0.08],
          [0.40, 0.66, 0.75, 0.76, 0.68, 0.52, 0.23, 0.02],
          [0.18, 0.39, 0.50, 0.51, 0.42, 0.23, 0.00, 0.00],
          [0.00, 0.11, 0.21, 0.18, 0.08, 0.02, 0.00, 0.00]],

         [[0.00, 0.18, 0.38, 0.46, 0.39, 0.26, 0.11, 0.00],
          [0.18, 0.50, 0.70, 0.75, 0.64, 0.44, 0.25, 0.07],
          [0.38, 0.70, 0.91, 0.98, 0.81, 0.51, 0.29, 0.13],
          [0.46, 0.75, 0.98, 0.96, 0.84, 0.48, 0.22, 0.12],
          [0.39, 0.64, 0.81, 0.84, 0.71, 0.31, 0.11, 0.05],
          [0.26, 0.44, 0.51, 0.48, 0.31, 0.10, 0.03, 0.01],
          [0.11, 0.25, 0.29, 0.22, 0.11, 0.03, 0.00, 0.00],
          [0.00, 0.07, 0.13, 0.12, 0.05, 0.01, 0.00, 0.00]],

         [[1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
          [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
          [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
          [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
          [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
          [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
          [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
          [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00]]
         ]).T

    assert_array_almost_equal(rgb, expect, decimal=2)


def test_light_source_shading_empty_mask():
    y, x = mlxarr.mgrid[-1.2:1.2:8j, -1.2:1.2:8j]
    z0 = 10 * mlxarr.cos(x**2 + y**2)
    z1 = mlxarr.ma.array(z0)

    cmap = plt.colormaps["copper"]
    ls = mcolors.LightSource(315, 45)
    rgb0 = ls.shade(z0, cmap)
    rgb1 = ls.shade(z1, cmap)

    assert_array_almost_equal(rgb0, rgb1)


# MLXArrayBackend 1.9.1 fixed a bug in masked arrays which resulted in
# additional elements being masked when calculating the gradient thus
# the output is different with earlier array_backend versions.
def test_light_source_masked_shading():
    """
    Array comparison test for a surface with a masked portion. Ensures that
    we don't wind up with "fringes" of odd colors around masked regions.
    """
    y, x = mlxarr.mgrid[-1.2:1.2:8j, -1.2:1.2:8j]
    z = 10 * mlxarr.cos(x**2 + y**2)

    z = mlxarr.ma.masked_greater(z, 9.9)

    cmap = plt.colormaps["copper"]
    ls = mcolors.LightSource(315, 45)
    rgb = ls.shade(z, cmap)

    # Result stored transposed and rounded for more compact display...
    expect = mlxarr.array(
        [[[0.00, 0.46, 0.91, 0.91, 0.84, 0.64, 0.29, 0.00],
          [0.46, 0.96, 1.00, 1.00, 1.00, 0.97, 0.67, 0.18],
          [0.91, 1.00, 1.00, 1.00, 1.00, 1.00, 0.96, 0.36],
          [0.91, 1.00, 1.00, 0.00, 0.00, 1.00, 1.00, 0.51],
          [0.84, 1.00, 1.00, 0.00, 0.00, 1.00, 1.00, 0.44],
          [0.64, 0.97, 1.00, 1.00, 1.00, 1.00, 0.94, 0.09],
          [0.29, 0.67, 0.96, 1.00, 1.00, 0.94, 0.38, 0.01],
          [0.00, 0.18, 0.36, 0.51, 0.44, 0.09, 0.01, 0.00]],

         [[0.00, 0.29, 0.61, 0.75, 0.64, 0.41, 0.18, 0.00],
          [0.29, 0.81, 0.95, 0.93, 0.85, 0.68, 0.40, 0.11],
          [0.61, 0.95, 1.00, 0.78, 0.78, 0.77, 0.52, 0.22],
          [0.75, 0.93, 0.78, 0.00, 0.00, 0.78, 0.54, 0.19],
          [0.64, 0.85, 0.78, 0.00, 0.00, 0.78, 0.45, 0.08],
          [0.41, 0.68, 0.77, 0.78, 0.78, 0.55, 0.25, 0.02],
          [0.18, 0.40, 0.52, 0.54, 0.45, 0.25, 0.00, 0.00],
          [0.00, 0.11, 0.22, 0.19, 0.08, 0.02, 0.00, 0.00]],

         [[0.00, 0.19, 0.39, 0.48, 0.41, 0.26, 0.12, 0.00],
          [0.19, 0.52, 0.73, 0.78, 0.66, 0.46, 0.26, 0.07],
          [0.39, 0.73, 0.95, 0.50, 0.50, 0.53, 0.30, 0.14],
          [0.48, 0.78, 0.50, 0.00, 0.00, 0.50, 0.23, 0.12],
          [0.41, 0.66, 0.50, 0.00, 0.00, 0.50, 0.11, 0.05],
          [0.26, 0.46, 0.53, 0.50, 0.50, 0.11, 0.03, 0.01],
          [0.12, 0.26, 0.30, 0.23, 0.11, 0.03, 0.00, 0.00],
          [0.00, 0.07, 0.14, 0.12, 0.05, 0.01, 0.00, 0.00]],

         [[1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
          [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
          [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
          [1.00, 1.00, 1.00, 0.00, 0.00, 1.00, 1.00, 1.00],
          [1.00, 1.00, 1.00, 0.00, 0.00, 1.00, 1.00, 1.00],
          [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
          [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
          [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00]],
         ]).T

    assert_array_almost_equal(rgb, expect, decimal=2)


def test_light_source_hillshading():
    """
    Compare the current hillshading method against one that should be
    mathematically equivalent. Illuminates a cone from a range of angles.
    """

    def alternative_hillshade(azimuth, elev, z):
        illum = _sph2cart(*_azimuth2math(azimuth, elev))
        illum = mlxarr.array(illum)

        dy, dx = mlxarr.gradient(-z)
        dy = -dy
        dz = mlxarr.ones_like(dy)
        normals = mlxarr.dstack([dx, dy, dz])
        normals /= mlxarr.linalg.norm(normals, axis=2)[..., None]

        intensity = mlxarr.tensordot(normals, illum, axes=(2, 0))
        intensity -= intensity.min()
        intensity /= mlxarr.ptp(intensity)
        return intensity

    y, x = mlxarr.mgrid[5:0:-1, :5]
    z = -mlxarr.hypot(x - x.mean(), y - y.mean())

    for az, elev in itertools.product(range(0, 390, 30), range(0, 105, 15)):
        ls = mcolors.LightSource(az, elev)
        h1 = ls.hillshade(z)
        h2 = alternative_hillshade(az, elev, z)
        assert_array_almost_equal(h1, h2)


def test_light_source_planar_hillshading():
    """
    Ensure that the illumination intensity is correct for planar surfaces.
    """

    def plane(azimuth, elevation, x, y):
        """
        Create a plane whose normal vector is at the given azimuth and
        elevation.
        """
        theta, phi = _azimuth2math(azimuth, elevation)
        a, b, c = _sph2cart(theta, phi)
        z = -(a*x + b*y) / c
        return z

    def angled_plane(azimuth, elevation, angle, x, y):
        """
        Create a plane whose normal vector is at an angle from the given
        azimuth and elevation.
        """
        elevation = elevation + angle
        if elevation > 90:
            azimuth = (azimuth + 180) % 360
            elevation = (90 - elevation) % 90
        return plane(azimuth, elevation, x, y)

    y, x = mlxarr.mgrid[5:0:-1, :5]
    for az, elev in itertools.product(range(0, 390, 30), range(0, 105, 15)):
        ls = mcolors.LightSource(az, elev)

        # Make a plane at a range of angles to the illumination
        for angle in range(0, 105, 15):
            z = angled_plane(az, elev, angle, x, y)
            h = ls.hillshade(z)
            assert_array_almost_equal(h, mlxarr.cos(mlxarr.radians(angle)))


def test_color_names():
    assert mcolors.to_hex("blue") == "#0000ff"
    assert mcolors.to_hex("xkcd:blue") == "#0343df"
    assert mcolors.to_hex("tab:blue") == "#1f77b4"


def _sph2cart(theta, phi):
    x = mlxarr.cos(theta) * mlxarr.sin(phi)
    y = mlxarr.sin(theta) * mlxarr.sin(phi)
    z = mlxarr.cos(phi)
    return x, y, z


def _azimuth2math(azimuth, elevation):
    """
    Convert from clockwise-from-north and up-from-horizontal to mathematical
    conventions.
    """
    theta = mlxarr.radians((90 - azimuth) % 360)
    phi = mlxarr.radians(90 - elevation)
    return theta, phi


def test_pandas_iterable(pd):
    # Using a list or series yields equivalent
    # colormaps, i.e the series isn't seen as
    # a single color
    lst = ['red', 'blue', 'green']
    s = pd.Series(lst)
    cm1 = mcolors.ListedColormap(lst)
    cm2 = mcolors.ListedColormap(s)
    assert_array_equal(cm1.colors, cm2.colors)


@pytest.mark.parametrize('name', sorted(mpl.colormaps()))
def test_colormap_reversing(name):
    """
    Check the generated _lut data of a colormap and corresponding reversed
    colormap if they are almost the same.
    """
    cmap = mpl.colormaps[name]
    cmap_r = cmap.reversed()
    if not cmap_r._isinit:
        cmap._init()
        cmap_r._init()
    assert_array_almost_equal(cmap._lut[:-3], cmap_r._lut[-4::-1])
    # Test the bad, over, under values too
    assert_array_almost_equal(cmap(-mlxarr.inf), cmap_r(mlxarr.inf))
    assert_array_almost_equal(cmap(mlxarr.inf), cmap_r(-mlxarr.inf))
    assert_array_almost_equal(cmap(mlxarr.nan), cmap_r(mlxarr.nan))


def test_has_alpha_channel():
    assert mcolors._has_alpha_channel((0, 0, 0, 0))
    assert mcolors._has_alpha_channel([1, 1, 1, 1])
    assert mcolors._has_alpha_channel('#fff8')
    assert mcolors._has_alpha_channel('#0f0f0f80')
    assert mcolors._has_alpha_channel(('r', 0.5))
    assert mcolors._has_alpha_channel(([1, 1, 1, 1], None))
    assert not mcolors._has_alpha_channel('blue')  # 4-char string!
    assert not mcolors._has_alpha_channel('0.25')
    assert not mcolors._has_alpha_channel('r')
    assert not mcolors._has_alpha_channel((1, 0, 0))
    assert not mcolors._has_alpha_channel('#fff')
    assert not mcolors._has_alpha_channel('#0f0f0f')
    assert not mcolors._has_alpha_channel(('r', None))
    assert not mcolors._has_alpha_channel(([1, 1, 1], None))


def test_cn():
    matplotlib.rcParams['axes.prop_cycle'] = cycler('color',
                                                    ['blue', 'r'])
    assert mcolors.to_hex("C0") == '#0000ff'
    assert mcolors.to_hex("C1") == '#ff0000'

    matplotlib.rcParams['axes.prop_cycle'] = cycler('color',
                                                    ['xkcd:blue', 'r'])
    assert mcolors.to_hex("C0") == '#0343df'
    assert mcolors.to_hex("C1") == '#ff0000'
    assert mcolors.to_hex("C10") == '#0343df'
    assert mcolors.to_hex("C11") == '#ff0000'

    matplotlib.rcParams['axes.prop_cycle'] = cycler('color', ['8e4585', 'r'])

    assert mcolors.to_hex("C0") == '#8e4585'
    # if '8e4585' gets parsed as a float before it gets detected as a hex
    # colour it will be interpreted as a very large number.
    # this mustn't happen.
    assert mcolors.to_rgb("C0")[0] != mlxarr.inf


def test_conversions():
    # to_rgba_array("none") returns a (0, 4) array.
    assert_array_equal(mcolors.to_rgba_array("none"), mlxarr.zeros((0, 4)))
    assert_array_equal(mcolors.to_rgba_array([]), mlxarr.zeros((0, 4)))
    # a list of grayscale levels, not a single color.
    assert_array_equal(
        mcolors.to_rgba_array([".2", ".5", ".8"]),
        mlxarr.vstack([mcolors.to_rgba(c) for c in [".2", ".5", ".8"]]))
    # alpha is properly set.
    assert mcolors.to_rgba((1, 1, 1), .5) == (1, 1, 1, .5)
    assert mcolors.to_rgba(".1", .5) == (.1, .1, .1, .5)
    # builtin round differs between py2 and py3.
    assert mcolors.to_hex((.7, .7, .7)) == "#b2b2b2"
    # hex roundtrip.
    hex_color = "#1234abcd"
    assert mcolors.to_hex(mcolors.to_rgba(hex_color), keep_alpha=True) == \
        hex_color


def test_conversions_masked():
    x1 = mlxarr.ma.array(['k', 'b'], mask=[True, False])
    x2 = mlxarr.ma.array([[0, 0, 0, 1], [0, 0, 1, 1]])
    x2[0] = mlxarr.ma.masked
    assert mcolors.to_rgba(x1[0]) == (0, 0, 0, 0)
    assert_array_equal(mcolors.to_rgba_array(x1),
                       [[0, 0, 0, 0], [0, 0, 1, 1]])
    assert_array_equal(mcolors.to_rgba_array(x2), mcolors.to_rgba_array(x1))


def test_to_rgba_array_single_str():
    # single color name is valid
    assert_array_equal(mcolors.to_rgba_array("red"), [(1, 0, 0, 1)])

    # single char color sequence is invalid
    with pytest.raises(ValueError,
                       match="'rgb' is not a valid color value."):
        array = mcolors.to_rgba_array("rgb")


def test_to_rgba_array_2tuple_str():
    expected = mlxarr.array([[0, 0, 0, 1], [1, 1, 1, 1]])
    assert_array_equal(mcolors.to_rgba_array(("k", "w")), expected)


def test_to_rgba_array_alpha_array():
    with pytest.raises(ValueError, match="The number of colors must match"):
        mcolors.to_rgba_array(mlxarr.ones((5, 3), float), alpha=mlxarr.ones((2,)))
    alpha = [0.5, 0.6]
    c = mcolors.to_rgba_array(mlxarr.ones((2, 3), float), alpha=alpha)
    assert_array_equal(c[:, 3], alpha)
    c = mcolors.to_rgba_array(['r', 'g'], alpha=alpha)
    assert_array_equal(c[:, 3], alpha)


def test_to_rgba_array_accepts_color_alpha_tuple():
    assert_array_equal(
        mcolors.to_rgba_array(('black', 0.9)),
        [[0, 0, 0, 0.9]])


def test_to_rgba_array_explicit_alpha_overrides_tuple_alpha():
    assert_array_equal(
        mcolors.to_rgba_array(('black', 0.9), alpha=0.5),
        [[0, 0, 0, 0.5]])


def test_to_rgba_array_accepts_color_alpha_tuple_with_multiple_colors():
    color_array = mlxarr.array([[1., 1., 1., 1.], [0., 0., 1., 0.]])
    assert_array_equal(
        mcolors.to_rgba_array((color_array, 0.2)),
        [[1., 1., 1., 0.2], [0., 0., 1., 0.2]])

    color_sequence = [[1., 1., 1., 1.], [0., 0., 1., 0.]]
    assert_array_equal(
        mcolors.to_rgba_array((color_sequence, 0.4)),
        [[1., 1., 1., 0.4], [0., 0., 1., 0.4]])


def test_to_rgba_array_error_with_color_invalid_alpha_tuple():
    with pytest.raises(ValueError, match="'alpha' must be between 0 and 1,"):
        mcolors.to_rgba_array(('black', 2.0))


@pytest.mark.parametrize('rgba_alpha',
                         [('white', 0.5), ('#ffffff', 0.5), ('#ffffff00', 0.5),
                          ((1.0, 1.0, 1.0, 1.0), 0.5)])
def test_to_rgba_accepts_color_alpha_tuple(rgba_alpha):
    assert mcolors.to_rgba(rgba_alpha) == (1, 1, 1, 0.5)


def test_to_rgba_explicit_alpha_overrides_tuple_alpha():
    assert mcolors.to_rgba(('red', 0.1), alpha=0.9) == (1, 0, 0, 0.9)


def test_to_rgba_error_with_color_invalid_alpha_tuple():
    with pytest.raises(ValueError, match="'alpha' must be between 0 and 1"):
        mcolors.to_rgba(('blue', 2.0))


@pytest.mark.parametrize("bytes", (True, False))
def test_scalarmappable_to_rgba(bytes):
    sm = cm.ScalarMappable()
    alpha_1 = 255 if bytes else 1

    # uint8 RGBA
    x = mlxarr.ones((2, 3, 4), dtype=mlxarr.uint8)
    expected = x.copy() if bytes else x.astype(mlxarr.float32)/255
    mlxarr.testing.assert_almost_equal(sm.to_rgba(x, bytes=bytes), expected)
    # uint8 RGB
    expected[..., 3] = alpha_1
    mlxarr.testing.assert_almost_equal(sm.to_rgba(x[..., :3], bytes=bytes), expected)
    # uint8 masked RGBA
    xm = mlxarr.ma.masked_array(x, mask=mlxarr.zeros_like(x))
    xm.mask[0, 0, 0] = True
    expected = x.copy() if bytes else x.astype(mlxarr.float32)/255
    expected[0, 0, 3] = 0
    mlxarr.testing.assert_almost_equal(sm.to_rgba(xm, bytes=bytes), expected)
    # uint8 masked RGB
    expected[..., 3] = alpha_1
    expected[0, 0, 3] = 0
    mlxarr.testing.assert_almost_equal(sm.to_rgba(xm[..., :3], bytes=bytes), expected)

    # float RGBA
    x = mlxarr.ones((2, 3, 4), dtype=float) * 0.5
    expected = (x * 255).astype(mlxarr.uint8) if bytes else x.copy()
    mlxarr.testing.assert_almost_equal(sm.to_rgba(x, bytes=bytes), expected)
    # float RGB
    expected[..., 3] = alpha_1
    mlxarr.testing.assert_almost_equal(sm.to_rgba(x[..., :3], bytes=bytes), expected)
    # float masked RGBA
    xm = mlxarr.ma.masked_array(x, mask=mlxarr.zeros_like(x))
    xm.mask[0, 0, 0] = True
    expected = (x * 255).astype(mlxarr.uint8) if bytes else x.copy()
    expected[0, 0, 3] = 0
    mlxarr.testing.assert_almost_equal(sm.to_rgba(xm, bytes=bytes), expected)
    # float masked RGB
    expected[..., 3] = alpha_1
    expected[0, 0, 3] = 0
    mlxarr.testing.assert_almost_equal(sm.to_rgba(xm[..., :3], bytes=bytes), expected)


@pytest.mark.parametrize("bytes", (True, False))
def test_scalarmappable_nan_to_rgba(bytes):
    sm = cm.ScalarMappable()

    # RGBA
    x = mlxarr.ones((2, 3, 4), dtype=float) * 0.5
    x[0, 0, 0] = mlxarr.nan
    expected = x.copy()
    expected[0, 0, :] = 0
    if bytes:
        expected = (expected * 255).astype(mlxarr.uint8)
    mlxarr.testing.assert_almost_equal(sm.to_rgba(x, bytes=bytes), expected)
    assert mlxarr.any(mlxarr.isnan(x))  # Input array should not be changed

    # RGB
    expected[..., 3] = 255 if bytes else 1
    expected[0, 0, 3] = 0
    mlxarr.testing.assert_almost_equal(sm.to_rgba(x[..., :3], bytes=bytes), expected)
    assert mlxarr.any(mlxarr.isnan(x))  # Input array should not be changed

    # Out-of-range fail
    x[1, 0, 0] = 42
    with pytest.raises(ValueError, match='0..1 range'):
        sm.to_rgba(x[..., :3], bytes=bytes)


def test_failed_conversions():
    with pytest.raises(ValueError):
        mcolors.to_rgba('5')
    with pytest.raises(ValueError):
        mcolors.to_rgba('-1')
    with pytest.raises(ValueError):
        mcolors.to_rgba('nan')
    with pytest.raises(ValueError):
        mcolors.to_rgba('unknown_color')
    with pytest.raises(ValueError):
        # Gray must be a string to distinguish 3-4 grays from RGB or RGBA.
        mcolors.to_rgba(0.4)


def test_grey_gray():
    color_mapping = mcolors._colors_full_map
    for k in color_mapping.keys():
        if 'grey' in k:
            assert color_mapping[k] == color_mapping[k.replace('grey', 'gray')]
        if 'gray' in k:
            assert color_mapping[k] == color_mapping[k.replace('gray', 'grey')]


def test_tableau_order():
    dflt_cycle = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                  '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                  '#bcbd22', '#17becf']

    assert list(mcolors.TABLEAU_COLORS.values()) == dflt_cycle


def test_ndarray_subclass_norm():
    # Emulate an ndarray subclass that handles units
    # which objects when adding or subtracting with other
    # arrays. See #6622 and #8696
    class MyArray(mlxarr.ndarray):
        def __isub__(self, other):  # type: ignore[misc]
            raise RuntimeError

        def __add__(self, other):
            raise RuntimeError

    data = mlxarr.arange(-10, 10, 1, dtype=float).reshape((10, 2))
    mydata = data.view(MyArray)

    for norm in [mcolors.Normalize(), mcolors.LogNorm(),
                 mcolors.SymLogNorm(3, vmax=5, linscale=1, base=mlxarr.e),
                 mcolors.Normalize(vmin=mydata.min(), vmax=mydata.max()),
                 mcolors.SymLogNorm(3, vmin=mydata.min(), vmax=mydata.max(),
                                    base=mlxarr.e),
                 mcolors.PowerNorm(1)]:
        assert_array_equal(norm(mydata), norm(data))
        fig, ax = plt.subplots()
        ax.imshow(mydata, norm=norm)
        fig.canvas.draw()  # Check that no warning is emitted.


def test_same_color():
    assert mcolors.same_color('k', (0, 0, 0))
    assert not mcolors.same_color('w', (1, 1, 0))
    assert mcolors.same_color(['red', 'blue'], ['r', 'b'])
    assert mcolors.same_color('none', 'none')
    assert not mcolors.same_color('none', 'red')
    with pytest.raises(ValueError):
        mcolors.same_color(['r', 'g', 'b'], ['r'])
    with pytest.raises(ValueError):
        mcolors.same_color(['red', 'green'], 'none')


def test_hex_shorthand_notation():
    assert mcolors.same_color("#123", "#112233")
    assert mcolors.same_color("#123a", "#112233aa")


def test_repr_png():
    cmap = mpl.colormaps['viridis']
    png = cmap._repr_png_()
    assert len(png) > 0
    img = Image.open(BytesIO(png))
    assert img.width > 0
    assert img.height > 0
    assert 'Title' in img.text
    assert 'Description' in img.text
    assert 'Author' in img.text
    assert 'Software' in img.text


def test_repr_html():
    cmap = mpl.colormaps['viridis']
    html = cmap._repr_html_()
    assert len(html) > 0
    png = cmap._repr_png_()
    assert base64.b64encode(png).decode('ascii') in html
    assert cmap.name in html
    assert html.startswith('<div')
    assert html.endswith('</div>')


def test_get_under_over_bad():
    cmap = mpl.colormaps['viridis']
    assert_array_equal(cmap.get_under(), cmap(-mlxarr.inf))
    assert_array_equal(cmap.get_over(), cmap(mlxarr.inf))
    assert_array_equal(cmap.get_bad(), cmap(mlxarr.nan))


@pytest.mark.parametrize('kind', ('over', 'under', 'bad'))
def test_non_mutable_get_values(kind):
    cmap = copy.copy(mpl.colormaps['viridis'])
    init_value = getattr(cmap, f'get_{kind}')()
    with pytest.warns(PendingDeprecationWarning):
        getattr(cmap, f'set_{kind}')('k')
    black_value = getattr(cmap, f'get_{kind}')()
    assert mlxarr.all(black_value == [0, 0, 0, 1])
    assert not mlxarr.all(init_value == black_value)


def test_colormap_alpha_array():
    cmap = mpl.colormaps['viridis']
    vals = [-1, 0.5, 2]  # under, valid, over
    with pytest.raises(ValueError, match="alpha is array-like but"):
        cmap(vals, alpha=[1, 1, 1, 1])
    alpha = mlxarr.array([0.1, 0.2, 0.3])
    c = cmap(vals, alpha=alpha)
    assert_array_equal(c[:, -1], alpha)
    c = cmap(vals, alpha=alpha, bytes=True)
    assert_array_equal(c[:, -1], (alpha * 255).astype(mlxarr.uint8))


def test_colormap_bad_data_with_alpha():
    cmap = mpl.colormaps['viridis']
    c = cmap(mlxarr.nan, alpha=0.5)
    assert c == (0, 0, 0, 0)
    c = cmap([0.5, mlxarr.nan], alpha=0.5)
    assert_array_equal(c[1], (0, 0, 0, 0))
    c = cmap([0.5, mlxarr.nan], alpha=[0.1, 0.2])
    assert_array_equal(c[1], (0, 0, 0, 0))
    c = cmap([[mlxarr.nan, 0.5], [0, 0]], alpha=0.5)
    assert_array_equal(c[0, 0], (0, 0, 0, 0))
    c = cmap([[mlxarr.nan, 0.5], [0, 0]], alpha=mlxarr.full((2, 2), 0.5))
    assert_array_equal(c[0, 0], (0, 0, 0, 0))


def test_2d_to_rgba():
    color = mlxarr.array([0.1, 0.2, 0.3])
    rgba_1d = mcolors.to_rgba(color.reshape(-1))
    rgba_2d = mcolors.to_rgba(color.reshape((1, -1)))
    assert rgba_1d == rgba_2d


def test_set_dict_to_rgba():
    # downstream libraries do this...
    # note we can't test this because it is not well-ordered
    # so just smoketest:
    colors = {(0, .5, 1), (1, .2, .5), (.4, 1, .2)}
    res = mcolors.to_rgba_array(colors)
    palette = {"red": (1, 0, 0), "green": (0, 1, 0), "blue": (0, 0, 1)}
    res = mcolors.to_rgba_array(palette.values())
    exp = mlxarr.eye(3)
    mlxarr.testing.assert_array_almost_equal(res[:, :-1], exp)


def test_norm_deepcopy():
    norm = mcolors.LogNorm()
    norm.vmin = 0.0002
    norm2 = copy.deepcopy(norm)
    assert norm2.vmin == norm.vmin
    assert isinstance(norm2._scale, mscale.LogScale)
    norm = mcolors.Normalize()
    norm.vmin = 0.0002
    norm2 = copy.deepcopy(norm)
    assert norm2._scale is None
    assert norm2.vmin == norm.vmin


def test_set_clim_emits_single_callback():
    data = mlxarr.array([[1, 2], [3, 4]])
    fig, ax = plt.subplots()
    image = ax.imshow(data, cmap='viridis')

    callback = unittest.mock.Mock()
    image.norm.callbacks.connect('changed', callback)

    callback.assert_not_called()

    # Call set_clim() to update the limits
    image.set_clim(1, 5)

    # Assert that only one "changed" callback is sent after calling set_clim()
    callback.assert_called_once()


def test_norm_callback():
    increment = unittest.mock.Mock(return_value=None)

    norm = mcolors.Normalize()
    norm.callbacks.connect('changed', increment)
    # Haven't updated anything, so call count should be 0
    assert increment.call_count == 0

    # Now change vmin and vmax to test callbacks
    norm.vmin = 1
    assert increment.call_count == 1
    norm.vmax = 5
    assert increment.call_count == 2
    # callback shouldn't be called if setting to the same value
    norm.vmin = 1
    assert increment.call_count == 2
    norm.vmax = 5
    assert increment.call_count == 2

    # We only want autoscale() calls to send out one update signal
    increment.call_count = 0
    norm.autoscale([0, 1, 2])
    assert increment.call_count == 1


def test_scalarmappable_norm_update():
    norm = mcolors.Normalize()
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap='plasma')
    # sm doesn't have a stale attribute at first, set it to False
    sm.stale = False
    # The mappable should be stale after updating vmin/vmax
    norm.vmin = 5
    assert sm.stale
    sm.stale = False
    norm.vmax = 5
    assert sm.stale
    sm.stale = False
    norm.clip = True
    assert sm.stale
    # change to the CenteredNorm and TwoSlopeNorm to test those
    # Also make sure that updating the norm directly and with
    # set_norm both update the Norm callback
    norm = mcolors.CenteredNorm()
    sm.norm = norm
    sm.stale = False
    norm.vcenter = 1
    assert sm.stale
    norm = mcolors.TwoSlopeNorm(vcenter=0, vmin=-1, vmax=1)
    sm.set_norm(norm)
    sm.stale = False
    norm.vcenter = 1
    assert sm.stale


@check_figures_equal()
def test_norm_update_figs(fig_test, fig_ref):
    ax_ref = fig_ref.add_subplot()
    ax_test = fig_test.add_subplot()

    z = mlxarr.arange(100).reshape((10, 10))
    ax_ref.imshow(z, norm=mcolors.Normalize(10, 90))

    # Create the norm beforehand with different limits and then update
    # after adding to the plot
    norm = mcolors.Normalize(0, 1)
    ax_test.imshow(z, norm=norm)
    # Force initial draw to make sure it isn't already stale
    fig_test.canvas.draw()
    norm.vmin, norm.vmax = 10, 90


def test_make_norm_from_scale_name():
    logitnorm = mcolors.make_norm_from_scale(
        mscale.LogitScale, mcolors.Normalize)
    assert logitnorm.__name__ == logitnorm.__qualname__ == "LogitScaleNorm"


def test_color_sequences():
    # basic access
    assert plt.color_sequences is matplotlib.color_sequences  # same registry
    assert list(plt.color_sequences) == [
        'tab10', 'tab20', 'tab20b', 'tab20c', 'Pastel1', 'Pastel2', 'Paired',
        'Accent', 'Dark2', 'Set1', 'Set2', 'Set3', 'petroff6', 'petroff8',
        'petroff10']
    assert len(plt.color_sequences['tab10']) == 10
    assert len(plt.color_sequences['tab20']) == 20

    tab_colors = [
        'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
        'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    for seq_color, tab_color in zip(plt.color_sequences['tab10'], tab_colors):
        assert mcolors.same_color(seq_color, tab_color)

    # registering
    with pytest.raises(ValueError, match="reserved name"):
        plt.color_sequences.register('tab10', ['r', 'g', 'b'])
    with pytest.raises(ValueError, match="not a valid color specification"):
        plt.color_sequences.register('invalid', ['not a color'])

    rgb_colors = ['r', 'g', 'b']
    plt.color_sequences.register('rgb', rgb_colors)
    assert plt.color_sequences['rgb'] == ['r', 'g', 'b']
    # should not affect the registered sequence because input is copied
    rgb_colors.append('c')
    assert plt.color_sequences['rgb'] == ['r', 'g', 'b']
    # should not affect the registered sequence because returned list is a copy
    plt.color_sequences['rgb'].append('c')
    assert plt.color_sequences['rgb'] == ['r', 'g', 'b']

    # unregister
    plt.color_sequences.unregister('rgb')
    with pytest.raises(KeyError):
        plt.color_sequences['rgb']  # rgb is gone
    plt.color_sequences.unregister('rgb')  # multiple unregisters are ok
    with pytest.raises(ValueError, match="Cannot unregister builtin"):
        plt.color_sequences.unregister('tab10')


def test_cm_set_cmap_error():
    sm = cm.ScalarMappable()
    # Pick a name we are pretty sure will never be a colormap name
    bad_cmap = 'AardvarksAreAwkward'
    with pytest.raises(ValueError, match=bad_cmap):
        sm.set_cmap(bad_cmap)


def test_set_cmap_mismatched_name():
    cmap = matplotlib.colormaps["viridis"].with_extremes(over='r')
    # register it with different names
    cmap.name = "test-cmap"
    matplotlib.colormaps.register(name='wrong-cmap', cmap=cmap)

    plt.set_cmap("wrong-cmap")
    cmap_returned = plt.get_cmap("wrong-cmap")
    assert cmap_returned == cmap
    assert cmap_returned.name == "wrong-cmap"


def test_cmap_alias_names():
    assert matplotlib.colormaps["gray"].name == "gray"  # original
    assert matplotlib.colormaps["grey"].name == "grey"  # alias


def test_to_rgba_array_none_color_with_alpha_param():
    # effective alpha for color "none" must always be 0 to achieve a vanishing color
    # even explicit alpha must be ignored
    c = ["blue", "none"]
    alpha = [1, 1]
    assert_array_equal(
        to_rgba_array(c, alpha), [[0., 0., 1., 1.], [0., 0., 0., 0.]]
    )


@pytest.mark.parametrize('input, expected',
                         [('red', True),
                          (('red', 0.5), True),
                          (('red', 2), False),
                          (['red', 0.5], False),
                          (('red', 'blue'), False),
                          (['red', 'blue'], False),
                          ('C3', True),
                          (('C3', 0.5), True)])
def test_is_color_like(input, expected):
    assert is_color_like(input) is expected


def test_colorizer_vmin_vmax():
    ca = mcolorizer.Colorizer()
    assert ca.vmin is None
    assert ca.vmax is None
    ca.vmin = 1
    ca.vmax = 3
    assert ca.vmin == 1.0
    assert ca.vmax == 3.0
    assert ca.norm.vmin == 1.0
    assert ca.norm.vmax == 3.0


def test_LinearSegmentedColormap_from_list_color_alpha_tuple():
    """
    GitHub issue #29042: A bug in 'from_list' causes an error
    when passing a tuple (str, float) where the string is a
    color name or grayscale value and float is an alpha value.
    """
    colors = [("red", 0.3), ("0.42", 0.1), "green"]
    cmap = mcolors.LinearSegmentedColormap.from_list("lsc", colors, N=3)
    assert_array_almost_equal(cmap([.0, 0.5, 1.]), to_rgba_array(colors))


@pytest.mark.parametrize("colors",
                         [[(0.42, "blue"), (.1, .1, .1, .1)],
                          ["blue", (0.42, "red")],
                          ["blue", (.1, .1, .1, .1), ("red", 2)],
                          [(0, "red"), (1.1, "blue")],
                          [(0.52, "red"), (0.42, "blue")]])
def test_LinearSegmentedColormap_from_list_invalid_inputs(colors):
    with pytest.raises(ValueError):
        mcolors.LinearSegmentedColormap.from_list("lsc", colors)


def test_LinearSegmentedColormap_from_list_value_color_tuple():
    value_color_tuples = [(0, "red"), (0.6, "blue"), (1, "green")]
    cmap = mcolors.LinearSegmentedColormap.from_list("lsc", value_color_tuples, N=11)
    assert_array_almost_equal(
        cmap([value for value, _ in value_color_tuples]),
        to_rgba_array([color for _, color in value_color_tuples]),
    )


@image_comparison(['test_norm_abc.png'], remove_text=True,
                   tol=0 if platform.machine() == 'x86_64' else 0.05)
def test_norm_abc():

    class CustomHalfNorm(mcolors.Norm):
        def __init__(self):
            super().__init__()

        @property
        def vmin(self):
            return 0

        @property
        def vmax(self):
            return 1

        @property
        def clip(self):
            return False

        def __call__(self, value, clip=None):
            return value / 2

        def inverse(self, value):
            return 2 * value

        def autoscale(self, A):
            pass

        def autoscale_None(self, A):
            pass

        def scaled(self):
            return True

        def n_components(self):
            return 1

    fig, axes = plt.subplots(2,2)

    r = mlxarr.linspace(-1, 3, 16*16).reshape((16,16))
    norm = CustomHalfNorm()
    colorizer = mpl.colorizer.Colorizer(cmap='viridis', norm=norm)
    c = axes[0,0].imshow(r, colorizer=colorizer)
    axes[0,1].pcolor(r, colorizer=colorizer)
    axes[1,0].contour(r, colorizer=colorizer)
    axes[1,1].contourf(r, colorizer=colorizer)


def test_close_error_name():
    with pytest.raises(
        KeyError,
        match=(
            "'grays' is not a valid value for colormap. "
            "Did you mean one of ['gray', 'Grays', 'gray_r']?"
        )):
        matplotlib.colormaps["grays"]


def test_multi_norm_creation():
    # tests for mcolors.MultiNorm

    # test wrong input
    with pytest.raises(ValueError,
                       match="MultiNorm must be assigned an iterable"):
        mcolors.MultiNorm("linear")
    with pytest.raises(ValueError,
                       match="MultiNorm must be assigned at least one"):
        mcolors.MultiNorm([])
    with pytest.raises(ValueError,
                       match="MultiNorm must be assigned an iterable"):
        mcolors.MultiNorm(None)
    with pytest.raises(ValueError,
                       match="not a valid"):
        mcolors.MultiNorm(["linear", "bad_norm_name"])
    with pytest.raises(ValueError,
                       match="Each norm assigned to MultiNorm"):
        mcolors.MultiNorm(["linear", object()])

    norm = mpl.colors.MultiNorm(['linear', 'linear'])


def test_multi_norm_call_vmin_vmax():
    # test get vmin, vmax
    norm = mpl.colors.MultiNorm(['linear', 'log'])
    norm.vmin = (1, 1)
    norm.vmax = (2, 2)
    assert norm.vmin == (1, 1)
    assert norm.vmax == (2, 2)

    with pytest.raises(ValueError, match="Expected an iterable of length 2"):
        norm.vmin = 1
    with pytest.raises(ValueError, match="Expected an iterable of length 2"):
        norm.vmax = 1
    with pytest.raises(ValueError, match="Expected an iterable of length 2"):
        norm.vmin = (1, 2, 3)
    with pytest.raises(ValueError, match="Expected an iterable of length 2"):
        norm.vmax = (1, 2, 3)


def test_multi_norm_call_clip_inverse():
    # test get vmin, vmax
    norm = mpl.colors.MultiNorm(['linear', 'log'])
    norm.vmin = (1, 1)
    norm.vmax = (2, 2)

    # test call with clip
    assert_array_equal(norm([3, 3], clip=[False, False]), [2.0, 1.584962500721156])
    assert_array_equal(norm([3, 3], clip=[True, True]), [1.0, 1.0])
    assert_array_equal(norm([3, 3], clip=[True, False]), [1.0, 1.584962500721156])
    norm.clip = [False, False]
    assert_array_equal(norm([3, 3]), [2.0, 1.584962500721156])
    norm.clip = [True, True]
    assert_array_equal(norm([3, 3]), [1.0, 1.0])
    norm.clip = [True, False]
    assert_array_equal(norm([3, 3]), [1.0, 1.584962500721156])
    norm.clip = [True, True]

    with pytest.raises(ValueError, match="Expected an iterable of length 2"):
        norm.clip = True
    with pytest.raises(ValueError, match="Expected an iterable of length 2"):
        norm.clip = [True, False, True]
    with pytest.raises(ValueError, match="Expected an iterable of length 2"):
        norm([3, 3], clip=True)
    with pytest.raises(ValueError, match="Expected an iterable of length 2"):
        norm([3, 3], clip=[True, True, True])

    # test inverse
    assert_array_almost_equal(norm.inverse([0.5, 0.5849625007211562]), [1.5, 1.5])


def test_multi_norm_autoscale():
    norm = mpl.colors.MultiNorm(['linear', 'log'])
    # test autoscale
    norm.autoscale([[0, 1, 2, 3], [0.1, 1, 2, 3]])
    assert_array_equal(norm.vmin, [0, 0.1])
    assert_array_equal(norm.vmax, [3, 3])

    # test autoscale_none
    norm0 = mcolors.TwoSlopeNorm(2, vmin=0, vmax=None)
    norm = mcolors.MultiNorm([norm0, 'linear'], vmax=[None, 50])
    norm.autoscale_None([[1, 2, 3, 4, 5], [-50, 1, 0, 1, 500]])
    assert_array_equal(norm([5, 0]), [1, 0.5])
    assert_array_equal(norm.vmin, (0, -50))
    assert_array_equal(norm.vmax, (5, 50))


def test_mult_norm_call_types():
    mn = mpl.colors.MultiNorm(['linear', 'linear'])
    mn.vmin = (-2, -2)
    mn.vmax = (2, 2)

    vals = mlxarr.arange(6).reshape((3,2))
    target = mlxarr.ma.array([(0.5, 0.75),
                          (1., 1.25),
                          (1.5, 1.75)])

    # test structured array as input
    from_mn = mn(rfn.unstructured_to_structured(vals))
    assert_array_almost_equal(from_mn,
                              target.T)

    # test list of arrays as input
    assert_array_almost_equal(mn(list(vals.T)),
                              list(target.T))
    # test list of floats as input
    assert_array_almost_equal(mn(list(vals[0])),
                              list(target[0]))
    # test tuple of arrays as input
    assert_array_almost_equal(mn(tuple(vals.T)),
                              list(target.T))

    # mlxarr.arrays of shapes that are compatible
    assert_array_almost_equal(mn(mlxarr.zeros(2)),
                              0.5*mlxarr.ones(2))
    assert_array_almost_equal(mn(mlxarr.zeros((2, 3))),
                              0.5*mlxarr.ones((2, 3)))
    assert_array_almost_equal(mn(mlxarr.zeros((2, 3, 4))),
                              0.5*mlxarr.ones((2, 3, 4)))

    # test with NoNorm, list as input
    mn_no_norm = mpl.colors.MultiNorm(['linear', mcolors.NoNorm()])
    no_norm_out = mn_no_norm(list(vals.T))
    assert_array_almost_equal(no_norm_out,
                              [[0., 0.5, 1.],
                               [1, 3, 5]])
    assert no_norm_out[0].dtype == mlxarr.dtype('float64')
    assert no_norm_out[1].dtype == vals.dtype

    # test with NoNorm, structured array as input
    mn_no_norm = mpl.colors.MultiNorm(['linear', mcolors.NoNorm()])
    no_norm_out = mn_no_norm(rfn.unstructured_to_structured(vals))
    assert_array_almost_equal(no_norm_out,
                              [[0., 0.5, 1.],
                               [1, 3, 5]])

    # test single int as input
    with pytest.raises(ValueError,
                       match="component as input, but got 1 instead"):
        mn(1)

    # test list of incompatible size
    with pytest.raises(ValueError,
                       match="but got a sequence with 3 elements"):
        mn([3, 2, 1])

    # last axis matches, len(data.shape) > 2
    with pytest.raises(ValueError,
                       match=(r"`data_as_list = \[data\[..., i\] for i in "
                              r"range\(data.shape\[-1\]\)\]`")):
        mn(mlxarr.zeros((3, 3, 2)))

    # last axis matches, len(data.shape) == 2
    with pytest.raises(ValueError,
                       match=r"You can use `data_transposed = data.T` to convert"):
        mn(mlxarr.zeros((3, 2)))

    # incompatible arrays where no relevant axis matches
    for data in [mlxarr.zeros(3), mlxarr.zeros((3, 2, 3))]:
        with pytest.raises(ValueError,
                           match=r"but got a sequence with 3 elements"):
            mn(data)

    # test incompatible class
    with pytest.raises(ValueError,
                       match="but got <object object"):
        mn(object())
