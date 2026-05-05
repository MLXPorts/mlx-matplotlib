"""
Various transforms used for by the 3D code
"""
import mlx.core as mx
from matplotlib import _api


def _cross3(a, b):
    stream = mx.cpu if a.dtype == mx.float64 or b.dtype == mx.float64 else None
    return mx.stack([
        mx.subtract(mx.multiply(a[1], b[2], stream=stream),
                    mx.multiply(a[2], b[1], stream=stream), stream=stream),
        mx.subtract(mx.multiply(a[2], b[0], stream=stream),
                    mx.multiply(a[0], b[2], stream=stream), stream=stream),
        mx.subtract(mx.multiply(a[0], b[1], stream=stream),
                    mx.multiply(a[1], b[0], stream=stream), stream=stream),
    ], stream=stream)


def world_transformation(xmin, xmax,
                         ymin, ymax,
                         zmin, zmax, pb_aspect=None):
    """
    Produce a matrix that scales homogeneous coords in the specified ranges
    to [0, 1], or [0, pb_aspect[i]] if the plotbox aspect ratio is specified.
    """
    dx = xmax - xmin
    dy = ymax - ymin
    dz = zmax - zmin
    if pb_aspect is not None:
        ax, ay, az = pb_aspect
        dx /= ax
        dy /= ay
        dz /= az

    return mx.array([[1/dx,    0,    0, -xmin/dx],
                     [   0, 1/dy,    0, -ymin/dy],
                     [   0,    0, 1/dz, -zmin/dz],
                     [   0,    0,    0,        1]])


def _rotation_about_vector(v, angle):
    """
    Produce a rotation matrix for an angle in radians about a vector.
    """
    vx, vy, vz = v / mx.linalg.norm(v)
    s = mx.sin(angle)
    c = mx.cos(angle)
    t = 2*mx.sin(angle/2)**2  # more numerically stable than t = 1-c

    R = mx.array([
        [t*vx*vx + c,    t*vx*vy - vz*s, t*vx*vz + vy*s],
        [t*vy*vx + vz*s, t*vy*vy + c,    t*vy*vz - vx*s],
        [t*vz*vx - vy*s, t*vz*vy + vx*s, t*vz*vz + c]])

    return R


def _view_axes(E, R, V, roll):
    """
    Get the unit viewing axes in data coordinates.

    Parameters
    ----------
    E : 3-element array_backend array
        The coordinates of the eye/camera.
    R : 3-element array_backend array
        The coordinates of the center of the view box.
    V : 3-element array_backend array
        Unit vector in the direction of the vertical axis.
    roll : float
        The roll angle in radians.

    Returns
    -------
    u : 3-element array_backend array
        Unit vector pointing towards the right of the screen.
    v : 3-element array_backend array
        Unit vector pointing towards the top of the screen.
    w : 3-element array_backend array
        Unit vector pointing out of the screen.
    """
    w = (E - R)
    w = w/mx.linalg.norm(w)
    u = _cross3(V, w)
    u = u/mx.linalg.norm(u)
    v = _cross3(w, u)  # Will be a unit vector

    # Save some computation for the default roll=0
    if roll != 0:
        # A positive rotation of the camera is a negative rotation of the world
        Rroll = _rotation_about_vector(w, -roll)
        u = mx.matmul(Rroll, u)
        v = mx.matmul(Rroll, v)
    return u, v, w


def _view_transformation_uvw(u, v, w, E):
    """
    Return the view transformation matrix.

    Parameters
    ----------
    u : 3-element array_backend array
        Unit vector pointing towards the right of the screen.
    v : 3-element array_backend array
        Unit vector pointing towards the top of the screen.
    w : 3-element array_backend array
        Unit vector pointing out of the screen.
    E : 3-element array_backend array
        The coordinates of the eye/camera.
    """
    dtype = u.dtype
    stream = mx.cpu if dtype == mx.float64 else None
    zeros_col = mx.zeros((3, 1), dtype=dtype, stream=stream)
    bottom = mx.array([[0, 0, 0, 1]], dtype=dtype)
    Mr = mx.concatenate([
        mx.concatenate([mx.stack([u, v, w], axis=0), zeros_col], axis=1),
        bottom], axis=0)
    Mt = mx.concatenate([
        mx.concatenate([mx.eye(3, dtype=dtype, stream=stream),
                        mx.reshape(mx.negative(E, stream=stream), (3, 1))],
                       axis=1),
        bottom], axis=0)
    M = mx.matmul(Mr, Mt)
    return M


def _persp_transformation(zfront, zback, focal_length):
    e = focal_length
    a = 1  # aspect ratio
    b = (zfront+zback)/(zfront-zback)
    c = -2*(zfront*zback)/(zfront-zback)
    proj_matrix = mx.array([[e,   0,  0, 0],
                            [0, e/a,  0, 0],
                            [0,   0,  b, c],
                            [0,   0, -1, 0]])
    return proj_matrix


def _ortho_transformation(zfront, zback):
    # note: w component in the resulting vector will be (zback-zfront), not 1
    a = -(zfront + zback)
    b = -(zfront - zback)
    proj_matrix = mx.array([[2, 0,  0, 0],
                            [0, 2,  0, 0],
                            [0, 0, -2, 0],
                            [0, 0,  a, b]])
    return proj_matrix


def _proj_transform_vec(vec, M):
    vecw = mx.matmul(M, vec)
    ts = mx.divide(vecw[0:3], vecw[3])
    return ts[0], ts[1], ts[2]


def _proj_transform_vectors(vecs, M):
    """
    Vectorized version of ``_proj_transform_vec``.

    Parameters
    ----------
    vecs : ... x 3 mx.array
        Input vectors
    M : 4 x 4 mx.array
        Projection matrix
    """
    vecs_shape = vecs.shape
    vecs = mx.transpose(mx.reshape(vecs, (-1, 3)))
    vecs_pad = mx.concatenate(
        [vecs, mx.ones((1, vecs.shape[1]), dtype=vecs.dtype)], axis=0)
    product = mx.matmul(M, vecs_pad)
    tvecs = mx.divide(product[:3], product[3])

    return mx.reshape(mx.transpose(tvecs), vecs_shape)


def _proj_transform_vec_clip(vec, M, focal_length):
    vecw = mx.matmul(M, vec)
    txs, tys, tzs = mx.divide(vecw[0:3], vecw[3])
    if focal_length == mx.inf:  # don't clip orthographic projection
        tis = mx.ones(txs.shape, dtype=mx.bool_)
    else:
        tis = mx.logical_and(
            mx.logical_and(mx.greater_equal(txs, -1), mx.less_equal(txs, 1)),
            mx.logical_and(
                mx.logical_and(mx.greater_equal(tys, -1), mx.less_equal(tys, 1)),
                mx.less_equal(tzs, 0)))
    return txs, tys, tzs, tis


def inv_transform(xs, ys, zs, invM):
    """
    Transform the points by the inverse of the projection matrix, *invM*.
    """
    vec = _vec_pad_ones(xs, ys, zs)
    vecr = mx.matmul(invM, vec)
    if vecr.shape == (4,):
        vecr = vecr.reshape((4, 1))
    for i in range(vecr.shape[1]):
        if vecr[3][i] != 0:
            vecr[:, i] = vecr[:, i] / vecr[3][i]
    return vecr[0], vecr[1], vecr[2]


def _vec_pad_ones(xs, ys, zs):
    xs = xs if isinstance(xs, mx.array) else mx.array(xs)
    ys = ys if isinstance(ys, mx.array) else mx.array(ys)
    zs = zs if isinstance(zs, mx.array) else mx.array(zs)
    return mx.stack([xs, ys, zs, mx.ones_like(xs)], axis=0)


def proj_transform(xs, ys, zs, M):
    """
    Transform the points by the projection matrix *M*.
    """
    vec = _vec_pad_ones(xs, ys, zs)
    return _proj_transform_vec(vec, M)


@_api.deprecated("3.10")
def proj_transform_clip(xs, ys, zs, M):
    return _proj_transform_clip(xs, ys, zs, M, focal_length=mx.inf)


def _proj_transform_clip(xs, ys, zs, M, focal_length):
    """
    Transform the points by the projection matrix
    and return the clipping result
    returns txs, tys, tzs, tis
    """
    vec = _vec_pad_ones(xs, ys, zs)
    return _proj_transform_vec_clip(vec, M, focal_length)


def _proj_points(points, M):
    return mx.stack(_proj_trans_points(points, M), axis=1)


def _proj_trans_points(points, M):
    points = points if isinstance(points, mx.array) else mx.array(points)
    xs, ys, zs = points[:, 0], points[:, 1], points[:, 2]
    return proj_transform(xs, ys, zs, M)
