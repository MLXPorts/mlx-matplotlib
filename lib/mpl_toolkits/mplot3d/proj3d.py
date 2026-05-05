"""
Various transforms used for by the 3D code
"""
import mlx.core as mx
from matplotlib import _api


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
    u = mx.cross(V, w)
    u = u/mx.linalg.norm(u)
    v = mx.cross(w, u)  # Will be a unit vector

    # Save some computation for the default roll=0
    if roll != 0:
        # A positive rotation of the camera is a negative rotation of the world
        Rroll = _rotation_about_vector(w, -roll)
        u = mx.dot(Rroll, u)
        v = mx.dot(Rroll, v)
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
    Mr = mx.eye(4)
    Mt = mx.eye(4)
    Mr[:3, :3] = [u, v, w]
    Mt[:3, -1] = -E
    M = mx.dot(Mr, Mt)
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
    vecw = mx.dot(M, vec.data)
    ts = vecw[0:3]/vecw[3]
    if mx.ma.isMA(vec):
        ts = mx.ma.array(ts, mask=vec.mask)
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
    vecs = vecs.reshape(-1, 3).T

    vecs_pad = mx.zeros((vecs.shape[0] + 1,) + vecs.shape[1:])
    vecs_pad[:-1] = vecs
    vecs_pad[-1] = 1
    product = mx.dot(M, vecs_pad)
    tvecs = product[:3] / product[3]

    return tvecs.T.reshape(vecs_shape)


def _proj_transform_vec_clip(vec, M, focal_length):
    vecw = mx.dot(M, vec.data)
    txs, tys, tzs = vecw[0:3] / vecw[3]
    if mx.isinf(focal_length):  # don't clip orthographic projection
        tis = mx.ones(txs.shape, dtype=bool)
    else:
        tis = (-1 <= txs) & (txs <= 1) & (-1 <= tys) & (tys <= 1) & (tzs <= 0)
    if mx.ma.isMA(vec[0]):
        tis = tis & ~vec[0].mask
    if mx.ma.isMA(vec[1]):
        tis = tis & ~vec[1].mask
    if mx.ma.isMA(vec[2]):
        tis = tis & ~vec[2].mask

    txs = mx.ma.masked_array(txs, ~tis)
    tys = mx.ma.masked_array(tys, ~tis)
    tzs = mx.ma.masked_array(tzs, ~tis)
    return txs, tys, tzs, tis


def inv_transform(xs, ys, zs, invM):
    """
    Transform the points by the inverse of the projection matrix, *invM*.
    """
    vec = _vec_pad_ones(xs, ys, zs)
    vecr = mx.dot(invM, vec)
    if vecr.shape == (4,):
        vecr = vecr.reshape((4, 1))
    for i in range(vecr.shape[1]):
        if vecr[3][i] != 0:
            vecr[:, i] = vecr[:, i] / vecr[3][i]
    return vecr[0], vecr[1], vecr[2]


def _vec_pad_ones(xs, ys, zs):
    if mx.ma.isMA(xs) or mx.ma.isMA(ys) or mx.ma.isMA(zs):
        return mx.ma.array([xs, ys, zs, mx.ones_like(xs)])
    else:
        return mx.array([xs, ys, zs, mx.ones_like(xs)])


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
    return mx.column_stack(_proj_trans_points(points, M))


def _proj_trans_points(points, M):
    points = mx.asarray(points)
    xs, ys, zs = points[:, 0], points[:, 1], points[:, 2]
    return proj_transform(xs, ys, zs, M)
