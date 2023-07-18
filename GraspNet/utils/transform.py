import numpy as np
from math import sqrt

def quaternion_to_rotation_matrix(quat):
  q = quat.copy()
  n = np.dot(q, q)
  if n < np.finfo(q.dtype).eps:
    return np.identity(4)
  q = q * np.sqrt(2.0 / n)
  q = np.outer(q, q)
  rot_matrix = np.array(
    [[1.0 - q[2, 2] - q[3, 3], q[1, 2] + q[3, 0], q[1, 3] - q[2, 0], 0.0],
     [q[1, 2] - q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] + q[1, 0], 0.0],
     [q[1, 3] + q[2, 0], q[2, 3] - q[1, 0], 1.0 - q[1, 1] - q[2, 2], 0.0],
     [0.0, 0.0, 0.0, 1.0]],
    dtype=q.dtype)
  return rot_matrix

def rotation_matrix_to_quaternion(R):
    t = R.flatten()
    w = sqrt(t[0]+t[4]+t[8]+1)/2
    x = sqrt(t[0]-t[4]-t[8]+1)/2
    y = sqrt(-t[0]+t[4]-t[8]+1)/2
    z = sqrt(-t[0]-t[4]+t[8]+1)/2
    a = [w,x,y,z]
    m = a.index(max(a))
    if m == 0:
        x = (t[7]-t[5])/(4*w)
        y = (t[2]-t[6])/(4*w)
        z = (t[3]-t[1])/(4*w)
    if m == 1:
        w = (t[7]-t[5])/(4*x)
        y = (t[1]+t[3])/(4*x)
        z = (t[6]+t[2])/(4*x)
    if m == 2:
        w = (t[2]-t[6])/(4*y)
        x = (t[1]+t[3])/(4*y)
        z = (t[5]+t[7])/(4*y)
    if m == 3:
        w = (t[3]-t[1])/(4*z)
        x = (t[6]+t[2])/(4*z)
        y = (t[5]+t[7])/(4*z)
    return np.asarray([w, x, y, z])

def aug_label(self, label, R, trans):
    augmented_label = label
    Rq = quaternion_to_rotation_matrix(augmented_label[:4])
    # augmented_points = np.sum(np.expand_dims(points, 2) * R, axis=1) + noise + trans

    return augmented_label