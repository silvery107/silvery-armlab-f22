import numpy as np
DTYPE = np.float32


def recover_homogenous_affine_transformation(p, p_prime):
    '''points_transformed_1 = points_transformed_1 = np.dot(
    A1, np.transpose(np.column_stack((points_camera, (1, 1, 1, 1)))))np.dot(
    A1, np.transpose(np.column_stack((points_camera, (1, 1, 1, 1)))))
    Find the unique homogeneous affine transformation that
    maps a set of 3 points to another set of 3 points in 3D
    space:

        p_prime == np.dot(p, R) + t

    where `R` is an unknown rotation matrix, `t` is an unknown
    translation vector, and `p` and `p_prime` are the original
    and transformed set of points stored as row vectors:

        p       = np.array((p1,       p2,       p3))
        p_prime = np.array((p1_prime, p2_prime, p3_prime))

    The result of this function is an augmented 4-by-4
    matrix `A` that represents this affine transformation:

        np.column_stack((p_prime, (1, 1, 1))) == \
            np.dot(np.column_stack((p, (1, 1, 1))), A)

    Source: https://math.stackexchange.com/a/222170 (robjohn)
    '''

    # construct intermediate matrix
    Q = p[1:] - p[0]
    Q_prime = p_prime[1:] - p_prime[0]

    # calculate rotation matrix
    R = np.dot(np.linalg.inv(np.row_stack((Q, np.cross(*Q)))),
               np.row_stack((Q_prime, np.cross(*Q_prime))))

    # calculate translation vector
    t = p_prime[0] - np.dot(p[0], R)

    # calculate affine transformation matrix
    return np.transpose(np.column_stack((np.row_stack((R, t)), (0, 0, 0, 1))))


def recover_homogeneous_transform_svd(m, d):
    ''' 
    finds the rigid body transform that maps m to d: 
    d == np.dot(m,R) + T
    http://graphics.stanford.edu/~smr/ICP/comparison/eggert_comparison_mva97.pdf
    '''
    # calculate the centroid for each set of points
    d_bar = np.sum(d, axis=0) / np.shape(d)[0]
    m_bar = np.sum(m, axis=0) / np.shape(m)[0]

    # we are using row vectors, so tanspose the first one
    # H should be 3x3, if it is not, we've done this wrong
    H = np.dot(np.transpose(d - d_bar), m - m_bar)
    [U, S, V] = np.linalg.svd(H)

    R = np.matmul(V, np.transpose(U))
    # if det(R) is -1, we've made a reflection, not a rotation
    # fix it by negating the 3rd column of V
    if np.linalg.det(R) < 0:
        V = [1, 1, -1] * V
        R = np.matmul(V, np.transpose(U))
    T = d_bar - np.dot(m_bar, R)
    return np.transpose(np.column_stack((np.row_stack((R, T)), (0, 0, 0, 1))))

class Quaternion:
    def __init__(self, w=1, x=0, y=0, z=0):
        self.w = float(w)
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self._norm = np.sqrt(self.w*self.w+self.x*self.x+self.y*self.y+self.z*self.z, dtype=DTYPE)

    def toNumpy(self):
        """convert to an (4,1) numpy array"""
        return np.array([self.w,self.x,self.y,self.z], dtype=DTYPE).reshape((4,1))
    
    def unit(self):
        """return the unit quaternion"""
        return Quaternion(self.w/self._norm,self.x/self._norm,self.y/self._norm,self.z/self._norm)

    def conjugate(self):
        return Quaternion(self.w, -self.x, -self.y, -self.z)
    
    def reverse(self):
        """return the reverse rotation representation as the same as the transpose op of rotation matrix"""
        return Quaternion(-self.w,self.x,self.y,self.z)

    def inverse(self):
        return Quaternion(self.w/(self._norm*self._norm),-self.x/(self._norm*self._norm),-self.y/(self._norm*self._norm),-self.z/(self._norm*self._norm))
    
    def __str__(self):
        return '['+str(self.w)+', '+str(self.x)+', '+str(self.y)+', '+str(self.z)+']'


def rot_to_quat(rot):
    """
    * Convert a coordinate transformation matrix to an orientation quaternion.
    """
    q = Quaternion()
    r = rot.T.copy() # important
    tr = np.trace(r)
    if tr>0.0:
        S = math.sqrt(tr + 1.0) * 2.0
        q.w = 0.25 * S
        q.x = (r[2,1] - r[1,2])/S
        q.y = (r[0,2] - r[2,0])/S
        q.z = (r[1,0] - r[0,1])/S

    elif (r[0, 0] > r[1, 1]) and (r[0, 0] > r[2, 2]):
        S = math.sqrt(1.0 + r[0,0] - r[1,1] - r[2,2]) * 2.0
        q.w = (r[2,1] - r[1,2])/S
        q.x = 0.25 * S
        q.y = (r[0,1] + r[1,0])/S
        q.z = (r[0,2] + r[2,0])/S

    elif r[1,1]>r[2,2]:
        S = math.sqrt(1.0 + r[1,1] -r[0,0] -r[2,2]) * 2.0
        q.w = (r[0,2] - r[2,0])/S
        q.x = (r[0,1] + r[1,0])/S
        q.y = 0.25 * S
        q.z = (r[1,2] + r[2,1])/S
        
    else:
        S = math.sqrt(1.0 + r[2,2] - r[0,0] - r[1,1]) * 2.0
        q.w = (r[1,0] - r[0,1])/S
        q.x = (r[0,2] + r[2,0])/S
        q.y = (r[1,2] + r[2,1])/S
        q.z = 0.25 * S
    
    return q

def quat_to_rpy(q):
    """
    * Convert a quaternion to RPY. Return
    * angles in (roll, pitch, yaw).
    """
    rpy = np.zeros((3,1), dtype=DTYPE)
    as_ = np.min([-2.*(q.x*q.z-q.w*q.y),.99999])
    # roll
    rpy[0] = np.arctan2(2.*(q.y*q.z+q.w*q.x), q.w*q.w - q.x*q.x - q.y*q.y + q.z*q.z)
    # pitch
    rpy[1] = np.arcsin(as_)
    # yaw
    rpy[2] = np.arctan2(2.*(q.x*q.y+q.w*q.z), q.w*q.w + q.x*q.x - q.y*q.y - q.z*q.z)
    return rpy

def rot_to_rpy(R):
    return quat_to_rpy(rot_to_quat(R))