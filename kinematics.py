"""!
Implements Forward and Inverse kinematics with DH parametrs and product of exponentials

TODO: Here is where you will write all of your kinematics functions
There are some functions to start with, you may need to implement a few more
"""

import math
import numpy as np
# expm is a matrix exponential function
from scipy.linalg import expm

from utils import DTYPE, Quaternion
from math import sin, cos


def clamp(angle):
    """!
    @brief      Clamp angles between (-pi, pi]

    @param      angle  The angle

    @return     Clamped angle
    """
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle <= -np.pi:
        angle += 2 * np.pi
    return angle


def FK_dh(dh_params, joint_angles, link):
    """!
    @brief      Get the 4x4 transformation matrix from link to world

                TODO: implement this function

                Calculate forward kinematics for rexarm using DH convention

                return a transformation matrix representing the pose of the desired link

                note: phi is the euler angle about the y-axis in the base frame

    @param      dh_params     The dh parameters as a 2D list each row represents a link and has the format [a, alpha, d,
                              theta]
    @param      joint_angles  The joint angles of the links
    @param      link          The link to transform from

    @return     a transformation matrix representing the pose of the desired link
    """
    H = np.identity(4, dtype=DTYPE)
    for idx, t in enumerate(joint_angles):
        a, alpha, d, theta = dh_params[idx]
        if alpha == -1:
            alpha = t
        elif theta == -1:
            theta = t
        A = get_transform_from_dh(a, alpha, d, theta)
        H = np.matmul(H, A)

    pose = get_pose_from_T(H)
    rpy = get_euler_angles_from_T(H)

    return np.array([pose[0], pose[1], pose[2], rpy[1]], dtype=DTYPE)


def get_transform_from_dh(a, alpha, d, theta):
    """!
    @brief      Gets the transformation matrix from dh parameters.

    TODO: Find the T matrix from a row of a DH table

    @param      a      a meters
    @param      alpha  alpha radians
    @param      d      d meters
    @param      theta  theta radians

    @return     The 4x4 transform matrix.
    """
    Rot1 = np.array([[cos(theta), -sin(theta), 0, 0],
                     [sin(theta), cos(theta), 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]], dtype=DTYPE)
    Trans1 = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, d],
                       [0, 0, 0, 1]], dtype=DTYPE)
    Trans2 = np.array([[1, 0, 0, a],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]], dtype=DTYPE)
    Rot2 = np.array([[1, 0, 0, 0],
                     [0, cos(alpha), -sin(alpha), 0],
                     [0, sin(alpha), cos(alpha), 0],
                     [0, 0, 0, 1]], dtype=DTYPE)
    T = np.matmul(np.matmul(np.matmul(Rot1, Trans1), Trans2), Rot2)
    return T


def get_euler_angles_from_T(T):
    """!
    @brief      Gets the euler angles from a transformation matrix.

                TODO: Implement this function return the Euler angles from a T matrix

    @param      T     transformation matrix

    @return     The euler angles from T.
    """
    R = T[:3, :3]
    if abs(R[2, 2]) > 1:
        phi = 0.0
    else:
        phi = np.arctan2(np.sqrt(1-R[2, 2]*R[2, 2]), R[2, 2])
    
    # print(phi)
    # rpy = rot_to_rpy(T[:3, :3]).flatten()
    # return rpy
    return [0, abs(phi), 0]


def get_pose_from_T(T):
    """!
    @brief      Gets the pose from T.

                TODO: implement this function return the joint pose from a T matrix of the form (x,y,z,phi) where phi is
                rotation about base frame y-axis

    @param      T     transformation matrix

    @return     The pose from T.
    """
    return T[:3, 3].flatten()


def FK_pox(joint_angles, m_mat, s_lst):
    """!
    @brief      Get a 4-tuple (x, y, z, phi) representing the pose of the desired link

                TODO: implement this function, Calculate forward kinematics for rexarm using product of exponential
                formulation return a 4-tuple (x, y, z, phi) representing the pose of the desired link note: phi is the euler
                angle about y in the base frame

    @param      joint_angles  The joint angles
                m_mat         The M matrix
                s_lst         List of screw vectors

    @return     a 4-tuple (x, y, z, phi) representing the pose of the desired link
    """
    T = np.identity(4, dtype=DTYPE)
    for idx, t in enumerate(joint_angles):
        w = s_lst[idx,0:3]
        v = s_lst[idx,3:]
        
        wmat = to_w_matrix(w)
        smat = to_s_matrix(wmat,v)
        est = expm(smat * t)
        T = np.matmul(T, est)
    
    T = np.matmul(T,m_mat)
    pose = get_pose_from_T(T)
    rpy = get_euler_angles_from_T(T)

    return np.array([pose[0], pose[1], pose[2], rpy[1]], dtype=DTYPE)


def to_w_matrix(w):
    wmat = np.array([[0, -w[2], w[1]],
                     [w[2], 0, -w[0]],
                     [-w[1], w[0], 0]], dtype=DTYPE)
    return wmat


def to_s_matrix(w, v):
    """!
    @brief      Convert to s matrix.

    TODO: implement this function
    Find the [s] matrix for the POX method e^([s]*theta)

    @param      w     { parameter_description }
    @param      v     { parameter_description }

    @return     { description_of_the_return_value }
    """
    smat = np.column_stack((w, v))
    smat = np.row_stack((smat, np.array([0,0,0,0])))
    return smat


def IK_geometric(pose, dh_params=None, m_matrix=None, s_list=None):
    """!
    @brief      Get all possible joint configs that produce the pose.

                TODO: Convert a desired end-effector pose as np.array x,y,z,phi to joint angles

    @param      dh_params  The dh parameters
    @param      pose       The desired pose as np.array x,y,z,phi

    @return     All four possible joint configurations in a numpy array 4x4 where each row is one possible joint
                configuration
    """
    x, y, z, phi = pose
    l1 = 206.155
    # l2 = 331
    l2 = 408.575 - 50
    base_offset = 104.57
    x_c = np.sqrt(x*x + y*y)
    y_c = z - base_offset
    t1 = np.arctan2(y, x)
    t2_offset = np.arctan2(50, 200)
    t3_offset = np.pi/2 - t2_offset
    t3 = - math.acos((x_c*x_c + y_c*y_c - l1*l1 - l2*l2)/(2*l1*l2))
    t2 = np.arctan2(y_c, x_c) - np.arctan2(l2*np.sin(t3), l1 + l2*np.cos(t3))
    t2 = np.pi/2 - t2
    t2 -= t2_offset
    t3 += t3_offset
    t3 = -t3
    
    # t4 = 0 # TODO 
    t4 = phi - (t2 + t3)
    # assert t4 > 0
    t5 = 0 # TODO this should be set to block theta or 0
    joint_angles = np.array([t1, t2, t3, t4, t5]).reshape((1, -1))

    if m_matrix is None or s_list is None:
        return joint_angles

    # !!! Test IK with FK
    fk_pose = FK_pox(joint_angles, m_matrix, s_list)
    vclamp = np.vectorize(clamp)
    compare = vclamp(fk_pose - pose)
    print('Pose: {} '.format(pose))
    if np.allclose(compare, np.zeros_like(compare), rtol=1e-3, atol=1e-4):
        print('FK Pose: {}'.format(fk_pose))
        print('Pose matches with FK')
        return joint_angles
    else:
        print('No match to the IK pose found! Go home!')
        return np.zeros((1, 5))

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

