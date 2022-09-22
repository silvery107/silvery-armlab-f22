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
        if idx == 0:
            value = t
        elif idx == 1:
            value = np.pi/2.0 - t
        elif idx == 2:
            value = np.pi/2.0 - t
        elif idx == 3:
            value = t - np.pi/2.0
        elif idx == 4:
            value = np.pi/2.0

        if alpha == -1:
            alpha = t
        elif theta == -1:
            theta = t
        A = get_transform_from_dh(a, alpha, d, theta)
        H = np.matmul(H, A)

    pose = get_pose_from_T(H)
    return pose


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
    # designated to euler ypr angle
    rpy = rot_to_rpy(T[:3, :3]).flatten()
    return rpy

def get_pose_from_T(T):
    """!
    @brief      Gets the pose from T.

                TODO: implement this function return the joint pose from a T matrix of the form (x,y,z,phi) where phi is
                rotation about base frame y-axis

    @param      T     transformation matrix

    @return     The pose from T.
    """
    # returns the pose : (x,y,z,phi) required for FK output and IK input
    xyz = T[:3, 3].flatten()
    
    R = T[:3, :3]
    if abs(R[2, 2]) > 1:
        phi = 0.0
    else:
        phi = np.arctan2(np.sqrt(1-R[2, 2]*R[2, 2]), R[2, 2])

    pose = np.append(xyz, phi)
    return pose


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
    rot = np.array([[0, -1, 0, 0],
                    [1, 0, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])
    T = np.matmul(rot,T)
    pose = get_pose_from_T(T)
    return pose


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

# def IK_multireach(pose, dh_params=None, m_matrix=None, s_list=None):
#     reachable_1, joint_angles_1 = IK_geometric([above_world_pos[0], 
#                                     above_world_pos[1], 
#                                     above_world_pos[2], 
#                                     np.pi/2],
#                                     m_matrix=self.rxarm.M_matrix,
#                                     s_list=self.rxarm.S_list)


def IK_geometric(pose, dh_params=None, m_matrix=None, s_list=None):
    """!
    @brief      Get all possible joint configs that produce the pose.

                TODO: Convert a desired end-effector pose as np.array x,y,z,phi to joint angles

    @param      dh_params  The dh parameters
    @param      pose       The desired pose as np.array x,y,z,phi

    @return     All four possible joint configurations in a numpy array 4x4 where each row is one possible joint
                configuration
    """
    l1 = 104.57                 # from t1 to t2, aka base offset
    l2 = np.sqrt(200*200+50*50) # from t2 to t3, shoulder to elbow shortest distance
    l3 = 200                    # from t3 to t4, elbow to wrist
    l4 = 408.575 - 200 - 50     # from t4 to ee, center of gripper (?)
    t_offset = np.arctan2(50, 200) # offset angle bewteen t3 and t2

    # two cases for t1
    theta1 = np.arctan2(-pose[0], pose[1])

    phi = pose[3]
    # r: orientation of l4 w.r.t. origion
    l4_unit = np.array([-np.sin(theta1)*np.cos(phi), np.cos(theta1)*np.cos(phi), -np.sin(phi)])
    xc, yc, zc = pose[0:3] - l4*l4_unit # xyz of the wrist (t4)
    # print((xc,yc,zc))
    if np.sqrt(xc*xc + yc*yc + (zc - l1)*(zc - l1)) > (l2 + l3):
        print("Pose can't reach!!! Go home!!")
        return False, [0, 0, 0, 0, 0]

    r = np.sqrt(xc*xc + yc*yc)   # (r, s) are planar xy of the wrist 
    s = zc - l1
    
    # two cases for t3: t3 = t3; t3 = -t3
    theta3 = - np.arccos((r*r + s*s - l2*l2 - l3*l3)/(2*l2*l3))
    theta2 = np.arctan2(s, r) - np.arctan2(l3*np.sin(theta3), l2 + l3*np.cos(theta3)) # TODO something to do with offset
    
    # t3 = t3 + t_offset - np.pi/2 # offset
    theta3 += np.pi/2 -t_offset
    theta3 = -theta3
    theta2 = np.pi/2 - t_offset - theta2 # offset

    theta4 = phi - (theta2 + theta3) # by geometry
    # assert t4 > 0

    theta5 = 0 # TODO 
    # a. vertical pick: depends on block orientation; 
    # b. hori. pick: 0 

    joint_angles = [theta1, theta2, theta3, theta4, theta5]#.reshape((1, -1))
    # print(joint_angles)

    if m_matrix is None or s_list is None:
        return joint_angles

    # !!! Test IK with FK
    print("IK angles {}".format(joint_angles))
    fk_pose = FK_pox(joint_angles, m_matrix, s_list)
    compare = fk_pose - pose
    print('Tgt Pose: {} '.format(pose))
    print('FK Pose:  {}'.format(fk_pose))
    if np.allclose(compare, np.zeros_like(compare), rtol=1e-1, atol=1e-1):
        print('Pose matches with FK')
        return True, joint_angles
    else:
        print('No match to the FK pose found! Go home!')
        return False, [0, 0, 0, 0, 0]

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

