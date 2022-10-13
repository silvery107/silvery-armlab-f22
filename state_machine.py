"""!
The state machine that implements the logic.
"""
from copy import copy, deepcopy
from PyQt4.QtCore import (QThread, Qt, pyqtSignal, pyqtSlot, QTimer)
import time
import numpy as np
from kinematics import IK_geometric
import rospy
import cv2
import math
from utils import *

class StateMachine():
    """!
    @brief      This class describes a state machine.

                TODO: Add states and state functions to this class to implement all of the required logic for the armlab
    """

    def __init__(self, rxarm, camera):
        """!
        @brief      Constructs a new instance.

        @param      rxarm   The rxarm
        @param      planner  The planner
        @param      camera   The camera
        """
        self.rxarm = rxarm
        self.camera = camera
        self.status_message = "State: Idle"
        self.current_state = "idle"
        self.next_state = "idle"
        self.waypoints = [
            [-np.pi/2,       -0.5,      -0.3,            0.0,       0.0],
            [0.75*-np.pi/2,   0.5,      0.3,      0.0,       np.pi/2],
            [0.5*-np.pi/2,   -0.5,     -0.3,     np.pi / 2,     0.0],
            [0.25*-np.pi/2,   0.5,     0.3,     0.0,       np.pi/2],
            [0.0,             0.0,      0.0,         0.0,     0.0],
            [0.25*np.pi/2,   -0.5,      -0.3,      0.0,       np.pi/2],
            [0.5*np.pi/2,     0.5,     0.3,     np.pi / 2,     0.0],
            [0.75*np.pi/2,   -0.5,     -0.3,     0.0,       np.pi/2],
            [np.pi/2,         0.5,     0.3,      0.0,     0.0],
            [0.0,             0.0,     0.0,      0.0,     0.0]]
        self.record_waypoints = []
        self.record_gripper = []
        self.waypoint_played = False
        
        self.world_pos = np.empty((3,3))
        self.pick_size = -1

    def set_next_state(self, state):
        """!
        @brief      Sets the next state.

            This is in a different thread than run so we do nothing here and let run handle it on the next iteration.

        @param      state  a string representing the next state.
        """
        self.next_state = state

    def run(self):
        """!
        @brief      Run the logic for the next state

                    This is run in its own thread.

                    TODO: Add states and funcitons as needed.
        """
        if self.next_state == "initialize_rxarm":
            self.initialize_rxarm()

        if self.next_state == "idle":
            self.idle()

        if self.next_state == "estop":
            self.estop()

        if self.next_state == "execute":
            self.execute()

        if self.next_state == "calibrate":
            self.calibrate()

        if self.next_state == "detect":
            self.detect()

        if self.next_state == "manual":
            self.manual()

        if self.next_state == "record":
            self.record()

        if self.next_state == "play":
            self.play()

        if self.next_state == "clean":
            self.clean()

        if self.next_state == "pick":
            self.pick()

        if self.next_state == "place":
            self.place()

        if self.next_state == "task_test":
            self.task_test()

        if self.next_state == "task1":
            self.task1()

        if self.next_state == "task2":
            self.task2()

        if self.next_state == "task3":
            self.task3()

        if self.next_state == "task4":
            self.task4()

        if self.next_state == "task5":
            self.task5()

    """Functions run for each state"""

    def manual(self):
        """!
        @brief      Manually control the rxarm
        """
        self.status_message = "State: Manual - Use sliders to control arm"
        self.current_state = "manual"

    def idle(self):
        """!
        @brief      Do nothing
        """
        self.status_message = "State: Idle - Waiting for input"
        self.current_state = "idle"

    def estop(self):
        """!
        @brief      Emergency stop disable torque.
        """
        self.status_message = "EMERGENCY STOP - Check rxarm and restart program"
        self.current_state = "estop"
        self.rxarm.disable_torque()

    def execute(self):
        """!
        @brief      Go through all waypoints
        TODO: Implement this function to execute a waypoint plan
              Make sure you respect estop signal
        """
        self.status_message = "State: Execute - Executing motion plan"
        self.current_state = "execute"
        
        self.rxarm.estop = False
        self.rxarm.enable_torque()
        for point in self.waypoints:
            self.rxarm.set_positions(point)
            rospy.sleep(2)
            if self.next_state == "estop":
                break
        if not self.next_state == "estop":
            self.next_state = "idle"
    
    def record(self):
        self.status_message = "State: Record - Recording waypoints"
        self.current_state = "record"
        if self.waypoint_played:
            self.record_waypoints = []
            self.record_gripper = []
            self.waypoint_played = False
        self.record_waypoints.append(self.rxarm.get_positions())
        self.record_gripper.append(self.rxarm.gripper_state)
        self.next_state = "idle"
    
    def play(self):
        self.status_message = "State: Play - Executing recorded motion plan"
        self.current_state = "play"
        
        self.rxarm.estop = False
        self.rxarm.enable_torque()
        self.waypoint_played = True
        for idx, point in enumerate(self.record_waypoints):
            gripper_state = self.record_gripper[idx]
            move_time = 2.0
            ac_time = 0.5
            
            if idx > 0:
                pre_point = self.record_waypoints[idx - 1]
                displacement = point - pre_point
                angular_t = np.abs(displacement) / (np.pi / 5)
                move_time = np.max(angular_t)
                ac_time = move_time / 4.0
            
            self.rxarm.set_joint_positions(point,
                                 moving_time=move_time,
                                 accel_time=ac_time,
                                 blocking=True)
            #rospy.sleep(0.5)
            if self.next_state == "estop":
                break
            if gripper_state != self.rxarm.gripper_state:
                if gripper_state:
                    self.rxarm.open_gripper()
                    self.rxarm.gripper_state = True
                else:
                    self.rxarm.close_gripper()
                    self.rxarm.gripper_state = False
        if not self.next_state == "estop":
            self.next_state = "idle"
    
    def calMoveTime(self, target_joint):
        displacement = target_joint - self.rxarm.get_positions()
        angular_v = np.ones(displacement.shape) * (np.pi / 4)
        angular_v[0] = np.pi / 2.5
        angular_v[3] = np.pi / 3
        angular_v[4] = np.pi / 3
        angular_t = np.abs(displacement) / angular_v
        move_time = np.max(angular_t)
        if move_time < 0.4:
            move_time = 0.4
        ac_time = move_time / 3
        return move_time, ac_time

    def pick(self):
        self.status_message = "State: Pick - Click to pick"
        self.current_state = "pick"
        self.camera.new_click = False
        print("[CLICK PICK] Please click one point to pick...")
        while not self.camera.new_click:
            rospy.sleep(0.05)
        
        self.camera.new_click = False
        pt = self.camera.last_click
        z = self.camera.DepthFrameRaw[pt[1]][pt[0]]
        click_uvd = np.append(pt, z)
        target_world_pos, block_ori = self.get_block_xyz_from_click(click_uvd)

        self.rxarm.go_to_home_pose(moving_time=2,
                                    accel_time=0.5,
                                    blocking=True)
        
        self.auto_pick(target_world_pos, block_ori)
        if not self.next_state == "estop":
            self.next_state = "idle"

    def auto_pick(self, _target_world_pos, block_ori, phi=np.pi/2, double_check=False, to_sky=False):
        if to_sky:
            _target_world_pos = [-345, 0, 0]
        target_world_pos = deepcopy(_target_world_pos)
        above_world_pos = deepcopy(_target_world_pos)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!! pick pos:", target_world_pos)
        xy_norm = np.linalg.norm(target_world_pos[:2])
        print(xy_norm)
        if xy_norm >= 315 and xy_norm<=430:
            target_world_pos[2] = target_world_pos[2] + 1/55 * xy_norm
            print(1/50 * xy_norm)
        ############ Planning #############
        # print("[PICK] Planning waypoints...")
        pick_stable = True
        pick_height_offset = 10 + 19 # + 5
        pick_wrist_offset = 0 # np.pi/18.0/3.0
        target_world_pos[2] = target_world_pos[2] + pick_height_offset
        above_world_pos[2] = above_world_pos[2] + pick_height_offset + 80

        reachable_low, reachable_high = False, False

        # Try vertical reach with phi = pi/2
        reachable_low, joint_angles_2 = IK_geometric([target_world_pos[0], 
                                                    target_world_pos[1],
                                                    target_world_pos[2],
                                                    phi],
                                                    block_ori=block_ori,
                                                    m_matrix=self.rxarm.M_matrix,
                                                    s_list=self.rxarm.S_list)
        # phi = np.pi/2
        if reachable_low:
            while not reachable_high:
                reachable_high, joint_angles_1 = IK_geometric([above_world_pos[0], 
                                                            above_world_pos[1],
                                                            above_world_pos[2],
                                                            phi],
                                                            block_ori=block_ori,
                                                            m_matrix=self.rxarm.M_matrix,
                                                            s_list=self.rxarm.S_list)
                if reachable_high:
                    break
                if above_world_pos[2] - target_world_pos[2] > 40:
                    above_world_pos[2] = above_world_pos[2] - 10
                else:
                    if 0.98* math.sqrt(above_world_pos[0] * above_world_pos[0] + above_world_pos[1] * above_world_pos[1]) > 158.875:
                        above_world_pos[0] = above_world_pos[0] * 0.98
                        above_world_pos[1] = above_world_pos[1] * 0.98
                    phi = phi - np.pi/18.0
                if phi <= 0:
                    break

        # add horizontal reach by detecting distance between the projection of arm and the target point
        if self.check_path_clean(target_world_pos):
            # Try horizontal reach with phi = 0.0
            target_world_pos = deepcopy(_target_world_pos)
            above_world_pos = deepcopy(_target_world_pos)
            target_world_pos[2] = target_world_pos[2] + 5 + 19
            target_world_pos[0] = target_world_pos[0] * 0.97
            target_world_pos[1] = target_world_pos[1] * 0.97

            above_world_pos[2] = above_world_pos[2] + 5 + 80
            above_world_pos[0] = above_world_pos[0] * 0.97
            above_world_pos[1] = above_world_pos[1] * 0.97
            if not reachable_high or not reachable_low:
                pick_stable = False
                double_check = False
                reachable_low, joint_angles_2 = IK_geometric([target_world_pos[0], 
                                                            target_world_pos[1], 
                                                            target_world_pos[2], 
                                                            0.0],
                                                            m_matrix=self.rxarm.M_matrix,
                                                            s_list=self.rxarm.S_list)

                reachable_high, joint_angles_1 = IK_geometric([above_world_pos[0], 
                                                            above_world_pos[1], 
                                                            above_world_pos[2], 
                                                            0.0],
                                                            m_matrix=self.rxarm.M_matrix,
                                                            s_list=self.rxarm.S_list)

        if not reachable_high or not reachable_low:
            if not self.next_state == "estop":
                self.next_state = "idle"
            print("[PICK] Target point is unreachable, remain idle!!!")
            return False, pick_stable

        print("pick high: ", above_world_pos, ' ', phi)
        print("pick low: ", target_world_pos)
        
        ############ Executing #############
        # print("[PICK] Executing waypoints...")
        joint_angles_start = [0, 0, 0, 0, 0]
        joint_angles_start[0] = joint_angles_1[0]

        move_time, ac_time = self.calMoveTime(joint_angles_start)
        self.rxarm.set_single_joint_position("waist", joint_angles_1[0], moving_time=move_time, accel_time=ac_time, blocking=True)
        
        joint_angles_1[-2] = joint_angles_1[-2] + pick_wrist_offset
        move_time, ac_time = self.calMoveTime(joint_angles_1)
        self.rxarm.set_joint_positions(joint_angles_1,
                                        moving_time=move_time,
                                        accel_time=ac_time,
                                        blocking=True)

        # 3. go to the target pose and close gripper
        joint_angles_2[-2] = joint_angles_2[-2] + pick_wrist_offset
        move_time, ac_time = self.calMoveTime(joint_angles_2)
        self.rxarm.set_joint_positions(joint_angles_2,
                                        moving_time=move_time,
                                        accel_time=ac_time,
                                        blocking=True)
        self.rxarm.close_gripper()
        self.rxarm.gripper_state = False

        if to_sky:
            return True, pick_stable

        if double_check:
            self.rxarm.open_gripper()
            self.rxarm.gripper_state = True
            self.rxarm.set_ee_cartesian_trajectory(z=0.04, moving_time=0.5, wp_moving_time=0.1)
            self.rxarm.set_single_joint_position("wrist_rotate", joint_angles_2[-1] + np.pi/2, moving_time=0.3, accel_time=0.14)
            joint_angles_2[-1] = joint_angles_2[-1] + np.pi/2
            self.rxarm.set_joint_positions(joint_angles_2,
                                            moving_time=move_time,
                                            accel_time=ac_time,
                                            blocking=True)
            self.rxarm.close_gripper()
            self.rxarm.gripper_state = False

        
        # 4. raise to the point above the target point
        move_time, ac_time = self.calMoveTime(joint_angles_1)
        self.rxarm.set_joint_positions(joint_angles_1,
                                        moving_time=move_time,
                                        accel_time=ac_time,
                                        blocking=True)

        # 5. raise to the theta1 = 0 and theta2 = 0
        joint_angles_end = copy(joint_angles_1)
        joint_angles_end[1] = -np.pi/4
        joint_angles_end[2] = 0
        joint_angles_end[3] = -np.pi/2
        joint_angles_end[4] = 0
        move_time, ac_time = self.calMoveTime(joint_angles_end)
        self.rxarm.set_joint_positions(joint_angles_end,
                                        moving_time=move_time,
                                        accel_time=ac_time,
                                        blocking=True)

        # linear distance between the gripper fingers [m]
        gripper_distance = self.rxarm.get_gripper_position()
        print("!!!!!!!!!!!!!!!!!!!!!!!!! Gripper Dist: {:.8f}".format(gripper_distance))
        # TODO return pick fail according to gripper distance
        # if gripper_distance <= 0.0300000:
        #     print("[PICK] Failed to grab the block!")
        #     return False, pick_stable
        # else:
        # print("[PICK] Pick finished!")
        if gripper_distance>0.04:
            self.pick_size = 0 # large
        else:
            self.pick_size = 1 # small
        
        return True, pick_stable


    def place(self):
        self.status_message = "State: Place - Click to place"
        self.current_state = "place"
        self.camera.new_click = False
        print("[CLICK PLACE]    Please click one point to pick...")
        while not self.camera.new_click:
            rospy.sleep(0.1)
        
        self.camera.new_click = False
        pt = self.camera.last_click
        z = self.camera.DepthFrameRaw[pt[1]][pt[0]]
        click_uvd = np.append(pt, z)
        target_world_pos, block_ori = self.get_block_xyz_from_click(click_uvd)

        self.auto_place(target_world_pos, block_ori)

        if not self.next_state == "estop":
            self.next_state = "idle"

    def auto_place(self, _target_world_pos, block_ori=None, phi=np.pi/2, place_near=False):
        target_world_pos = deepcopy(_target_world_pos)
        above_world_pos = deepcopy(_target_world_pos)
        if place_near:
            target_world_pos = [0, 200, 0]
            above_world_pos = [0, 200, 0]

        ############ Planning #############
        # print("[PLACE]  Planning waypoints...")
        place_height_offset = 33
        place_wrist_offset = np.pi/18.0/3.0
        target_world_pos[2] = target_world_pos[2] + place_height_offset
        above_world_pos[2] = above_world_pos[2] + place_height_offset + 80

        reachable_low, reachable_high = False, False

        # Try vertical reach with phi = pi/2
        reachable_low, joint_angles_2 = IK_geometric([target_world_pos[0], 
                                                    target_world_pos[1],
                                                    target_world_pos[2],
                                                    phi],
                                                    block_ori=block_ori,
                                                    m_matrix=self.rxarm.M_matrix,
                                                    s_list=self.rxarm.S_list)

        if reachable_low:
            # phi = np.pi/2
            while not reachable_high:
                reachable_high, joint_angles_1 = IK_geometric([above_world_pos[0], 
                                                            above_world_pos[1],
                                                            above_world_pos[2],
                                                            phi],
                                                            block_ori=block_ori,
                                                            m_matrix=self.rxarm.M_matrix,
                                                            s_list=self.rxarm.S_list)
                if reachable_high:
                    break
                if above_world_pos[2] - target_world_pos[2] > 40:
                    above_world_pos[2] = above_world_pos[2] - 10
                elif _target_world_pos[2] < 38*4+10:
                    if 0.98* math.sqrt(above_world_pos[0] * above_world_pos[0] + above_world_pos[1] * above_world_pos[1]) > 158.875:
                        above_world_pos[0] = above_world_pos[0] * 0.98
                        above_world_pos[1] = above_world_pos[1] * 0.98
                    phi = phi - np.pi/18.0
                else:
                    break

                if phi <= 0:
                    break

        # Try horizontal reach with phi = 0.0
        if not reachable_high or not reachable_low:
            target_world_pos = deepcopy(_target_world_pos)
            above_world_pos = deepcopy(_target_world_pos)
            if _target_world_pos[2] >= 38*4+10:
                target_world_pos[1] = target_world_pos[1] - 15
                above_world_pos[1] = above_world_pos[1] - 15
            target_world_pos[2] = target_world_pos[2] + 12
            above_world_pos[2] = above_world_pos[2] + 12 + 80
            reachable_low, joint_angles_2 = IK_geometric([target_world_pos[0], 
                                                        target_world_pos[1],
                                                        target_world_pos[2],
                                                        0.0],
                                                        m_matrix=self.rxarm.M_matrix,
                                                        s_list=self.rxarm.S_list)

            while not reachable_high:
                reachable_high, joint_angles_1 = IK_geometric([above_world_pos[0], 
                                                            above_world_pos[1],
                                                            above_world_pos[2],
                                                            0.0],
                                                            block_ori=block_ori,
                                                            m_matrix=self.rxarm.M_matrix,
                                                            s_list=self.rxarm.S_list)
                # if 0.95* math.sqrt(above_world_pos[0] * above_world_pos[0] + above_world_pos[1] * above_world_pos[1]) > 158.875:
                #         above_world_pos[0] = above_world_pos[0] * 0.95
                #         above_world_pos[1] = above_world_pos[1] * 0.95
                if reachable_high:
                    break
                if above_world_pos[2] - target_world_pos[2] > 40:
                    above_world_pos[2] = above_world_pos[2] - 10
                else:
                    break

        # Unreachable
        if not reachable_high or not reachable_low:
            if not self.next_state == "estop":
                self.next_state = "idle"
            print("[PLACE]  Target point is unreachable, remain idle!!!")
            return False

        ############ Executing #############
        # print("[PLACE]  Executing waypoints...")
        # !!! TODO
        joint_angles_start = self.rxarm.get_positions()
        if _target_world_pos[2] > 38*4+10:
            joint_angles_start = self.rxarm.safe_pose

        move_time = np.abs(joint_angles_1[0] - joint_angles_start[0]) / (np.pi/3)
        ac_time = move_time / 4
        print("move time: ", move_time)
        self.rxarm.set_single_joint_position("waist", joint_angles_1[0], moving_time=move_time, accel_time=ac_time, blocking=True)

        # 1. go to point above target pose
        joint_angles_1[-2] = joint_angles_1[-2] + place_wrist_offset
        move_time, ac_time = self.calMoveTime(joint_angles_1)
        self.rxarm.set_joint_positions(joint_angles_1,
                                        moving_time=move_time,
                                        accel_time=ac_time,
                                        blocking=True)

        # 2. go to target pose and open gripper
        joint_angles_2[-2] = joint_angles_2[-2] + place_wrist_offset
        displacement = np.array(joint_angles_2) - np.array(joint_angles_1)
        displacement_unit =  displacement

        current_effort = self.rxarm.get_efforts()
        print("initial: ", current_effort)
        temp_joint = np.array(joint_angles_1)
        for i in range(10):
            # current_effort = self.rxarm.get_efforts()
            # print("initial: ", current_effort)
            displacement_unit = displacement_unit / 2
            temp_joint = temp_joint + displacement_unit
            move_time, ac_time = self.calMoveTime(temp_joint)
            self.rxarm.set_joint_positions(temp_joint.tolist(),
                                            moving_time=move_time,
                                            accel_time=ac_time,
                                            blocking=True)
            rospy.sleep(0.1)
            if i > 0:
                effort = self.rxarm.get_efforts()
                print(effort)
                # effort_diff = (effort[1] - current_effort[1])
                # if effort[1] > -150:
                #     break
                effort_diff = (effort - current_effort)[1:3]
                print("effort norm:", np.linalg.norm(effort_diff))
                if np.linalg.norm(effort_diff) > 100:
                    break
                if place_near:
                    break

        self.rxarm.open_gripper()
        self.rxarm.gripper_state = False
        
        move_time, ac_time = self.calMoveTime(joint_angles_1)
        self.rxarm.set_joint_positions(joint_angles_1,
                                        moving_time=move_time,
                                        accel_time=ac_time,
                                        blocking=True)

        joint_angles_end = copy(joint_angles_1)
        joint_angles_end[1] = -np.pi/6
        joint_angles_end[2] = 0
        joint_angles_end[3] = -np.pi/2
        joint_angles_end[4] = 0
        if _target_world_pos[2] >= 38*4+10:
            joint_angles_end[1] = -np.pi/3

        move_time, ac_time = self.calMoveTime(joint_angles_end)
        self.rxarm.set_joint_positions(joint_angles_end,
                                        moving_time=move_time,
                                        accel_time=ac_time,
                                        blocking=True)

        # print("[PLACE]  Place finished!")
        return True

    def clean(self):
        self.status_message = "State: Clean - Click to clean"
        self.current_state = "clean"
        self.camera.new_click = False
        print("[CLICK CLEAN] Please click one cluster to clean...")
        while not self.camera.new_click:
            rospy.sleep(0.05)
        
        self.camera.new_click = False
        pt = self.camera.last_click
        z = self.camera.DepthFrameRaw[pt[1]][pt[0]]
        click_uvd = np.append(pt, z)
        target_world_pos, block_ori = self.get_block_xyz_from_click(click_uvd)

        self.rxarm.go_to_home_pose(moving_time=2,
                                    accel_time=0.5,
                                    blocking=True)
        
        self.auto_clean(target_world_pos)
        if not self.next_state == "estop":
            self.next_state = "idle"

    def auto_clean(self, _target_world_pos):
        target_world_pos = deepcopy(_target_world_pos)

        ############ Planning #############
        print("[CLEAN] Planning waypoints...")
        clean_height_offset = 15 + 19
        clean_wrist_offset = np.pi/18.0/2.0
        target_world_pos[2] = 15 + clean_height_offset

        reachable_low = False

        # Try vertical reach with phi = pi/2
        reachable_low, joint_angles_2 = IK_geometric([target_world_pos[0], 
                                                    target_world_pos[1],
                                                    target_world_pos[2],
                                                    np.pi/2],
                                                    block_ori=-1, # negative value to enforce theta5 = np.pi/2
                                                    m_matrix=self.rxarm.M_matrix,
                                                    s_list=self.rxarm.S_list)

        if not reachable_low:
            reachable_low, joint_angles_2 = IK_geometric([target_world_pos[0], 
                                                        target_world_pos[1], 
                                                        target_world_pos[2], 
                                                        0.0],
                                                        m_matrix=self.rxarm.M_matrix,
                                                        s_list=self.rxarm.S_list)

        # Sweep a 45 degree sector
        joint_angles_1 = copy(joint_angles_2)
        joint_angles_1[0] = joint_angles_2[0] - np.pi/4/2
        joint_angles_1[4] = 0

        joint_angles_3 = copy(joint_angles_2)
        joint_angles_3[0] = joint_angles_2[0] + np.pi/4/2
        joint_angles_3[4] = 0

        if not reachable_low:
            if not self.next_state == "estop":
                self.next_state = "idle"
            print("[CLEAN] Target point is unreachable, remain idle!!!")
            return False

        ############ Executing #############
        print("[CLEAN] Executing waypoints...")
        # 1. rotate waise to the start pose
        joint_angles_start = [0, 0, 0, 0, 0]
        joint_angles_start[0] = joint_angles_1[0]

        move_time, ac_time = self.calMoveTime(joint_angles_start)
        self.rxarm.set_single_joint_position("waist", joint_angles_1[0], moving_time=move_time, accel_time=ac_time, blocking=True)

        # 2. go to the start clean pose
        move_time, ac_time = self.calMoveTime(joint_angles_1)
        self.rxarm.set_joint_positions(joint_angles_1,
                                        moving_time=move_time,
                                        accel_time=ac_time,
                                        blocking=True)

        # 3. go to the block cluster pose
        joint_angles_2[-2] = joint_angles_2[-2] + clean_wrist_offset
        # move_time, ac_time = self.calMoveTime(joint_angles_2)
        self.rxarm.set_joint_positions(joint_angles_2,
                                        moving_time=0.8,
                                        accel_time=0.39,
                                        blocking=True)
        
        
        # 4. go to the end clean pose
        move_time, ac_time = self.calMoveTime(joint_angles_3)
        self.rxarm.set_joint_positions(joint_angles_3,
                                        moving_time=0.8,
                                        accel_time=0.39,
                                        blocking=True)

        # 5. raise to the theta2 = -pi/4 and theta4 = pi/4
        joint_angles_end = copy(joint_angles_3)
        joint_angles_end[1] = -np.pi/4
        joint_angles_end[2] = 0
        joint_angles_end[3] = -np.pi/2
        joint_angles_end[4] = 0
        move_time, ac_time = self.calMoveTime(joint_angles_end)
        self.rxarm.set_joint_positions(joint_angles_end,
                                        moving_time=move_time,
                                        accel_time=ac_time,
                                        blocking=True)


        print("[CLEAN] Clean finished!")
        return True

    def auto_lineup(self, blocks, line_start_xyz, indices=None):
        if indices is None:
            indices = range(blocks.detected_num)
        
        print("[LINE UP]    Start auto lining up...")
        for idx in indices:
            print("[LINE UP]    Picking {} block...".format(self.camera.color_id[blocks.colors[idx]]))
            pick_ret, stable = self.auto_pick(blocks.xyzs[idx], blocks.thetas[idx])
            if pick_ret:
                print("[LINE UP]    Placing {} block...".format(self.camera.color_id[blocks.colors[idx]]))
                place_ret = self.auto_place(line_start_xyz, block_ori=0)
                if place_ret:
                    x_step = -55 if blocks.sizes[idx] == 0 else -42 # increase line up space by block's size
                    line_start_xyz[0] = line_start_xyz[0] + x_step
        
        print("[LINE UP]    Lining up finished")
        return line_start_xyz

    def auto_stack(self, blocks, stack_xyz, indices=None):
        if indices is None:
            indices = range(blocks.detected_num)
        
        print("[STACK]  Start auto stacking...")
        for idx in indices:
            print("[STACK]  Picking {} block...".format(self.camera.color_id[blocks.colors[idx]]))
            pick_ret, stable = self.auto_pick(blocks.xyzs[idx], blocks.thetas[idx])
            if pick_ret:
                print("[STACK]  Placing {} block...".format(self.camera.color_id[blocks.colors[idx]]))
                if not stable:
                    # If picked by horizontal reach, than place it horizontally
                    place_ret = self.auto_place(stack_xyz, phi=0.0)
                else:
                    place_ret = self.auto_place(stack_xyz)
                if place_ret:
                    height_step = 38 if blocks.sizes[idx] == 0 else 25 # increase stack height by block's size
                    stack_xyz[2] = stack_xyz[2] + height_step

        print("[STACK]  Stacking finished")
        return stack_xyz

    def task_test(self):
        self.current_state = "task_test"
        self.status_message = "This is the test"
        self.rxarm.go_to_sleep_pose(moving_time=2,
                                    accel_time=0.5,
                                    blocking=True)
        if not self.calibrate():
            self.next_state = "idle"
            return
        self.detect(ignore=3)
        blocks = self.camera.block_detections

        stack_xyz = [-300, -100, -10]
        destack_xyz = [300, -100, -10]
        self.rxarm.go_to_home_pose(moving_time=2,
                                    accel_time=0.5,
                                    blocking=True)
        # self.auto_pick(blocks.xyzs[0], blocks.thetas[0], double_check=True)
        # joint_names = self.rxarm.joint_names
        # pids = []
        # for name in joint_names:
        #     pid = self.rxarm.get_joint_position_pid_params(name)
        #     pids.append(pid)
        #     print(name, pid)
        #     # PID gains are in arbitrary units between 0 and 16,383.  Default [kp,ki,kd] is [100,0,0]
        #     # self.rxarm.set_joint_position_pid_params(name, [100, 0, 0])
        # pids = np.asarray(pids)
        # pids[:, 0] = 1000
        # print(pids)
        # for idx, name in enumerate(joint_names):
        #     self.rxarm.set_joint_position_pid_params(name, pids[idx, :])
        
        ############ Simple Test ##############
        # Choose to test auto stack or auto lineup
        # self.auto_stack(blocks, stack_xyz)
        # self.auto_lineup(blocks, destack_xyz)
        
        while blocks.detected_num > 0:
            print(self.detect(ignore=6, check_last_xyz=[-400, -100, -15]))
        # self.rxarm.go_to_home_pose(moving_time=2,
        #                             accel_time=0.5,
        #                             blocking=True)
        # self.rxarm.go_to_sleep_pose(moving_time=2,
        #                             accel_time=0.5,
        #                             blocking=True)
        print("Test Finished!!!")
        if not self.next_state == "estop":
            self.next_state = "idle"

    def task1(self):
        self.current_state = "task1"
        self.status_message = "Start task1!"

        self.rxarm.go_to_safe_pose(moving_time=2,
                                    accel_time=0.5,
                                    blocking=True)
        if not self.calibrate():
            self.next_state = "idle"
            return
        self.detect(ignore=5, sort_key="distance")
        blocks = self.camera.block_detections

        # self.rxarm.go_to_home_pose(moving_time=2,
        #                             accel_time=0.5,
        #                             blocking=True)
        large_xyz = [400, -100, -10]
        small_xyz = [-400, -100, -15]
        while blocks.detected_num > 0:
            block_xyz = blocks.xyzs[0]
            sz = blocks.sizes[0]
            ori = blocks.thetas[0]
            pick_sucess, pick_stable = self.auto_pick(block_xyz, ori)
            if not pick_sucess:
                self.rxarm.go_to_safe_pose(moving_time=2,
                                    accel_time=0.5,
                                    blocking=True)
            else:
                if self.pick_size == 0:
                    print("Place large block to ", large_xyz)
                    # if pick_stable:
                    self.auto_place(large_xyz)
                    large_xyz[0] = large_xyz[0] - 50
                    if large_xyz[0] < 150:
                        large_xyz[0] = 350
                        large_xyz[1] = -50
                    # else:
                    #     self.auto_place(large_xyz, place_near=True)
                    #     self.rxarm.go_to_safe_pose(moving_time=1, accel_time=0.5, blocking=True)
                elif self.pick_size == 1:
                    print("Place small block to ", small_xyz)
                    # if pick_stable:
                    self.auto_place(small_xyz)
                    small_xyz[0] = small_xyz[0] + 50
                    if small_xyz[0] > -150:
                        small_xyz[0] = -350
                        small_xyz[1] = -50
                    # else:
                    #     self.auto_place(small_xyz, place_near=True)
                    #     self.rxarm.go_to_safe_pose(moving_time=1, accel_time=0.5, blocking=True)

            self.detect(ignore=5, sort_key="distance")
        
        self.rxarm.go_to_safe_pose(moving_time=2,
                            accel_time=0.5,
                            blocking=True)

        if not self.next_state == "estop":
            self.next_state = "idle"


    def task2(self):
        self.current_state = "task2"
        self.status_message = "Start task2!"

        self.rxarm.go_to_safe_pose(moving_time=2,
                                    accel_time=0.5,
                                    blocking=True)
        if not self.calibrate():
            self.next_state = "idle"
            return
        self.detect(ignore=5, sort_key="distance")
        blocks = self.camera.block_detections

        # self.rxarm.go_to_home_pose(moving_time=2,
        #                             accel_time=0.5,
        #                             blocking=True)
        stack_xyz = [-350, -100, -10]
        small_xyz = [350, -100, -15]
        stack_num = 0
        tower_count = 0
        while blocks.detected_num > 0:
            block_xyz = blocks.xyzs[0]
            sz = blocks.sizes[0]
            ori = blocks.thetas[0]
            pick_sucess, pick_stable = self.auto_pick(block_xyz, ori)#, double_check=True)
            if not pick_sucess:
                self.rxarm.go_to_safe_pose(moving_time=2,
                                    accel_time=0.5,
                                    blocking=True)

            else:
                if self.pick_size == 0:
                    print("Place large block to ", stack_xyz)
                    if pick_stable:
                        self.auto_place(stack_xyz)
                        # # New
                        # self.rxarm.go_to_safe_pose(moving_time=2, accel_time=0.5, blocking=True)
                        # if self.detect(ignore=6, check_last_xyz=stack_xyz):
                        stack_xyz[2] = stack_xyz[2] + 38
                        stack_num = stack_num + 1
                        if stack_num >= 3:
                            tower_count = tower_count + 1
                            stack_xyz[0] = stack_xyz[0] + 100
                            stack_xyz[2] = -10
                            stack_num = 0
                            if tower_count == 2:
                                stack_xyz[0] = - 230
                                stack_xyz[1] = - 30
                                stack_xyz[2] = - 10
                    else:
                        self.auto_place(stack_xyz, place_near=True)
                        self.rxarm.go_to_safe_pose(moving_time=1, accel_time=0.5, blocking=True)
                elif self.pick_size == 1:
                    print("Place small block to ", small_xyz)
                    if pick_stable:
                        self.auto_place(small_xyz)
                        small_xyz[0] = small_xyz[0] - 75
                        if small_xyz[0] < 125:
                            small_xyz[0] = 350
                            small_xyz[1] = -35
                    else:
                        self.auto_place(small_xyz, place_near=True)
                        self.rxarm.go_to_safe_pose(moving_time=1, accel_time=0.5, blocking=True)

            self.detect(ignore=5, sort_key="distance")

        self.rxarm.go_to_safe_pose(moving_time=2,
                            accel_time=0.5,
                            blocking=True)


        self.detect(ignore=3, sort_key="distance")
        self.rxarm.go_to_safe_pose(moving_time=2,
                            accel_time=0.5,
                            blocking=True)
        # self.rxarm.go_to_home_pose(moving_time=2,
        #                             accel_time=0.5,
        #                             blocking=True)

        stack_xyz[2] = stack_xyz[2] - 5
        while blocks.detected_num > 0:
            block_xyz = blocks.xyzs[0]
            ori = blocks.thetas[0]
            pick_sucess, pick_stable = self.auto_pick(block_xyz, ori, double_check=True)
            if not pick_sucess:
                self.rxarm.go_to_safe_pose(moving_time=2,
                                    accel_time=0.5,
                                    blocking=True)
                # self.rxarm.go_to_home_pose(moving_time=2,
                #                     accel_time=0.5,
                #                     blocking=True)
                # self.rxarm.go_to_sleep_pose(moving_time=2,
                #                             accel_time=0.5,
                #                             blocking=True)
            else:
                print("Place small block to ", stack_xyz)
                # if pick_stable:
                self.auto_place(stack_xyz)
                # # New
                # self.rxarm.go_to_safe_pose(moving_time=2, accel_time=0.5, blocking=True)
                # if self.detect(ignore=6, check_last_xyz=stack_xyz):
                stack_xyz[2] = stack_xyz[2] + 25
                stack_num = stack_num + 1
                if stack_num >= 3:
                    tower_count = tower_count + 1
                    stack_xyz[0] = stack_xyz[0] + 100
                    stack_xyz[2] = -10
                    stack_num = 0
                    if tower_count == 2:
                        stack_xyz[0] = - 250
                        stack_xyz[1] = - 35
                        stack_xyz[2] = - 10
                # else:
                #     self.auto_place(stack_xyz, place_near=True)
                #     self.rxarm.go_to_safe_pose(moving_time=1, accel_time=0.5, blocking=True)

            self.detect(ignore=3, sort_key="distance")
        
        self.rxarm.go_to_safe_pose(moving_time=2, accel_time=0.5, blocking=True)
        # self.rxarm.go_to_home_pose(moving_time=2,
        #                             accel_time=0.5,
        #                             blocking=True)
        # self.rxarm.go_to_sleep_pose(moving_time=2,
        #                             accel_time=0.5,
        #                             blocking=True)

        if not self.next_state == "estop":
            self.next_state = "idle"

    def task3(self):
        self.current_state = "task3"
        self.status_message = "Start task3!"

        self.rxarm.go_to_safe_pose(moving_time=2,
                                    accel_time=0.5,
                                    blocking=True)
        if not self.calibrate():
            self.next_state = "idle"
            return
        self.detect(ignore=5, sort_key="distance")
        blocks = self.camera.block_detections

        large_xyz = [350, -100, -10]
        small_xyz = [-350, -100, -15]
        while blocks.detected_num > 0:
            block_xyz = blocks.xyzs[0]
            sz = blocks.sizes[0]
            ori = blocks.thetas[0]
            pick_sucess, pick_stable = self.auto_pick(block_xyz, ori)
            if not pick_sucess:
                self.rxarm.go_to_safe_pose(moving_time=2,
                                            accel_time=0.5,
                                            blocking=True)
                # self.rxarm.go_to_home_pose(moving_time=2,
                #                     accel_time=0.5,
                #                     blocking=True)
                # self.rxarm.go_to_sleep_pose(moving_time=2,
                #                             accel_time=0.5,
                #                             blocking=True)
            else:
                if self.pick_size == 0:
                    print("Place large block to ", large_xyz)
                    if pick_stable:
                        self.auto_place(large_xyz, block_ori=0)
                        large_xyz[0] = large_xyz[0] - 100
                        if large_xyz[0] < 150:
                            large_xyz[0] = 350
                            large_xyz[1] = -35
                    else:
                        self.auto_place(large_xyz, place_near=True)
                        self.rxarm.go_to_safe_pose(moving_time=1, accel_time=0.5, blocking=True)
                elif self.pick_size == 1:
                    print("Place small block to ", small_xyz)
                    if pick_stable:
                        self.auto_place(small_xyz, block_ori=0)
                        small_xyz[0] = small_xyz[0] + 100
                        if small_xyz[0] > -150:
                            small_xyz[0] = -350
                            small_xyz[1] = -40
                    else:
                        self.auto_place(large_xyz, place_near=True)
                        self.rxarm.go_to_safe_pose(moving_time=1, accel_time=0.5, blocking=True)

            self.detect(ignore=5, sort_key="distance")

        self.rxarm.go_to_safe_pose(moving_time=2,
                                    accel_time=0.5,
                                    blocking=True)
        # self.rxarm.go_to_home_pose(moving_time=2,
        #                             accel_time=0.5,
        #                             blocking=True)
        # self.rxarm.go_to_sleep_pose(moving_time=2,
        #                             accel_time=0.5,
        #                             blocking=True)
        self.detect(ignore=6, sort_key="color")
        # self.rxarm.go_to_safe_pose(moving_time=2,
        #                             accel_time=0.5,
        #                             blocking=True)
        # self.rxarm.go_to_home_pose(moving_time=2,
        #                             accel_time=0.5,
        #                             blocking=True)
        
        large_xyz = [225, 225, -10]
        small_xyz = [-250, 225, -15]
        while blocks.detected_num > 0:
            block_xyz = blocks.xyzs[0]
            sz = blocks.sizes[0]
            ori = blocks.thetas[0]
            pick_sucess, pick_stable = self.auto_pick(block_xyz, ori, double_check=False)
            if not pick_sucess:
                self.rxarm.go_to_safe_pose(moving_time=2,
                                            accel_time=0.5,
                                            blocking=True)
                # self.rxarm.go_to_home_pose(moving_time=2,
                #                     accel_time=0.5,
                #                     blocking=True)
                # self.rxarm.go_to_sleep_pose(moving_time=2,
                #                             accel_time=0.5,
                #                             blocking=True)
            else:
                if self.pick_size == 0:
                    print("Place large block to ", large_xyz)
                    self.auto_place(large_xyz, block_ori=0)
                    large_xyz[0] = large_xyz[0] - 50
                elif self.pick_size == 1:
                    print("Place small block to ", small_xyz)
                    self.auto_place(small_xyz, block_ori=0)
                    small_xyz[0] = small_xyz[0] + 35

            self.rxarm.go_to_safe_pose(moving_time=1,
                                    accel_time=0.25,
                                    blocking=True)
            self.detect(ignore=6, sort_key="color")
        self.rxarm.go_to_safe_pose(moving_time=2,
                                    accel_time=0.5,
                                    blocking=True)


        if not self.next_state == "estop":
            self.next_state = "idle"

    def task4(self):
        self.current_state = "task4"
        self.status_message = "Start task4!"

        self.rxarm.go_to_safe_pose(moving_time=2,
                                    accel_time=0.5,
                                    blocking=True)
        if not self.calibrate():
            self.next_state = "idle"
            return
        self.detect(ignore=5, sort_key="distance")
        blocks = self.camera.block_detections

        # self.rxarm.go_to_home_pose(moving_time=2,
        #                             accel_time=0.5,
        #                             blocking=True)
        large_xyz = [350, -100, -10]
        small_xyz = [-350, -100, -15]
        while blocks.detected_num > 0:
            block_xyz = blocks.xyzs[0]
            sz = blocks.sizes[0]
            ori = blocks.thetas[0]
            pick_sucess, pick_stable = self.auto_pick(block_xyz, ori)
            if not pick_sucess:
                self.rxarm.go_to_safe_pose(moving_time=2, accel_time=0.5, blocking=True)
                # self.rxarm.go_to_home_pose(moving_time=2,
                #                     accel_time=0.5,
                #                     blocking=True)
                # self.rxarm.go_to_sleep_pose(moving_time=2,
                #                             accel_time=0.5,
                #                             blocking=True)
            else:
                if self.pick_size == 0:
                    print("Place large block to ", large_xyz)
                    if pick_stable:
                        self.auto_place(large_xyz, block_ori=0)
                        self.rxarm.go_to_safe_pose(moving_time=2, accel_time=0.5, blocking=True)
                        if self.detect(ignore=6, check_last_xyz=large_xyz):
                            large_xyz[0] = large_xyz[0] - 100
                            if large_xyz[0] < 150:
                                large_xyz[0] = 350
                                large_xyz[1] = -35
                    else:
                        self.auto_place(large_xyz, place_near=True)
                        self.rxarm.go_to_safe_pose(moving_time=1, accel_time=0.5, blocking=True)
                elif self.pick_size == 1:
                    print("Place small block to ", small_xyz)
                    if pick_stable:
                        self.auto_place(small_xyz, block_ori=0)
                        self.rxarm.go_to_safe_pose(moving_time=2, accel_time=0.5, blocking=True)
                        if self.detect(ignore=6, check_last_xyz=small_xyz):
                            small_xyz[0] = small_xyz[0] + 100
                            if small_xyz[0] > -150:
                                small_xyz[0] = -350
                                small_xyz[1] = -40
                    else:
                        self.auto_place(large_xyz, place_near=True)
                        self.rxarm.go_to_safe_pose(moving_time=1, accel_time=0.5, blocking=True)

            self.detect(ignore=5, sort_key="distance")

        self.rxarm.go_to_safe_pose(moving_time=2, accel_time=0.5, blocking=True)
        # self.rxarm.go_to_home_pose(moving_time=2,
        #                             accel_time=0.5,
        #                             blocking=True)
        # self.rxarm.go_to_sleep_pose(moving_time=2,
        #                             accel_time=0.5,
        #                             blocking=True)
        self.detect(ignore=6, sort_key="color")
        # self.rxarm.go_to_safe_pose(moving_time=1, accel_time=0.5, blocking=True)
        # self.rxarm.go_to_home_pose(moving_time=2,
        #                             accel_time=0.5,
        #                             blocking=True)
        
        large_xyz = [150, 175, -10]
        small_xyz = [-150, 175, -15]
        while blocks.detected_num > 0:
            block_xyz = blocks.xyzs[0]
            sz = blocks.sizes[0]
            ori = blocks.thetas[0]
            pick_sucess, pick_stable = self.auto_pick(block_xyz, ori, double_check=True)
            if not pick_sucess:
                self.rxarm.go_to_safe_pose(moving_time=2, accel_time=0.5, blocking=True)
                # self.rxarm.go_to_home_pose(moving_time=2,
                #                     accel_time=0.5,
                #                     blocking=True)
                # self.rxarm.go_to_sleep_pose(moving_time=2,
                #                             accel_time=0.5,
                #                             blocking=True)
            else:
                if sz == 0:
                    print("Place large block to ", large_xyz)
                    self.auto_place(large_xyz, block_ori=0)
                    large_xyz[2] = large_xyz[2] + 38
                else:
                    print("Place small block to ", small_xyz)
                    self.auto_place(small_xyz, block_ori=0)
                    small_xyz[2] = small_xyz[2] + 25

            self.detect(ignore=6, sort_key="color")


        self.rxarm.go_to_safe_pose(moving_time=1, accel_time=0.5, blocking=True)
        # self.rxarm.go_to_home_pose(moving_time=2,
        #                             accel_time=0.5,
        #                             blocking=True)
        # self.rxarm.go_to_sleep_pose(moving_time=2,
        #                             accel_time=0.5,
        #                             blocking=True)

        if not self.next_state == "estop":
            self.next_state = "idle"

    def task5(self):
        self.current_state = "task5"
        self.status_message = "Start task5!"

        self.rxarm.go_to_safe_pose(moving_time=2,
                                    accel_time=0.5,
                                    blocking=True)
        if not self.calibrate():
            self.next_state = "idle"
            return
        # self.detect(ignore=5, sort_key="distance")
        # blocks = self.camera.block_detections

        # self.rxarm.go_to_home_pose(moving_time=2,
        #                             accel_time=0.5,
        #                             blocking=True)
        # large_xyz = [350, -110, -10]
        # cnt = 0
        # while blocks.detected_num > 0:
        #     block_xyz = blocks.xyzs[0]
        #     sz = blocks.sizes[0]
        #     ori = blocks.thetas[0]
        #     pick_sucess, pick_stable = self.auto_pick(block_xyz, ori)
        #     if not pick_sucess:
        #         self.rxarm.go_to_safe_pose(moving_time=2, accel_time=0.5, blocking=True)

        #     else:
        #         print("Place large block to ", large_xyz)
        #         if pick_stable:
        #             self.auto_place(large_xyz, block_ori=0)
        #             cnt = cnt + 1
        #             if cnt < 3:
        #                 large_xyz[0] = large_xyz[0] - 100
        #             if cnt == 3:
        #                 large_xyz = [350, -35, -10]
        #             if cnt > 3 and cnt < 6:
        #                 large_xyz[0] = large_xyz[0] - 100
        #             if cnt == 6:
        #                 large_xyz = [-350, -100, -10]
        #             if cnt > 6 and cnt < 9:
        #                 large_xyz[0] = large_xyz[0] + 100
        #             if cnt == 9:
        #                 large_xyz = [-350, -35, -10]
        #             if cnt > 9 and cnt < 12:
        #                 large_xyz[0] = large_xyz[0] + 100
        #         else:
        #             self.auto_place(large_xyz, place_near=True)
        #             self.rxarm.go_to_safe_pose(moving_time=1, accel_time=0.5, blocking=True)

        #     self.detect(ignore=5, sort_key="distance")

        # self.rxarm.go_to_safe_pose(moving_time=2, accel_time=0.5, blocking=True)

        self.detect(ignore=6, sort_key="distance", frac=4/5)
        # self.rxarm.go_to_safe_pose(moving_time=1, accel_time=0.5, blocking=True)
        blocks = self.camera.block_detections
        
        test = False
        large_xyz = [0, 275, -10]
        counter = 0
        while blocks.detected_num > 0 and not test:
            block_xyz = blocks.xyzs[0]
            sz = blocks.sizes[0]
            ori = blocks.thetas[0]
            pick_sucess, pick_stable = self.auto_pick(block_xyz, ori, double_check=True)
            if not pick_sucess:
                self.rxarm.go_to_safe_pose(moving_time=2, accel_time=0.5, blocking=True)
            else:
                print("Place large block to ", large_xyz)
                self.auto_place(large_xyz, block_ori=0)
                large_xyz[2] = large_xyz[2] + 38
                counter = counter + 1
                if counter > 8:
                    break

            self.detect(ignore=6, sort_key="distance", frac=4/5)
        
        self.detect(ignore=6, sort_key="distance", frac=4/5)

        large_xyz = [-350, 0, -10]
        while blocks.detected_num > 0:
            block_xyz = blocks.xyzs[0]
            sz = blocks.sizes[0]
            ori = blocks.thetas[0]
            pick_sucess, pick_stable = self.auto_pick(block_xyz, ori, double_check=True)
            if not pick_sucess:
                self.rxarm.go_to_safe_pose(moving_time=2, accel_time=0.5, blocking=True)
            else:
                print("Place large block to ", large_xyz)
                self.auto_place(large_xyz, block_ori=0)
                large_xyz[2] = large_xyz[2] + 38
                counter = counter + 1
                if counter > 8 + 3:
                    break

            self.detect(ignore=6, sort_key="distance", frac=4/5)

        self.detect(ignore=6, sort_key="distance", frac=4/5)

        large_xyz = [-350, 0, 0]
        pick_sucess, pick_stable = self.auto_pick(large_xyz, block_ori=None, phi=0.0, double_check=True, to_sky=True)
        reachable, joint_angles_sky = IK_geometric([-275, 
                                            large_xyz[1],
                                            large_xyz[2]+342+20,
                                            0.0],
                                            block_ori=None,
                                            m_matrix=self.rxarm.M_matrix,
                                            s_list=self.rxarm.S_list)
        if reachable:
            self.rxarm.set_joint_positions(joint_angles_sky,
                                            moving_time=7,
                                            accel_time=7/4,
                                            blocking=True)

        # self.rxarm.set_ee_cartesian_trajectory(z=342, moving_time=6)
        self.rxarm.set_single_joint_position("waist", 0, moving_time=6, accel_time=2, blocking=True)
        target_pos_xyz = [0, 275, -10+342]
        reachable_end, joint_angles_end = IK_geometric([target_pos_xyz[0], 
                                            target_pos_xyz[1],
                                            target_pos_xyz[2],
                                            0.0],
                                            block_ori=None,
                                            m_matrix=self.rxarm.M_matrix,
                                            s_list=self.rxarm.S_list)
        if reachable_end:
            self.rxarm.set_joint_positions(joint_angles_end,
                                            moving_time=5,
                                            accel_time=5/4,
                                            blocking=True)
        self.rxarm.open_gripper()
        self.rxarm.gripper_state = True


        if not self.next_state == "estop":
            self.next_state = "idle"

    def calibrate(self):
        """!
        @brief      Gets the user input to perform the calibration
        """
        self.current_state = "calibrate"

        """Perform camera calibration routine here"""
        self.status_message = "Calibration - Completed Calibration"

        # Align depth image with color image using homography
        points_dst = np.array([[226, 89],
                                [1091, 90],
                                [226, 700],
                                [1091, 701]])

        points_src = np.array([[238, 98],
                                [1079, 99],
                                [235, 690],
                                [1080, 690]])

        self.camera.homography, _ = cv2.findHomography(points_src, points_dst)
        self.camera.homography = self.camera.homography / self.camera.homography[2,2]
        # print(self.homography)
        
        tagPoints = np.zeros((4, 3), dtype=DTYPE)
        if len(self.camera.tag_detections.detections)<4:
            self.next_state = "idel"
            print("[CALIBRATE]  Calibration failed! Less than 4 tags were detected.")
            self.camera.cameraCalibrated = False
            return False
        for detection in self.camera.tag_detections.detections:
            tagPoints[detection.id[0] - 1, 0] = detection.pose.pose.pose.position.x * 1000
            tagPoints[detection.id[0] - 1, 1] = detection.pose.pose.pose.position.y * 1000
            tagPoints[detection.id[0] - 1, 2] = detection.pose.pose.pose.position.z * 1000

        # !!! Change this intrinsic_matrix to the default one in /cmaera/camera_info/K
        apriltag_intrinsic = np.array([908.3550415039062, 0.0, 642.5927124023438, 0.0, 908.4041137695312, 353.12652587890625, 0.0, 0.0, 1.0], dtype=DTYPE).reshape((3,3))
        imagePoints = np.matmul(apriltag_intrinsic, tagPoints.T).T
        
        imagePoints = imagePoints[:, 0:2] / imagePoints[:, 2].reshape((4, 1))
        objectPoints = np.array([[-250, -25, 0], [250, -25, 0], [250, 275, 0], [-250, 275, 0]], dtype=DTYPE)

        # !!! try solvePnPRansac() use more than 4 points
        retval, rvec, tvec = cv2.solvePnP(objectPoints, imagePoints, self.camera.intrinsic_matrix, self.camera.distortion_coefficients)
        rmat, jacobian = cv2.Rodrigues(rvec) # 3x3
        tvec = tvec + np.array([0, 0, -25]).reshape((3,1)) # 3x1 z offset
        extrinsic_temp = np.column_stack((rmat, tvec)) # 3x4
        extrinsic_pad = np.array([0, 0, 0, 1], dtype=DTYPE) # 4,
        self.camera.extrinsic_matrix = np.row_stack((extrinsic_temp, extrinsic_pad)) # 4x4
        self.camera.extrinsic_matrix_inv = np.linalg.pinv(self.camera.extrinsic_matrix)
        self.camera.cameraCalibrated = True
        # print(self.camera.extrinsic_matrix)
        print("[CALIBRATE]  Calibration successed!")

        self.next_state = "idle"
        return True
        
    def get_block_xyz_from_click(self, click_uvd):
        click_xyz = self.camera.coord_pixel_to_world(click_uvd[0], click_uvd[1], click_uvd[2]).flatten().tolist()
        if self.camera.block_detections.detected_num == 0:
            return click_xyz, 0.0
        blocks_uv = self.camera.block_detections.uvds[:, :2]
        # print(blocks_uv.shape)
        # print(click_uvd[:2].shape)
        dist = blocks_uv - click_uvd[:2]
        dist_norm = np.linalg.norm(dist, axis=1)
        dist_min = np.min(dist_norm)
        # print("pixel dist:", dist_min)
        # !! TODO check the threshold here
        if dist_min < 50:
            return self.camera.block_detections.xyzs[np.argmin(dist_norm)], self.camera.block_detections.thetas[np.argmin(dist_norm)]
        else:
            return click_xyz, 0.0
    
    def line_dist(self, line_xy, target_xy):
        dist = (-line_xy[1]*target_xy[0] + line_xy[0]*target_xy[1])/np.linalg.norm(line_xy)
        return abs(dist)
    
    def check_path_clean(self, target_xyz):
        blocks = self.camera.block_detections
        for i in range(blocks.detected_num):
            dist = self.line_dist(blocks.xyzs[i, :2], target_xyz[:2])
            # print("path point dist {0}".format(dist))
            if dist == 0:
                continue
            if dist < 50:
                return False

        return True

    def detect(self, ignore=None, sort_key="color", check_last_xyz=None, frac=3/4):
        """!
        @brief      Detect the blocks
        """
        # 1280x720
        # ----------------- 6 := 1 + 2
        # |  2    |    1  |
        # ----------------- frac {3}{4}
        # |  3  |ARM|  4  |
        # ----------------- 5 := 3 + 4
        self.current_state = "detect"
        self.status_message = "Detecting blocks..."
        img_h, img_w = 720, 1280
        # frac = 3.0 / 4.0
        blind_rectangle = None
        if ignore==1:
            blind_rectangle = [(int(img_w/2), 0), (img_w, int(img_h*frac))]
        elif ignore==2:
            blind_rectangle = [(0, 0), (int(img_w/2), int(img_h/3/2))]
        elif ignore==3:
            blind_rectangle = [(0, int(img_h*frac)), (int(img_w/2), img_h)]
        elif ignore==4:
            blind_rectangle = [(int(img_w/2), int(img_h*frac)), (img_w, img_h)]
        elif ignore==5: # negative half plane
            blind_rectangle = [(0, int(img_h*frac)), (img_w, img_h)]
        elif ignore==6: # positive half plane
            blind_rectangle = [(0, 0), (img_w, int(img_h*frac))]
        
        # for i in range(3):
        self.camera.detectBlocksInDepthImage(blind_rect=blind_rectangle, sort_key=sort_key)

        blocks = self.camera.block_detections
        if blocks.has_cluster:
            self.rxarm.go_to_safe_pose(moving_time=2,
                                        accel_time=0.5,
                                        blocking=True)
        while blocks.has_cluster:
            self.auto_clean(blocks.xyzs[0])
            self.rxarm.go_to_safe_pose(moving_time=2,
                                        accel_time=0.5,
                                        blocking=True)

            self.camera.detectBlocksInDepthImage(blind_rect=blind_rectangle, sort_key=sort_key)
            blocks = self.camera.block_detections

            self.rxarm.go_to_safe_pose(moving_time=2,
                                        accel_time=0.5,
                                        blocking=True)
        
        self.next_state = "idel"

        if check_last_xyz is not None:
            check_last_xyz = np.array(check_last_xyz).reshape((1, 3))
            if blocks.detected_num == 0:
                return False
            blocks_xyz = blocks.xyzs
            dist = blocks_xyz[:, :2] - check_last_xyz[:, :2]
            dist_norm = np.linalg.norm(dist, axis=1)
            dist_min = np.min(dist_norm)
            dist_z = np.abs(blocks_xyz[np.argmin(dist_norm), 2] - check_last_xyz[0, 2])
            print("!!!!!!!!!!!!!!!!!!!! pixel dist:", dist_min)
            print("!!!!!!!!!!!!!!!!!!!! block z:", dist_z)
            # !! TODO check the threshold here
            if dist_min < 50:
                if blocks.sizes[np.argmin(dist_norm)] == 0:
                    if dist_z < 50:
                        return True
                    else:
                        return False
                else:
                    if dist_z < 30:
                        return True
                    else:
                        return False

            else:
                return False
        else:
            return True

    def initialize_rxarm(self):
        """!
        @brief      Initializes the rxarm.
        """
        self.current_state = "initialize_rxarm"
        self.status_message = "RXArm Initialized!"
        if not self.rxarm.initialize():
            print('Failed to initialize the rxarm')
            self.status_message = "State: Failed to initialize the rxarm!"
            rospy.sleep(5)
        self.next_state = "idle"

class StateMachineThread(QThread):
    """!
    @brief      Runs the state machine
    """
    updateStatusMessage = pyqtSignal(str)
    
    def __init__(self, state_machine, parent=None):
        """!
        @brief      Constructs a new instance.

        @param      state_machine  The state machine
        @param      parent         The parent
        """
        QThread.__init__(self, parent=parent)
        self.sm=state_machine

    def run(self):
        """!
        @brief      Update the state machine at a set rate
        """
        while True:
            self.sm.run()
            self.updateStatusMessage.emit(self.sm.status_message)
            rospy.sleep(0.02) # 50 Hz
