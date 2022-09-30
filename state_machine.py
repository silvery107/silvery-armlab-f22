"""!
The state machine that implements the logic.
"""
from copy import deepcopy
from PyQt4.QtCore import (QThread, Qt, pyqtSignal, pyqtSlot, QTimer)
import time
import numpy as np
from kinematics import IK_geometric
import rospy
import cv2
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

        if self.next_state == "pick":
            self.pick()

        if self.next_state == "place":
            self.place()

        if self.next_state == "task_test":
            self.task_test()


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
        angular_t = np.abs(displacement) / (np.pi / 5)
        move_time = np.max(angular_t)
        ac_time = move_time / 4.0
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

        # target_world_pos = self.camera.coord_pixel_to_world(block_uvd[0], block_uvd[1], block_uvd[2]).flatten().tolist()
        self.auto_pick(target_world_pos, block_ori)
        if not self.next_state == "estop":
            self.next_state = "idle"

    def auto_pick(self, _target_world_pos, block_ori):
        target_world_pos = deepcopy(_target_world_pos)
        ############ Planning #############
        print("[PICK] Planning waypoints...")
        pick_height_offset = 10
        pick_wrist_offset = np.pi/18.0/2.0
        joint_angles_home = [0, 0, 0, 0, 0]
        target_world_pos[2] = target_world_pos[2] + pick_height_offset
        above_world_pos = deepcopy(target_world_pos)
        above_world_pos[2] = target_world_pos[2] + pick_height_offset + 80

        reachable_low, reachable_high = False, False

        reachable_low, joint_angles_2 = IK_geometric([target_world_pos[0], 
                                    target_world_pos[1], 
                                    target_world_pos[2], 
                                    np.pi/2],
                                    block_ori=block_ori,
                                    m_matrix=self.rxarm.M_matrix,
                                    s_list=self.rxarm.S_list)

        if reachable_low:
            phi = np.pi/2
            while not reachable_high:
                reachable_high, joint_angles_1 = IK_geometric([above_world_pos[0], 
                                            above_world_pos[1], 
                                            above_world_pos[2], 
                                            phi],
                                            block_ori=block_ori,
                                            m_matrix=self.rxarm.M_matrix,
                                            s_list=self.rxarm.S_list)
                phi = phi - np.pi/18.0
                if phi < 0:
                    break

        if not reachable_high or not reachable_low:
            if not self.next_state == "estop":
                self.next_state = "idle"
            print("[PICK] Target point is unreachable, remain idle!!!")
            return False
        
        ############ Executing #############
        print("[PICK] Executing waypoints...")
        # 1. go to the home pose
        move_time, ac_time = self.calMoveTime(joint_angles_home)
        self.rxarm.go_to_home_pose(moving_time=move_time,
                                    accel_time=ac_time,
                                    blocking=True)
        if not self.rxarm.gripper_state:
            self.rxarm.open_gripper()
            self.rxarm.gripper_state = True

        # 2. go to the point above target pose with waist angle move first
        move_time = np.abs(joint_angles_1[0]) / (np.pi / 5)
        ac_time = move_time / 4.0
            
        self.rxarm.set_single_joint_position("waist", joint_angles_1[0], moving_time=move_time, accel_time=ac_time, blocking=True)

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
        
        # 4. raise to the point above the target point
        move_time, ac_time = self.calMoveTime(joint_angles_1)
        self.rxarm.set_joint_positions(joint_angles_1,
                                        moving_time=move_time,
                                        accel_time=ac_time,
                                        blocking=True)

        print("[PICK] PICK finished!")
        return True

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

        # print(target_world_pos)

        # target_world_pos = self.camera.coord_pixel_to_world(block_uvd[0], block_uvd[1], block_uvd[2]).flatten().tolist()
        self.auto_place(target_world_pos, block_ori)

        if not self.next_state == "estop":
            self.next_state = "idle"

    def auto_place(self, _target_world_pos, _block_ori=None):
        target_world_pos = deepcopy(_target_world_pos)
        # print(target_world_pos)
        if _block_ori is not None:
            block_ori = _block_ori
        else:
            block_ori = 0.0
        ############ Planning #############
        print("[PLACE]  Planning waypoints...")
        place_height_offset = 30
        place_wrist_offset = np.pi/18.0/2.0
        target_world_pos[2] = target_world_pos[2] + place_height_offset
        above_world_pos = deepcopy(target_world_pos)
        above_world_pos[2] = target_world_pos[2] + place_height_offset + 80

        reachable_low, reachable_high = False, False

        # Try vertical reach with phi = pi/2
        reachable_low, joint_angles_2 = IK_geometric([target_world_pos[0], 
                                    target_world_pos[1], 
                                    target_world_pos[2], 
                                    np.pi/2],
                                    block_ori=block_ori,
                                    m_matrix=self.rxarm.M_matrix,
                                    s_list=self.rxarm.S_list)

        if reachable_low:
            phi = np.pi/2
            while not reachable_high:
                reachable_high, joint_angles_1 = IK_geometric([above_world_pos[0], 
                                            above_world_pos[1], 
                                            above_world_pos[2], 
                                            phi],
                                            block_ori=block_ori,
                                            m_matrix=self.rxarm.M_matrix,
                                            s_list=self.rxarm.S_list)
                phi = phi - np.pi / 18.0
                if phi < 0:
                    break

        # Try horizontal reach with phi = 0.0
        if not reachable_high or not reachable_low:
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

        # Unreachable
        if not reachable_high or not reachable_low:
            if not self.next_state == "estop":
                self.next_state = "idle"
            print("[PLACE]  Target point is unreachable, remain idle!!!")
            return False

        ############ Executing #############
        print("[PLACE]  Executing waypoints...")
        # 1. go to point above target pose
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
        for i in range(6):
            displacement_unit = displacement_unit / 2
            temp_joint = temp_joint + displacement_unit
            move_time, ac_time = self.calMoveTime(temp_joint)
            self.rxarm.set_joint_positions(temp_joint.tolist(),
                                            moving_time=move_time,
                                            accel_time=ac_time,
                                            blocking=True)
            effort = self.rxarm.get_efforts()
            print(effort)
            # effort_diff = (effort[1] - current_effort[1])
            # if effort[1] > -150:
            #     break
            effort_diff = (effort - current_effort)[1:3]
            print("effort norm:", np.linalg.norm(effort_diff))
            if np.linalg.norm(effort_diff) > 300:
                break

        self.rxarm.open_gripper()
        self.rxarm.gripper_state = False
        
        move_time, ac_time = self.calMoveTime(joint_angles_1)
        self.rxarm.set_joint_positions(joint_angles_1,
                                        moving_time=move_time,
                                        accel_time=ac_time,
                                        blocking=True)


        print("[PLACE]  PLACE finished!")
        return True

    def auto_lineup(self, blocks, line_start_xyz, indices=None):
        if indices is None:
            indices = range(blocks.detected_num)
        # !!! line up incremental alone x-axis
        print("[LINE UP]    Start auto lining up...")
        for idx in indices:
            print("[LINE UP]    Picking {} block...".format(self.camera.color_id[blocks.colors[idx]]))
            pick_ret = self.auto_pick(blocks.xyzs[idx], blocks.thetas[idx])
            if pick_ret:
                print("[LINE UP]    Placing {} block...".format(self.camera.color_id[blocks.colors[idx]]))
                place_ret = self.auto_place(line_start_xyz)
                if place_ret:
                    x_step = -50 if blocks.sizes[idx] == 0 else -35 # increase line up space by block's size
                    line_start_xyz[0] = line_start_xyz[0] + x_step
                    print("[LINE UP]    One block lined successfully!")
        
        print("[LINE UP]    Lining up finished")
        return line_start_xyz

    def auto_stack(self, blocks, stack_xyz, indices=None):
        if indices is None:
            indices = range(blocks.detected_num)
        # !!! experiment with fix point open loop stack
        print("[STACK]  Start auto stacking...")
        for idx in indices:
            print("[STACK]  Picking {} block...".format(self.camera.color_id[blocks.colors[idx]]))
            pick_ret = self.auto_pick(blocks.xyzs[idx], blocks.thetas[idx])
            if pick_ret:
                print("[STACK]  Placing {} block...".format(self.camera.color_id[blocks.colors[idx]]))
                place_ret = self.auto_place(stack_xyz)
                if place_ret:
                    height_step = 38 if blocks.sizes[idx] == 0 else 20 # increase stack height by block's size
                    stack_xyz[2] = stack_xyz[2] + height_step
                    print("[STACK]  One block stacked successfully!")

        print("[STACK]  Stacking finished")
        return stack_xyz

    def task_test(self):
        self.current_state = "task_test"
        self.status_message = "This is the real autonomy!"
        self.rxarm.go_to_sleep_pose(moving_time=2,
                                    accel_time=0.5,
                                    blocking=True)
        if not self.calibrate():
            self.next_state = "idle"
            return
        self.detect(ignore=3)
        blocks = self.camera.block_detections
        
        target_color = 0
        stack_xyz = [-250, 25, -5]
        destack_xyz = [-50, 200, 0]

        ############ Real Test ##############
        while blocks.detected_num>0:
            stack_order = []
            destack_order = []
            for i in range(blocks.detected_num):
                # Find blocks in rainbow color order to stack
                if blocks.colors[i] == target_color:
                    target_color = target_color + 1
                    stack_order.append(i)
                else:
                    # Destack blocks more than 1 high
                    for idx in range(blocks.detected_num):
                        if blocks.xyzs[idx, 2] > 50 and stack_order.count(idx)<1:
                            destack_order.append(idx)
                    break

            # Execute auto stack according to stack_order
            stack_xyz = self.auto_stack(blocks, stack_xyz, stack_order)
            # Execute auto lineup according to destack_order
            destack_xyz = self.auto_lineup(blocks, destack_xyz, destack_order)

            self.rxarm.go_to_home_pose(moving_time=2,
                                        accel_time=0.5,
                                        blocking=True)
            self.rxarm.go_to_sleep_pose(moving_time=2,
                                        accel_time=0.5,
                                        blocking=True)
            self.detect(ignore=3)
            blocks = self.camera.block_detections

        ############ Simple Test ##############
        # Choose to test auto stack or auto lineup
        # self.auto_stack(blocks, stack_xyz)
        # self.auto_lineup(blocks, destack_xyz)
        self.rxarm.go_to_home_pose(moving_time=2,
                                    accel_time=0.5,
                                    blocking=True)
        self.rxarm.go_to_sleep_pose(moving_time=2,
                                    accel_time=0.5,
                                    blocking=True)

        if not self.next_state == "estop":
            self.next_state = "idle"


    def calibrate(self):
        """!
        @brief      Gets the user input to perform the calibration
        """
        self.current_state = "calibrate"

        """Perform camera calibration routine here"""
        self.status_message = "Calibration - Completed Calibration"
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
        print("[CALIBRATE]  Calibration successed!")
        # print(self.camera.extrinsic_matrix)
        
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
        print("pixel dist:", dist_min)
        # !! TODO check the threshold here
        if dist_min < 50:
            return self.camera.block_detections.xyzs[np.argmin(dist_norm)], self.camera.block_detections.thetas[np.argmin(dist_norm)]
        else:
            return click_xyz, 0.0

    def detect(self, ignore=None):
        """!
        @brief      Detect the blocks
        """
        # 1280x720
        # -----------------
        # |  2    |    1  |
        # ----------------- frac {2}{3}
        # |  3  |ARM|  4  |
        # ----------------- 5 := 3+4
        self.current_state = "detect"
        self.status_message = "Detecting blocks..."
        img_h, img_w = 720, 1280
        frac = 2/3
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

        self.camera.detectBlocksInDepthImage(blind_rect=blind_rectangle)
        rospy.sleep(0.1)
        self.next_state = "idel"

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
