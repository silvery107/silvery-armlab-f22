"""!
The state machine that implements the logic.
"""
from PyQt4.QtCore import (QThread, Qt, pyqtSignal, pyqtSlot, QTimer)
import time
import numpy as np
import rospy
import cv2

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
        for point, gripper_state in zip(self.record_waypoints, self.record_gripper):
            self.rxarm.set_positions(point)
            rospy.sleep(1.5)
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

    def calibrate(self):
        """!
        @brief      Gets the user input to perform the calibration
        """
        self.current_state = "calibrate"

        """TODO Perform camera calibration routine here"""
        self.status_message = "Calibration - Completed Calibration"
        # print("start camera calibration: please click four points")
        # imagePoints = []
        # click_n = 0
        # while click_n < 4:
        #     if self.camera.new_click:
        #         click_n = click_n + 1
        #         imagePoints.append([self.camera.last_click[0], self.camera.last_click[1]])
        #         self.camera.new_click = False
        #     else:
        #         rospy.sleep(0.5)
        # imagePoints = np.asarray(imagePoints, dtype=np.float32)
        # print(imagePoints)
        tagPoints = np.zeros((4, 3), dtype=np.float32)
        for detection in self.camera.tag_detections.detections:
            tagPoints[detection.id[0] - 1, 0] = detection.pose.pose.pose.position.x * 1000
            tagPoints[detection.id[0] - 1, 1] = detection.pose.pose.pose.position.y * 1000
            tagPoints[detection.id[0] - 1, 2] = detection.pose.pose.pose.position.z * 1000

        # !!! Change this intrinsic_matrix to the default one in /cmaera/camera_info/K
        imagePoints = np.matmul(self.camera.intrinsic_matrix, tagPoints.T).T 
        imagePoints = imagePoints[:, 0:2] / imagePoints[:, 2].reshape((4, 1))
        objectPoints = np.array([[-250, -25, 0], [250, -25, 0], [250, 275, 0], [-250, 275, 0]], dtype=np.float32)
        print(imagePoints)
        print(objectPoints)

        # !!! try solvePnPRansac()
        retval, rvec, tvec = cv2.solvePnP(objectPoints, imagePoints, self.camera.intrinsic_matrix, self.camera.distortion_coefficients)

        rmat, jacobian = cv2.Rodrigues(rvec) # 3x3
        extrinsic_temp = np.column_stack((rmat, tvec)) # 3x4
        extrinsic_pad = np.array([0, 0, 0, 1], dtype=np.float32) # 4,
        self.camera.extrinsic_matrix = np.row_stack((extrinsic_temp, extrinsic_pad)) # 4x4
        self.camera.extrinsic_matrix_inv = np.linalg.pinv(self.camera.extrinsic_matrix)
        self.camera.cameraCalibrated = True
        print(self.camera.extrinsic_matrix)
        
        self.next_state = "idle"
        

    """ TODO """
    def detect(self):
        """!
        @brief      Detect the blocks
        """
        rospy.sleep(1)

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
            rospy.sleep(0.02)