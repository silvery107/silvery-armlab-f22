import math
import rospy
import roslib
roslib.load_manifest('urdfdom_py')
import threading
import numpy as np
import json
import random
import modern_robotics as mr
from interbotix_sdk import angle_manipulation as ang
from urdf_parser_py.urdf import URDF
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from interbotix_sdk.msg import JointCommands
from interbotix_sdk.msg import SingleCommand
from interbotix_sdk.srv import RobotInfo, RobotInfoRequest, RobotInfoResponse
from interbotix_sdk.srv import OperatingModes
from interbotix_sdk.srv import OperatingModesRequest
from interbotix_sdk.srv import RegisterValues
from interbotix_sdk.srv import RegisterValuesRequest
from interbotix_sdk.srv import RegisterValuesResponse

from gazebo_msgs.srv import SpawnModel
from geometry_msgs.msg import Pose


class GazeboShim(object):
    def __init__(self, use_gripper=True):
        rospy.init_node("armlab_gazebo_shim")

        robot = URDF.from_xml_file(
            "./URDFs/robot.urdf"
        )  #"/home/stanlew/src/armlab-w20-soln/robot.urdf")

        # member variables
        self.joint_names = []
        self.joint_ids = [
            1, 2, 4, 5, 6, 7
        ]  # hard coded for now - will need to get from config yaml
        self.lower_joint_limits = []
        self.upper_joint_limits = []
        self.velocity_limits = []
        self.lower_gripper_limit = 0.
        self.upper_gripper_limit = 0.
        self.use_gripper = use_gripper
        self.home_pos = [0.0, 0.0, 0.0, 0.0, 0.0]  # hard coded for now
        self.sleep_pos = [0, -1.80, -1.55, -0.8, 0]  # hard coded for now
        self.num_joints = 0
        self.num_single_joints = 0
        self.motor_register_mock = []
        self.gripper_pos_cmd = []

        # parse URDF
        self.info_from_urdf(robot)

        # services
        self.robo_info_srv = rospy.Service('rx200/get_robot_info', RobotInfo,
                                           self.get_robot_info)
        self.operating_mode_srv = rospy.Service('rx200/set_operating_modes',
                                                OperatingModes,
                                                self.set_operating_mode)
        self.set_motor_reg_srv = rospy.Service(
            'rx200/set_motor_register_values', RegisterValues,
            self.set_motor_reg)
        self.get_motor_reg_srv = rospy.Service(
            'rx200/get_motor_register_values', RegisterValues,
            self.get_motor_reg)
        # subscribers
        self.joint_state_sub = rospy.Subscriber("/rx200/joint_states",
                                                JointState,
                                                self.callback_JointState)
        self.sub_joint_commands = rospy.Subscriber("/rx200/joint/commands",
                                                   JointCommands,
                                                   self.callback_JointCommands)
        self.sub_single_command = rospy.Subscriber(
            "/rx200/single_joint/command", SingleCommand,
            self.callback_SingleCommand)
        self.sub_gripper_command = rospy.Subscriber(
            "/rx200/gripper/command", Float64, self.callback_GripperCommand)
        # publishers
        self.pub_gazebo_arm_commands = rospy.Publisher(
            "/rx200/arm_controller/command", JointTrajectory, queue_size=100)
        self.pub_gazebo_gripper_commands = rospy.Publisher(
            "/rx200/gripper_controller/command",
            JointTrajectory,
            queue_size=100)

    def info_from_urdf(self, robotURDF):
        idx = 0
        gripper_limit_established = False
        for joint in robotURDF.joints:
            if (joint.type != "fixed"):
                if ("finger" in joint.name):
                    if (self.use_gripper and not gripper_limit_established):
                        self.lower_gripper_limit = joint.limit.lower
                        self.upper_gripper_limit = joint.limit.upper
                        gripper_limit_established = True
                else:
                    self.joint_names.append(joint.name)
                    #self.joint_ids.append(idx)
                    idx = idx + 1
                    self.lower_joint_limits.append(joint.limit.lower)
                    self.upper_joint_limits.append(joint.limit.upper)
                    self.velocity_limits.append(joint.limit.velocity)
                    self.motor_register_mock.append({"Position_P_Gain" : 0.0, \
                        "Position_I_Gain" : 0.0, \
                        "Position_D_Gain" : 0.0, \
                        "Profile_Acceleration": 500, \
                        "Profile_Velocity": 2000})

        self.num_single_joints = len(self.joint_names)
        self.num_joints = self.num_single_joints - 1  # removing one because gripper

    def get_robot_info(self, req):
        # Multiple types of robot information are provided, including:
        #   1) joint_names - the names of all joints in the robot
        #   2) joint_ids - the Dynamixel IDs for all joints in the robot
        #   3) lower_joint_limits - the lower joint limits in radians (taken from URDF)
        #   4) upper_joint_limits - the upper joint limits in radians (taken from URDF)
        #   5) velocity_limits - the velocity limits in rad/s (taken from URDF)
        #   6) lower_gripper_limit - the lower gripper limit in radians (taken from URDF)
        #   7) upper_gripper_limit - the upper gripper limit in radians (taken from URDF)
        #   8) use_gripper - True if the driver node can control the gripper - otherwise, False
        #   9) home_pos - home position for each robot; essentially commands all joints
        #                 (excluding gripper) to 0 radians (taken from arm_poses.h)
        #   10) sleep_pos - sleep position for each robot; essentially commands all joints
        #                   (excluding gripper) to a specific position in radians so that
        #                   if the driver node is shutdown (which torques off all motors),
        #                   the arm doesn't come crashing down (taken from arm_poses.h)
        #   11) num_joints - the number of joints in the arm (excluding gripper)
        #   12) num_single_joints - the number of all joints in the robot (includes gripper and any 'single' joints)
        resp = RobotInfoResponse()
        resp.joint_names = self.joint_names
        resp.joint_ids = self.joint_ids
        resp.lower_joint_limits = self.lower_joint_limits
        resp.upper_joint_limits = self.upper_joint_limits
        resp.velocity_limits = self.velocity_limits
        resp.lower_gripper_limit = self.lower_gripper_limit
        resp.upper_gripper_limit = self.upper_gripper_limit
        resp.use_gripper = self.use_gripper
        resp.home_pos = self.home_pos
        resp.sleep_pos = self.sleep_pos
        resp.num_single_joints = self.num_single_joints
        resp.num_joints = self.num_joints
        return resp

    def set_operating_mode(self, req):
        # this service call returns nothing
        # but it does need to change the internal state of how commands are interpreted
        pass

    def set_motor_reg(self, req):
        # this service call returns nothing
        # but it might need to modify the gains in the Gazebo sim
        response = RegisterValuesResponse()

        if req.cmd is req.ARM_JOINTS:
            for i in range(len(self.joint_names)):
                if (self.joint_names[i] is not "gripper"):
                    addr_name = req.addr_name
                    self.motor_register_mock[i][addr_name] = req.value

        elif req.cmd is req.GRIPPER:
            raise Exception("not yet implemented!")

        elif req.cmd is req.ARM_JOINTS_AND_GRIPPER:
            raise Exception("not yet implemented!")

        elif req.cmd is req.SINGLE_MOTOR:
            motor_name = req.motor_name
            addr_name = req.addr_name
            motor_idx = self.joint_names.index(motor_name)
            self.motor_register_mock[motor_idx][addr_name] = req.value

        return response

    def get_motor_reg(self, req):
        resp = RegisterValuesResponse()
        if req.cmd is req.ARM_JOINTS:
            raise Exception("not yet implemented!")
        elif req.cmd is req.GRIPPER:
            raise Exception("not yet implemented!")
        elif req.cmd is req.ARM_JOINTS_AND_GRIPPER:
            raise Exception("not yet implemented!")
        elif req.cmd is req.SINGLE_MOTOR:
            motor_name = req.motor_name
            addr_name = req.addr_name
            motor_idx = self.joint_names.index(motor_name)
            addr_value = self.motor_register_mock[motor_idx][addr_name]
            resp.values.append(addr_value)
        return resp

    def callback_JointState(self, data):
        #print("received JointState")
        pass

    # note: profile_accels and profile_vels are actually TIMES in the control station implementation
    def generateTrajectory(self, joint_cmds, profile_accel, profile_vel):
        toReturn = JointTrajectory()

        for i in range(len(self.joint_names)):
            if (not ("gripper" in self.joint_names[i])):
                toReturn.joint_names.append(self.joint_names[i])

        trajPoint = JointTrajectoryPoint()
        trajPoint.time_from_start = rospy.Duration.from_sec(profile_accel /
                                                            1000.0)
        trajPoint.positions = joint_cmds

        toReturn.points = [trajPoint]

        print(toReturn)

        return toReturn

    def generateGripperTrajectory(self, cmd):
        toReturn = JointTrajectory()
        toReturn.joint_names.append("right_finger")
        toReturn.joint_names.append("left_finger")

        trajPoint = JointTrajectoryPoint()
        trajPoint.time_from_start = rospy.Duration.from_sec(1.0)
        if (cmd.data > 0):
            self.gripper_pos_cmd = [
                -1 * self.upper_gripper_limit, self.upper_gripper_limit
            ]
        elif (cmd.data < 0):
            self.gripper_pos_cmd = [
                -1 * self.lower_gripper_limit, self.lower_gripper_limit
            ]

        trajPoint.positions = self.gripper_pos_cmd
        trajPoint.effort = [-cmd.data * 1000, cmd.data * 1000]

        toReturn.points = [trajPoint]
        print("returning!")
        print(toReturn)

        return toReturn

    def callback_JointCommands(self, data):
        print("received JointCommands")
        trajectory =  self.generateTrajectory(data.cmd,\
                self.motor_register_mock[0]["Profile_Acceleration"],\
                self.motor_register_mock[0]["Profile_Velocity"])
        self.pub_gazebo_arm_commands.publish(trajectory)

    def callback_SingleCommand(self, data):
        print("received SingleCommand")
        pass

    def callback_GripperCommand(self, data):
        print("received GripperCommand")
        print(data)
        self.pub_gazebo_gripper_commands.publish(
            self.generateGripperTrajectory(data))
        pass


class Block(object):
    def __init__(self, sdf_contents, pose_t, pose_euler):
        self.sdf_data = sdf_contents
        self.pose_t = pose_t
        self.pose_euler = pose_euler

    def spawn(self, model_spawner, idx):
        block_pose = Pose()
        block_pose.position.x = self.pose_t[0]
        block_pose.position.y = self.pose_t[1]
        block_pose.position.z = self.pose_t[2]
        quat = euler_to_quaternion(self.pose_euler[0], self.pose_euler[1],
                                   self.pose_euler[2])
        block_pose.orientation.x = quat[0]
        block_pose.orientation.y = quat[1]
        block_pose.orientation.z = quat[2]
        block_pose.orientation.w = quat[3]
        model_spawner("block_" + str(idx), self.sdf_data, "block_" + str(idx),
                      block_pose, "world")


class BlockSpawner(object):
    def __init__(self, config=None, num_random_blocks=0):
        self.baseBlockSdf = "./URDFs/block_red.sdf"
        self.colorDict = {\
            "red" : "1 0 0 1",\
            "pink" : "1 .4 .4 1",\
            "blue" : "0 0 1 1",\
            "green" : "0 0.5 0 1",\
            "yellow" : "1 1 0 1",\
            "orange" :  "1 0.5 0 1",\
            "purple" : "0.5 0 1 1",\
            "black" : "0 0 0 1"}

        self.blocks = []

        if config is not None:
            configData = json.load(open(config, 'r'))
            for blockConfig in configData["blocks"]:
                sdf_contents = open(self.baseBlockSdf, 'r').read()
                block_color = blockConfig["color"]
                replace_color = self.colorDict[block_color]
                sdf_contents = sdf_contents.replace(">1 0 0 1<",
                                                    ">" + replace_color + "<")
                block = Block(sdf_contents, blockConfig["pose_t"],
                              blockConfig["pose_euler"])
                self.blocks.append(block)

        for i in range(num_random_blocks):
            sdf_contents = open(self.baseBlockSdf, 'r').read()
            sdf_contents = sdf_contents.replace(
                ">1 0 0 1<",
                ">" + random.choice(list(self.colorDict.values())) + "<")
            pose_t = [random.uniform(0.1, 0.3) * random.choice([-1, 1]),\
                 random.uniform(0.1, 0.3) * random.choice([-1, 1]),\
                      random.uniform(0.1, 0.3)]
            pose_e = [
                random.uniform(-np.pi, np.pi),
                random.uniform(-np.pi, np.pi),
                random.uniform(-np.pi, np.pi)
            ]
            block = Block(sdf_contents, pose_t, pose_e)
            self.blocks.append(block)

    def spawnAllBlocks(self, model_spawner):
        i = 0
        for block in self.blocks:
            block.spawn(model_spawner, i)
            i = i + 1


# shamelessly taken from https://stackoverflow.com/questions/53033620/how-to-convert-euler-angles-to-quaternions-and-get-the-same-euler-angles-back-fr
def euler_to_quaternion(yaw, pitch, roll):

    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(
        roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(
        roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(
        roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(
        roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)

    return [qx, qy, qz, qw]


if __name__ == '__main__':
    shim = GazeboShim()

    realsense_pose = Pose()
    realsense_pose.position.x = 0
    realsense_pose.position.y = 0
    realsense_pose.position.z = 1.0
    quat = euler_to_quaternion(0.0, 3.1415 / 2, 0.0)
    realsense_pose.orientation.x = quat[0]
    realsense_pose.orientation.y = quat[1]
    realsense_pose.orientation.z = quat[2]
    realsense_pose.orientation.w = quat[3]

    light_pose = Pose()
    light_pose.position.x = 0
    light_pose.position.y = 0
    light_pose.position.z = 1.2
    quat = euler_to_quaternion(0.0, 3.1415 / 2, 0.0)
    light_pose.orientation.x = quat[0]
    light_pose.orientation.y = quat[1]
    light_pose.orientation.z = quat[2]
    light_pose.orientation.w = quat[3]

    board_pose = Pose()
    # the 0.321 and -0.286 are because the origin of the .obj file wasnt placed at 0,0
    board_pose.position.x = 0.321 + 0.305
    board_pose.position.y = -0.286 - 0.305
    board_pose.position.z = -0.019
    quat = euler_to_quaternion(3.1415 / 2, 0.0, 0.0)
    board_pose.orientation.x = quat[0]
    board_pose.orientation.y = quat[1]
    board_pose.orientation.z = quat[2]
    board_pose.orientation.w = quat[3]

    tag_pose = Pose()
    # the 0.321 and -0.286 are because the origin of the .obj file wasnt placed at 0,0
    tag_pose.position.x = -0.1425
    tag_pose.position.y = 0.0
    tag_pose.position.z = 0.038
    quat = euler_to_quaternion(-3.1415 / 2, 3.1415, 0.0)
    tag_pose.orientation.x = quat[0]
    tag_pose.orientation.y = quat[1]
    tag_pose.orientation.z = quat[2]
    tag_pose.orientation.w = quat[3]

    print("waiting on spawn_sdf_model service....")
    rospy.wait_for_service("gazebo/spawn_sdf_model")
    spawn_model_proxy = rospy.ServiceProxy("gazebo/spawn_sdf_model",
                                           SpawnModel)

    print("loading realsense model")
    f = open(
        "./gazebo_shim_deps/gazebo-realsense/models/realsense_camera/model.sdf",
        "r")
    sdff = f.read()
    # perhaps less than ideal way of turning off gravity
    sdff = sdff.replace("<gravity>1", "<gravity>0")
    print("calling spawn_sdf_model service for realsense")
    spawn_model_proxy("realsense_cam", sdff, "realsense_ns", realsense_pose,
                      "world")

    print("loading light model")
    f = open("./URDFs/overhead_light.sdf", "r")
    sdff = f.read()
    print("calling spawn_sdf_model service for light")
    spawn_model_proxy("overhead_light", sdff, "overhead_light", light_pose,
                      "world")

    print("loading board model")
    f = open("./URDFs/550board.sdf", "r")
    sdff = f.read()
    print("calling spawn_sdf_model service for board")
    spawn_model_proxy("550board", sdff, "550_board", board_pose, "world")

    print("loading final apriltag model")
    f = open("./URDFs/robot_tag.sdf", "r")
    sdff = f.read()
    print("calling spawn_sdf_model service for tag")
    spawn_model_proxy("robotApril", sdff, "robot_april", tag_pose, "world")

    print("loading blocks config & spawning blocks")
    #use the last positional arg to randomly generate blocks
    blockSpawner = BlockSpawner("sample_block_config.json", 0)
    blockSpawner.spawnAllBlocks(spawn_model_proxy)

    rospy.spin()
