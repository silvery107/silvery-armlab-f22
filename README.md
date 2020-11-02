Launch realsense:

roslaunch realsense2_camera rs_camera.launch align_depth:=true

Launch rx200 arm:

roslaunch interbotix_sdk arm_run.launch robot_name:=rx200 use_time_based_profile:=true gripper_operating_mode:=pwm
