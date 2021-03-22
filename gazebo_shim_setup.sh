echo "%%%%%%%%%%%%% installing gazebo-realsense plugin %%%%%%%%%%%%%"
mkdir gazebo_shim_deps
cd gazebo_shim_deps
git clone https://github.com/intel/gazebo-realsense
cd gazebo-realsense
mv ./models/realsense_camera/model.config ./models/realsense_camera/model.config.old
mv ./models/realsense_camera/model.sdf ./models/realsense_camera/model.sdf.old
cp ./../../URDFs/rs_l515_550.sdf ./models/realsense_camera/model.sdf
mkdir build
cd build
cmake ..
make
sudo make install
echo "%%%%%%%%%%%%% DONE installing gazebo-realsense plugin %%%%%%%%%%%%%"
cd ../../..
echo "%%%%%%%%%%%%% running xml macros & generating URDF's %%%%%%%%%%%%%"
cd URDFs
test -f "$1/src/interbotix_ros_arms/interbotix_descriptions/urdf/rx200.urdf.xacro"
cp rx200.urdf.xacro $1/src/interbotix_ros_arms/interbotix_descriptions/urdf/rx200.urdf.xacro
rosrun xacro xacro $1/src/interbotix_ros_arms/interbotix_descriptions/urdf/rx200.urdf.xacro > robot.urdf
rosrun xacro xacro ./550board.sdf.xacro uri:=$PWD/RXArm_Board.obj > 550board.sdf
rosrun xacro xacro ./robot_tag.sdf.xacro uri:=$PWD/TagStd41h12_id01.obj > robot_tag.sdf
echo "%%%%%%%%%%%%% DONE running xml macros & generating URDF's %%%%%%%%%%%%%"
cd ..
echo "%%%%%%%%%%%%% Building gazebo-realsense msg shim %%%%%%%%%%%%%"
cd gazebo_shim_rs
mkdir build
cd build
cmake ..
make
echo "%%%%%%%%%%%%% DONE Building gazebo-realsense msg shim %%%%%%%%%%%%%"
