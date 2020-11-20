cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

find_package(gazebo REQUIRED)
find_package(roscpp REQUIRED)
find_package(sensor_msgs REQUIRED)

include_directories(${GAZEBO_INCLUDE_DIRS})
include_directories(${roscpp_INCLUDE_DIRS})
include_directories(${sensor_msgs_INCLUDE_DIRS})

link_directories(${GAZEBO_LIBRARY_DIRS})
link_directories(${roscpp_LIBRARY_DIRS})
link_directories(${sensor_msgs_LIBRARY_DIRS})

list(APPEND CMAKE_CXX_FLAGS "${GAZEBO_CXX_FLAGS}")

add_executable(listener listener.cc)
target_link_libraries(listener ${GAZEBO_LIBRARIES} ${roscpp_LIBRARIES} ${sensor_msgs_LIBRARIES} pthread)
