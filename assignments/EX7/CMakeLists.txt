cmake_minimum_required(VERSION 2.8)

project(EX7)

find_package(PCL REQUIRED)

include_directories("/usr/include/eigen3" ${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(icp icp.cpp)
add_executable(pcl_icp pcl_icp.cpp)

target_link_libraries(icp ${PCL_LIBRARIES})
target_link_libraries(pcl_icp ${PCL_LIBRARIES})