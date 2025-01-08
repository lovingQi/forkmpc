cmake_minimum_required(VERSION 3.10)
project(mpc_forklift)

# 添加Eigen头文件路径
include_directories(${PROJECT_SOURCE_DIR}/Eigen)

find_package(osqp REQUIRED)

add_executable(mpc_forklift main.cpp mpc_forklift.cpp)
target_link_libraries(mpc_forklift osqp::osqp) 