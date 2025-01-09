cmake_minimum_required(VERSION 3.10)
project(forkmpc)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Release)

# 查找依赖包
find_package(Eigen3 REQUIRED)
find_package(osqp REQUIRED)

# 添加可执行文件
add_executable(mpc_forklift 
    main.cpp
    j_mpc_flt.cpp    # 添加新的源文件
)

# 包含目录
target_include_directories(mpc_forklift PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${EIGEN3_INCLUDE_DIRS}
)

# 链接库
target_link_libraries(mpc_forklift PRIVATE
    osqp::osqp
) 