cmake_minimum_required(VERSION 3.20)
project(GP_MPC)

find_package(OsqpEigen REQUIRED)


set(CMAKE_CXX_STANDARD 17)

add_executable(GP_MPC main.cpp gaussian_process.cpp gaussian_process.h)
target_link_libraries(GP_MPC OsqpEigen::OsqpEigen)
