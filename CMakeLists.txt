cmake_minimum_required(VERSION 3.12.0)
project(lab01 LANGUAGES CXX C)
add_compile_options(-g)
include(common/cmake-common.cmake)

add_executable(main src/main.cpp src/NSSteady.cpp src/NSUnsteady.cpp)

deal_ii_setup_target(main)
