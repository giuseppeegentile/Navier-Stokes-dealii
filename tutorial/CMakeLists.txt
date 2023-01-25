cmake_minimum_required(VERSION 3.12.0)
project(lab01 LANGUAGES CXX C)
add_compile_options()
include(../common/cmake-common.cmake)

add_executable(tut src/main.cpp)
deal_ii_setup_target(tut)
