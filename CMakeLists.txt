cmake_minimum_required(VERSION 3.12.0)
project(lab01 LANGUAGES CXX C)

include(common/cmake-common.cmake)
add_compile_options(-g)
add_executable(proj  src/Steady_NavStokes.cpp src/main.cpp)
deal_ii_setup_target(proj)
