cmake_minimum_required(VERSION 3.12.0)
project(tut LANGUAGES CXX C)

include(../common/cmake-common.cmake)

add_executable(tut src/main.cpp src/NavierStokesSolver.cpp)
deal_ii_setup_target(tut)
