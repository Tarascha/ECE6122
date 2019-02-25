cmake_minimum_required(VERSION 3.9.1)
project(P3)
find_package(CUDA QUIET REQUIRED)
SET(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-std=c++11 -pthread")
Cuda_add_executable(p33 final.cu)


find_package (Threads)
include_directories(
src
)
set(src
src/complex.h
src/complex.cc
src/input_image.h
src/input_image.cc
)
add_executable(p31 ${src} main.cc)

target_link_libraries (p31 ${CMAKE_THREAD_LIBS_INIT})

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

SET(CMAKE_C_COMPILER mpicc)
SET(CMAKE_CXX_COMPILER mpicxx)


add_executable(p32 fft2d.cc)
