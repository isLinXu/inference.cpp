cmake_minimum_required(VERSION 3.24)
project(inference.cpp)

set(CMAKE_CXX_STANDARD 17)

set(INFERENCE_CPP_DIR ${CMAKE_SOURCE_DIR}/library)
# 添加 library 头文件和库文件路径
set(INFERENCE_CPP_INCLUDE_DIR ${INFERENCE_CPP_DIR}/include)
set(INFERENCE_CPP_LIBRARY_DIR ${INFERENCE_CPP_DIR}/libs)
include_directories(${INFERENCE_CPP_INCLUDE_DIR})
link_directories(${INFERENCE_CPP_LIBRARY_DIR})

add_executable(inference main.cpp)

# 链接 library 库
target_link_libraries(
        inference
        ${tnn}
        ${mnn}
)
