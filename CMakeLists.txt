cmake_minimum_required(VERSION 3.17)
project(imls_slam CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenMP for parallelizing loop computations
find_package(OpenMP REQUIRED)

# Locate Eigen3
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Include current source directory where nanoflann.hpp lives
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_executable(test_features
        code/MapManager.h
        code/MapManager.cpp
        code/FeatureExtractor.cpp
        code/FeatureExtractor.h
        code/test/test_featureExtractor.cpp
)

target_link_libraries(test_features PRIVATE
        Eigen3::Eigen
        OpenMP::OpenMP_CXX
)