#INCLUDE_DIRECTORIES(/usr/include/arpack++/include/)
#INCLUDE_DIRECTORIES(/usr/include/arpack++)

#project name
PROJECT(LILIANTESTS)

cmake_minimum_required(VERSION 2.8)
# find_package( OpenCV 3.4 REQUIRED )


SET(BUILD_SHARED_LIBS ON)

## where are user-specific cmake modules
SET(ARPACK_INCLUDE_DIR /usr/local/include/arpackpp)
# SET(CMAKE_MODULE_PATH $ENV{CMAKE_MODULE_PATH})

INCLUDE_DIRECTORIES(${ARPACK_INCLUDE_DIR} /usr/local/include/arpackpp)


# source files of library "LineMatchingLib" to be created
SET(LineMatchingLib_SRCS
    PairwiseLineMatching.cpp
    LineDescriptor.cpp
    EDLineDetector.cpp
)
# header files to be installed
SET(LineMatchingLib_HEADER
    PairwiseLineMatching.hh
    LineDescriptor.hh
    EDLineDetector.hh
    LineStructure.hh
)


ADD_LIBRARY(LineMatchingLib
            ${LineMatchingLib_SRCS}
            ${LineMatchingLib_HEADER})
    
TARGET_LINK_LIBRARIES(LineMatchingLib ${OpenCV_LIBRARIES} arpack /usr/lib/x86_64-linux-gnu/libsuperlu.so) 


# ADD_EXECUTABLE(Test_Single_LMA Demo_LBD_single.cpp)
# TARGET_LINK_LIBRARIES(Test_Single_LMA   LineMatchingLib ${OpenCV_LIBRARIES})







