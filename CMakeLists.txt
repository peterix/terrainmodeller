cmake_minimum_required(VERSION 2.8.0 FATAL_ERROR)
PROJECT( terrainmodeller )

FIND_PACKAGE( OpenCV REQUIRED )
FIND_PACKAGE( OpenMP ) 

if (OPENMP_FOUND) 
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}") 
endif (OPENMP_FOUND) 

ADD_EXECUTABLE( terrainmodeller functions.cpp tm.cpp )
TARGET_LINK_LIBRARIES( terrainmodeller ${OpenCV_LIBS} ) 
