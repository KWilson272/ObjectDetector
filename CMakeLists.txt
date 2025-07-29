cmake_minimum_required(VERSION 3.15)
project(ObjectDetector VERSION 1.0.0)

find_package(depthai REQUIRED)
find_package(OpenCV REQUIRED)

# cxxopts is a small header-only library; we dont need an intensive installation 
include(FetchContent)
FetchContent_Declare(
  cxxopts
  GIT_REPOSITORY git@github.com:jarro2783/cxxopts.git 
  GIT_TAG        v3.3.1
)
FetchContent_MakeAvailable(cxxopts)

set(SOURCES
    src/main.cpp
    src/detection_display.cpp
)

add_executable(${PROJECT_NAME} ${SOURCES})

target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(${PROJECT_NAME} 
  PRIVATE 
    cxxopts
    depthai::core 
    ${OpenCV_LIBS}
)