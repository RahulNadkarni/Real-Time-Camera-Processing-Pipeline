cmake_minimum_required(VERSION 3.14)
project(RealTimeCameraPipeline LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Faster rebuilds: use Ninja and parallel jobs by default
if(NOT CMAKE_GENERATOR STREQUAL "Ninja" AND NOT CMAKE_BUILD_PARALLEL_LEVEL)
  set(CMAKE_BUILD_PARALLEL_LEVEL 4 CACHE STRING "Parallel build jobs")
endif()

# OpenCV (with DNN module for neural stages / ONNX)
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui videoio dnn)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Threads REQUIRED)

# Source files
set(PIPELINE_SOURCES
    src/pipeline/pipeline.cpp
    src/pipeline/frame_pool.cpp
)
# Existing image processing stages
set(STAGE_SOURCES
    src/stages/debayer_stage.cpp
    src/stages/noise_reduction_stage.cpp
    src/stages/tone_mapping_stage.cpp
    src/stages/histogram_stage.cpp
    src/stages/edge_detection_stage.cpp
)
# Neural inference stages (OpenCV DNN, ONNX) — add below existing stage files
list(APPEND STAGE_SOURCES
    src/stages/neural/scene_classifier_stage.cpp
    src/stages/neural/saliency_stage.cpp
    src/stages/neural/super_resolution_stage.cpp
    src/stages/neural/neural_dispatcher.cpp
)
set(PROFILING_SOURCES
    src/profiling/pipeline_stats.cpp
    src/profiling/scoped_timer.cpp
)
set(CONTROL_SOURCES
    src/controls/stage_controller.cpp
)
set(DISPLAY_SOURCES
    src/display/renderer.cpp
)

add_executable(${PROJECT_NAME}
    src/main.cpp
    ${PIPELINE_SOURCES}
    ${STAGE_SOURCES}
    ${PROFILING_SOURCES}
    ${CONTROL_SOURCES}
    ${DISPLAY_SOURCES}
)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    ${OpenCV_LIBS}
    Threads::Threads
)
