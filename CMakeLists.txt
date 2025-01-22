cmake_minimum_required(VERSION 3.21)
project(YoloV8-ONNX)
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/build/modules)

find_package(onnxruntime REQUIRED)
find_package(stb REQUIRED)

add_executable(yolov8 src/main.cpp)
target_compile_features(yolov8 PUBLIC cxx_std_20)
target_link_libraries(yolov8 PRIVATE onnxruntime::onnxruntime stb::stb)

# Copy ONNX model in the executable directory
add_custom_command(TARGET yolov8 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_SOURCE_DIR}/yolov8n.onnx ${CMAKE_CURRENT_BINARY_DIR}/yolov8n.onnx
    COMMENT "Copying yolov8n.onnx"
)

# Copy images directory in the executable directory
add_custom_command(TARGET yolov8 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/images ${CMAKE_CURRENT_BINARY_DIR}/images
    COMMENT "Copying images directory"
)

install(TARGETS yolov8 DESTINATION .)
install(FILES ${CMAKE_SOURCE_DIR}/yolov8n.onnx DESTINATION .)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/images DESTINATION .)
