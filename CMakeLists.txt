cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_STANDARD 11)
message("CMAKE_CXX_STANDARD = ${CMAKE_CXX_STANDARD}")

project(app)


#Executable files
add_executable(${PROJECT_NAME} src/main.cpp)

# add_subdirectory(lib/opencv)
set(CMAKE_INSTALL_RPATH "${PROJECT_SOURCE_DIR}/lib/opencv-4/lib/")
list(APPEND EXTRA_LIB_DIRS ${PROJECT_SOURCE_DIR}/lib/opencv-4/lib)
list(APPEND EXTRA_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/lib/opencv-4/include/opencv4)
list(APPEND EXTRA_LINKS  opencv_aruco 
opencv_bgsegm 
opencv_bioinspired 
opencv_calib3d 
opencv_ccalib 
opencv_core 
opencv_datasets 
opencv_dnn_objdetect 
opencv_dnn 
opencv_dpm 
opencv_face 
opencv_features2d 
opencv_flann 
opencv_fuzzy 
opencv_gapi 
opencv_hfs 
opencv_highgui 
opencv_imgcodecs 
opencv_img_hash 
opencv_imgproc 
opencv_line_descriptor 
opencv_ml 
opencv_objdetect 
opencv_optflow 
opencv_phase_unwrapping 
opencv_photo 
opencv_plot 
opencv_quality 
opencv_reg 
opencv_rgbd 
opencv_saliency 
opencv_shape 
opencv_stereo 
opencv_stitching 
opencv_structured_light 
opencv_superres 
opencv_surface_matching 
opencv_text 
opencv_tracking 
opencv_videoio 
opencv_video 
opencv_videostab 
opencv_xfeatures2d 
opencv_ximgproc 
opencv_xobjdetect 
opencv_xphoto)

#Add INCLUDE directories of third-party libraries
target_include_directories(${PROJECT_NAME} 
    PUBLIC ${PROJECT_BINARY_DIR}
    PUBLIC ${EXTRA_INCLUDE_DIRS})

#Add LIB or SRC directories of third-party libraries
target_link_directories(${PROJECT_NAME} PRIVATE ${EXTRA_LIB_DIRS})

#Add the name of third-party libraries
target_link_libraries(${PROJECT_NAME} ${EXTRA_LINKS})



