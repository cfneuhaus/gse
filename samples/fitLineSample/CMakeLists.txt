set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}")
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

ADD_DEFINITIONS(-DGSE_NO_CHOLMOD)

add_executable(fitLineSample FitLineSample.cpp main.cpp)

target_link_libraries(fitLineSample pthread)

#install(TARGETS bar
  # IMPORTANT: Add the bar executable to the "export-set"
 # EXPORT FooBarTargets
  #RUNTIME DESTINATION "${INSTALL_BIN_DIR}" COMPONENT bin)
