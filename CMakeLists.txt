cmake_minimum_required(VERSION 2.8)
project(GSE)

set(GSE_MAJOR_VERSION 0)
set(GSE_MINOR_VERSION 1)
set(GSE_PATCH_VERSION 0)
set(GSE_VERSION
  ${GSE_MAJOR_VERSION}.${GSE_MINOR_VERSION}.${GSE_PATCH_VERSION})

# Offer the user the choice of overriding the installation directories
set(INSTALL_LIB_DIR lib CACHE PATH "Installation directory for libraries")
set(INSTALL_BIN_DIR bin CACHE PATH "Installation directory for executables")
set(INSTALL_INCLUDE_DIR include CACHE PATH
  "Installation directory for header files")
if(WIN32 AND NOT CYGWIN)
  set(DEF_INSTALL_CMAKE_DIR CMake)
else()
  set(DEF_INSTALL_CMAKE_DIR lib/CMake/GSE)
endif()
set(INSTALL_CMAKE_DIR ${DEF_INSTALL_CMAKE_DIR} CACHE PATH
  "Installation directory for CMake files")

# Make relative paths absolute (needed later on)
foreach(p LIB BIN INCLUDE CMAKE)
  set(var INSTALL_${p}_DIR)
  if(NOT IS_ABSOLUTE "${${var}}")
    set(${var} "${CMAKE_INSTALL_PREFIX}/${${var}}")
  endif()
endforeach()

# set up include-directories
include_directories(
  "${PROJECT_SOURCE_DIR}"   # to find foo/foo.h
  "${PROJECT_BINARY_DIR}")  # to find foo/config.h
  
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11")

# Add sub-directories
add_subdirectory(GSE)
add_subdirectory(samples)

# The interesting stuff goes here
# ===============================

# Add all targets to the build-tree export set
export(TARGETS FILE "${PROJECT_BINARY_DIR}/GSETargets.cmake")

# Export the package for use from the build-tree
# (this registers the build-tree with a global CMake-registry)
export(PACKAGE GSE)

# Create the GSEConfig.cmake and GSEConfigVersion files
file(RELATIVE_PATH REL_INCLUDE_DIR "${INSTALL_CMAKE_DIR}" "${INSTALL_INCLUDE_DIR}")
# ... for the build tree
set(CONF_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}" "${PROJECT_BINARY_DIR}")
configure_file(GSEConfig.cmake.in "${PROJECT_BINARY_DIR}/GSEConfig.cmake" @ONLY)
# ... for the install tree
set(CONF_INCLUDE_DIRS "\${GSE_CMAKE_DIR}/${REL_INCLUDE_DIR}")
configure_file(GSEConfig.cmake.in "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/GSEConfig.cmake" @ONLY)
# ... for both
configure_file(GSEConfigVersion.cmake.in "${PROJECT_BINARY_DIR}/GSEConfigVersion.cmake" @ONLY)

# Install the GSEConfig.cmake and GSEConfigVersion.cmake
install(FILES
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/GSEConfig.cmake"
  "${PROJECT_BINARY_DIR}/GSEConfigVersion.cmake"
  DESTINATION "${INSTALL_CMAKE_DIR}" COMPONENT dev)

# Install the export set for use with the install-tree
#install(EXPORT GSETargets DESTINATION "${INSTALL_CMAKE_DIR}" COMPONENT dev)
