# add_executable( DamBreak dam_break.cpp )
# target_link_libraries( DamBreak PRIVATE exampm)

# add_executable( FreeFall free_fall.cpp )
# target_link_libraries( FreeFall PRIVATE exampm)

##  FFTX_HOME must be defined in the environment or on the command line
#if ( DEFINED ENV{FFTX_HOME} )
#message ( STATUS "FFTX_HOME = $ENV{FFTX_HOME}" )
#set ( FFTX_SOURCE_DIR $ENV{FFTX_HOME} )
#else ()
#if ( "x${FFTX_HOME}" STREQUAL "x" )
#    message ( FATAL_ERROR "FFTX_HOME environment variable undefined and not specified on command line" )
#endif ()
#set ( FFTX_SOURCE_DIR ${FFTX_HOME} )
#endif ()
#
## --- PATCH: check install/lib if it exists ---
#if (EXISTS "${FFTX_SOURCE_DIR}/install/lib")
#  set(FFTX_LIB_DIR "${FFTX_SOURCE_DIR}/install/lib")
#elseif(EXISTS "${FFTX_SOURCE_DIR}/lib")
#  set(FFTX_LIB_DIR "${FFTX_SOURCE_DIR}/lib")
#else()
#  message(FATAL_ERROR "${FFTX_SOURCE_DIR} has no lib or install/lib")
#endif()
#list(APPEND CMAKE_PREFIX_PATH "${FFTX_LIB_DIR}/..")
#
###  Include FFTX CMake functions
#include ( "${FFTX_SOURCE_DIR}/CMakeIncludes/FFTXCmakeFunctions.cmake" )
##include_directories("${CMAKE_SOURCE_DIR}/path/to/Common")
#
## add_executable          ( ${SOURCES} ${SOURCES} )
## target_link_libraries   ( exampm PRIVATE dl )
#
#add_executable( Hill Hills_Vortex.cpp )
#target_include_directories( Hill PUBLIC ${FFTX_SOURCE_DIR}/examples/rconv )
#
#target_link_libraries( Hill PRIVATE exampm dl nvrtc)
#FFTX_add_includes_libs_to_target ( Hill )
#target_compile_definitions(Hill PRIVATE FFTX_CUDA)
#
##install(TARGETS DamBreak FreeFall initGrid DESTINATION ${CMAKE_INSTALL_BINDIR})
#install(TARGETS Hill DESTINATION ${CMAKE_INSTALL_BINDIR})


# Ensure FFTX_HOME is defined
if (DEFINED ENV{FFTX_HOME})
    message(STATUS "FFTX_HOME = $ENV{FFTX_HOME}")
    set(FFTX_HOME $ENV{FFTX_HOME})
else()
    if ("x${FFTX_HOME}" STREQUAL "x")
        message(FATAL_ERROR "FFTX_HOME environment variable undefined and not specified on command line")
    endif()
endif()


# Include FFTX CMake helper functions
include("${FFTX_HOME}/CMakeIncludes/FFTXCmakeFunctions.cmake")

# --- Hill example ---
add_executable(Hill Hills_Vortex.cpp)

# Include FFTX headers
target_include_directories(Hill PUBLIC
    $ENV{FFTX_HOME}/examples/rconv
)

# Link with ExaMPM library, FFTX, and other dependencies
target_link_libraries(Hill PRIVATE exampm dl nvrtc) # nvToolsExt)
# Use FFTX helper to link the FFTX libraries
FFTX_add_includes_libs_to_target(Hill
    LIB_DIR ${FFTX_LIB_DIR}
)

# Define compilation flags for FFTX CUDA
target_compile_definitions(Hill PRIVATE FFTX_CUDA)

# Install target`
install(TARGETS Hill DESTINATION ${CMAKE_INSTALL_BINDIR})

