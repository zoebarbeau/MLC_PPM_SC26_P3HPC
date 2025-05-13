# add_executable( DamBreak dam_break.cpp )
# target_link_libraries( DamBreak PRIVATE exampm)

# add_executable( FreeFall free_fall.cpp )
# target_link_libraries( FreeFall PRIVATE exampm)

##  FFTX_HOME must be defined in the environment or on the command line
if ( DEFINED ENV{FFTX_HOME} )
message ( STATUS "FFTX_HOME = $ENV{FFTX_HOME}" )
set ( FFTX_SOURCE_DIR $ENV{FFTX_HOME} )
else ()
if ( "x${FFTX_HOME}" STREQUAL "x" )
    message ( FATAL_ERROR "FFTX_HOME environment variable undefined and not specified on command line" )
endif ()
set ( FFTX_SOURCE_DIR ${FFTX_HOME} )
endif ()

##  Include FFTX CMake functions
include ( "${FFTX_SOURCE_DIR}/CMakeIncludes/FFTXCmakeFunctions.cmake" )
#include_directories("${CMAKE_SOURCE_DIR}/path/to/Common")

# add_executable          ( ${SOURCES} ${SOURCES} )
# target_link_libraries   ( exampm PRIVATE dl )

add_executable( Hill Hills_Vortex.cpp )
target_include_directories( Hill PUBLIC ${FFTX_SOURCE_DIR}/examples/rconv )

target_link_libraries( Hill PRIVATE exampm dl nvrtc fftw3)
FFTX_add_includes_libs_to_target ( Hill )
target_compile_definitions(Hill PRIVATE FFTX_CUDA)

#install(TARGETS DamBreak FreeFall initGrid DESTINATION ${CMAKE_INSTALL_BINDIR})
install(TARGETS Hill DESTINATION ${CMAKE_INSTALL_BINDIR})
