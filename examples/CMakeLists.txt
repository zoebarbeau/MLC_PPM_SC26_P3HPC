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

if ( DEFINED ENV{PROTO_HOME} )
message ( STATUS "PROTO_HOME = $ENV{PROTO_HOME}" )
set ( PROTO_SOURCE_DIR $ENV{PROTO_HOME} )
else ()
if ( "x${PROTO_HOME}" STREQUAL "x" )
    message ( FATAL_ERROR "PROTO_HOME environment variable undefined and not specified on command line" )
endif ()
set ( PROTO_SOURCE_DIR ${PROTO_HOME} )
endif ()

##  Include FFTX CMake functions
include ( "${FFTX_SOURCE_DIR}/CMakeIncludes/FFTXCmakeFunctions.cmake" )
# target_link_libraries   ( exampm PRIVATE dl )

add_executable( Hill Hills_Vortex.cpp )
#target_compile_definitions(DIM PUBLIC DIM=3)
target_include_directories( Hill PUBLIC ${FFTX_SOURCE_DIR}/examples/rconv) # ${PROTO_SOURCE_DIR}/include ${PROTO_SOURCE_DIR}/include/ops ${PROTO_SOURCE_DIR}/include/utils ${PROTO_SOURCE_DIR}/include/base )
target_link_libraries( Hill PRIVATE exampm dl fftw3 )
FFTX_add_includes_libs_to_target ( Hill )

#install(TARGETS DamBreak FreeFall initGrid DESTINATION ${CMAKE_INSTALL_BINDIR})
install(TARGETS Hill DESTINATION ${CMAKE_INSTALL_BINDIR})
