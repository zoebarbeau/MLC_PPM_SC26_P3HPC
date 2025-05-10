# Install script for directory: /g/g16/barbeau2/GPU/MLC_PPM/src

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/g/g16/barbeau2/GPU/MLC_PPM/build/install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Debug")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "0")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES
    "/g/g16/barbeau2/GPU/MLC_PPM/src/ExaMPM_BoundaryConditions.hpp"
    "/g/g16/barbeau2/GPU/MLC_PPM/src/ExaMPM_Convolution.hpp"
    "/g/g16/barbeau2/GPU/MLC_PPM/src/ExaMPM_DenseLinearAlgebra.hpp"
    "/g/g16/barbeau2/GPU/MLC_PPM/src/ExaMPM_DriverGrid.hpp"
    "/g/g16/barbeau2/GPU/MLC_PPM/src/ExaMPM_GreensFunction.hpp"
    "/g/g16/barbeau2/GPU/MLC_PPM/src/ExaMPM_GridManager.hpp"
    "/g/g16/barbeau2/GPU/MLC_PPM/src/ExaMPM_LocalCorrection.hpp"
    "/g/g16/barbeau2/GPU/MLC_PPM/src/ExaMPM_Mesh.hpp"
    "/g/g16/barbeau2/GPU/MLC_PPM/src/ExaMPM_MLC_Interp.hpp"
    "/g/g16/barbeau2/GPU/MLC_PPM/src/ExaMPM_ParticleInit.hpp"
    "/g/g16/barbeau2/GPU/MLC_PPM/src/ExaMPM_ProblemManager2.hpp"
    "/g/g16/barbeau2/GPU/MLC_PPM/src/ExaMPM_Remap.hpp"
    "/g/g16/barbeau2/GPU/MLC_PPM/src/ExaMPM_Solver2.hpp"
    "/g/g16/barbeau2/GPU/MLC_PPM/src/ExaMPM_Types.hpp"
    "/g/g16/barbeau2/GPU/MLC_PPM/src/ExaMPM_VelocityInterpolation.hpp"
    "/g/g16/barbeau2/GPU/MLC_PPM/src/ExaMPM_VInterp.hpp"
    "/g/g16/barbeau2/GPU/MLC_PPM/src/ExaMPM_VInterpolation.hpp"
    "/g/g16/barbeau2/GPU/MLC_PPM/src/ExaMPM_FFTW_rconv_test.hpp"
    "/g/g16/barbeau2/GPU/MLC_PPM/src/FFTWLGFConvolution.H"
    "/g/g16/barbeau2/GPU/MLC_PPM/src/FFTWLGFConvolutionImplem.H"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib64" TYPE STATIC_LIBRARY FILES "/g/g16/barbeau2/GPU/MLC_PPM/build/src/libexampm.a")
endif()

