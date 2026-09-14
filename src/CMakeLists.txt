set(HEADERS
MLC_BoundaryConditions.hpp
MLC_DenseLinearAlgebra.hpp
MLC_GreensFunction.hpp
MLC_GridManager.hpp
MLC_LocalCorrection.hpp
MLC_Mesh.hpp
MLC_MLC_Interp.hpp
MLC_ParticleInit.hpp
MLC_ProblemManager2.hpp
MLC_Remap.hpp
MLC_Solver2.hpp
MLC_Types.hpp
MLC_ConvolutionGPU_C2C.hpp
counters.hpp
  )

set(SOURCES
  MLC_Mesh.cpp
  )

install(FILES ${HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
  
add_library(exampm ${SOURCES})

target_link_libraries(exampm
  Cabana::Core
  Cabana::Grid
)

target_include_directories(exampm
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  )

install(TARGETS exampm LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
