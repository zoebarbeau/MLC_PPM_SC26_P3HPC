add_executable( initGrid init_grid.cpp )
target_link_libraries( initGrid PRIVATE exampm)

add_executable( Nbody Nbody.cpp )
target_link_libraries( Nbody PRIVATE exampm)

add_executable( NbodyGrid NbodyGrid.cpp )
target_link_libraries( NbodyGrid PRIVATE exampm)

install(TARGETS initGrid DESTINATION ${CMAKE_INSTALL_BINDIR})
