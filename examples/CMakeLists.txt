add_executable( DamBreak dam_break.cpp )
target_link_libraries( DamBreak PRIVATE exampm)

add_executable( FreeFall free_fall.cpp )
target_link_libraries( FreeFall PRIVATE exampm)

add_executable( initGrid init_grid.cpp )
target_link_libraries( initGrid PRIVATE exampm)

install(TARGETS DamBreak FreeFall initGrid DESTINATION ${CMAKE_INSTALL_BINDIR})
