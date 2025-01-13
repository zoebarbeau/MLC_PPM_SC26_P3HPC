/****************************************************************************
 * Copyright (c) 2018-2020 by the ExaMPM authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ExaMPM library. ExaMPM is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef EXAMPM_PARTICLEINIT_HPP
#define EXAMPM_PARTICLEINIT_HPP

#include <ExaMPM_Types.hpp>

#include <Cabana_Core.hpp>
#include <Cabana_Grid.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

namespace ExaMPM
{
//---------------------------------------------------------------------------//
// Filter out empty particles that weren't created.
template <class CreationView, class ParticleList, class ExecutionSpace>
void filterEmpties( const ExecutionSpace& exec_space,
                    const int local_num_create,
                    const CreationView& particle_created,
                    ParticleList& particles )
{
    using memory_space = typename CreationView::memory_space;

    // Determine the empty particle positions in the compaction zone.
    int num_particles = particles.size();
    Kokkos::View<int*, memory_space> empties(
        Kokkos::ViewAllocateWithoutInitializing( "empties" ),
        std::min( num_particles - local_num_create, local_num_create ) );
    Kokkos::parallel_scan(
        Kokkos::RangePolicy<ExecutionSpace>( exec_space, 0, local_num_create ),
        KOKKOS_LAMBDA( const int i, int& count, const bool final_pass ) {
            if ( !particle_created( i ) )
            {
                if ( final_pass )
                {
                    empties( count ) = i;
                }
                ++count;
            }
        } );


    // Compact the list so the it only has real particles.
    Kokkos::parallel_scan(
        Kokkos::RangePolicy<ExecutionSpace>( exec_space, local_num_create,
                                             num_particles ),
        KOKKOS_LAMBDA( const int i, int& count, const bool final_pass ) {
            if ( particle_created( i ) )
            {
                if ( final_pass )
                {
                    particles.setTuple( empties( count ),
                                        particles.getTuple( i ) );
                }
                ++count;
            }
        } );

    particles.resize( local_num_create );

    particles.shrinkToFit();

}

//---------------------------------------------------------------------------//
/*!
  \brief Initialize a uniform number of particles in each cell given an
  initialization functor.

  \tparam LocalGridType The local grid type to use for construction

  \tparam InitFunctor Initialization functor type. See the documentation below
  for the create_functor parameter on the signature of this functor.

  \tparam ParticleList A Cabana::AoSoA type for holding particles. The tuple
  type in this AoSoA is the particle type.

  \param local_grid The local grid to use for initialization. Particles will
  not be initialized in the halo - only in the owned cells.

  \param particles_per_cell_dim The number of particles to populate each cell
  dimension with.

  \param create_functor A functor which populates a particle given the
  positions of a particle. This functor returns true if a particle was created
  and false if it was not giving the signature:

      bool createFunctor( const double px[3],
                          typename ParticleList::tuple_type& particle );

  \param particles The Cabana AoSoA of particles to populate. This will be
  filled with particles and resized to a size equal to the number of particles
  created.
*/
template <class ExecSpace, class LocalGridType, class InitFunctor,
          class ParticleList>
void initializeParticles( const ExecSpace& exec_space,
                          const LocalGridType& local_grid,
                          const int particles_per_cell_dim,
                          const InitFunctor& create_functor,
                          ParticleList& particles,
	                  const double center,
	                  const double hp,
			  const double extent)
{
    // Kokkos memory space.
    using memory_space = typename ParticleList::memory_space;

    // Particle type.
    using particle_type = typename ParticleList::tuple_type;

    // Create a local mesh.
    auto local_mesh = Cabana::Grid::createLocalMesh<memory_space>( local_grid );

    // Get the local set of owned cell indices.
    auto owned_cells = local_grid.indexSpace(
        Cabana::Grid::Ghost(), Cabana::Grid::Node(), Cabana::Grid::Local() );

    // Allocate enough space for the case the particles consume the entire
    // local grid.
    int particles_per_cell = particles_per_cell_dim * particles_per_cell_dim *
                             particles_per_cell_dim;
    int num_particles = particles_per_cell * owned_cells.size();
    particles.resize( num_particles );

    // Creation status.
    auto particle_created = Kokkos::View<bool*, memory_space>(
        Kokkos::ViewAllocateWithoutInitializing( "particle_created" ),
        num_particles );

    std::cout << " hp particle initialize " << hp << std::endl;
    // Initialize particles.
    int local_num_create = 0;
   // Kokkos::parallel_reduce(
   //     "init_particles_uniform",
   //     Cabana::Grid::createExecutionPolicy( owned_cells, exec_space ),
   //     KOKKOS_LAMBDA( const int i, const int j, const int k,
   //                    int& create_count ) {
    Cabana::Grid::grid_parallel_reduce(
        "uniform grid", exec_space, local_grid, Cabana::Grid::Ghost(),
        Cabana::Grid::Node(),
        KOKKOS_LAMBDA( const int i, const int j, const int k, int& create_count)
        {
            // Compute the owned local cell id.
            int i_own = i; // - owned_cells.min( Dim::I );
            int j_own = j; //- owned_cells.min( Dim::J );
            int k_own = k; //- owned_cells.min( Dim::K );
            int cell_id = i + (extent+1)*( j + k*(extent+1));


            // Particle.
	   double px[3];
            particle_type particle;
           int pid = cell_id; // * particles_per_cell + ip +
           // Set the particle position.
           px[0] = i_own * hp - center; //0.5 * spacing[Dim::I] +
                                     //ip * spacing[Dim::I] + low_coords[Dim::I];
           px[1] = j_own * hp - center; //0.5 * spacing[Dim::J] +
                                     //jp * spacing[Dim::J] + low_coords[Dim::J];
           px[2] = k_own * hp - center; //0.5 * spacing[Dim::K] +
                                     //kp * spacing[Dim::K] + low_coords[Dim::K];

                   
           // Create a new particle.
           particle_created( pid ) =
           create_functor( px, particle );

           // If we created a new particle insert it into the list.
           if ( particle_created( pid ) )
           {
                            particles.setTuple( pid, particle );
                            ++create_count;
           }
                 //   }
        },
        local_num_create );

    // Filter empties.
    filterEmpties( exec_space, local_num_create, particle_created, particles );
}

//---------------------------------------------------------------------------//
template <class ExecutionSpace, class LocalGridType, class InitFunctor,
          class ParticleList, class GridArrayType>
void remapParticles( const ExecutionSpace& exec_space,
                          const LocalGridType& local_grid,
                          const int particles_per_cell_dim,
                          const InitFunctor& create_functor,
			  const GridArrayType vort,
                          ParticleList& particles,
			  const double center,
			  const double hp,
			  const double extent)
	                  
{
    // Kokkos memory space.
    using memory_space = typename ParticleList::memory_space;

    // Particle type.
    // Particle type.
    using particle_type = typename ParticleList::tuple_type;

    auto vorticity = vort->view();
    // Create a local mesh.
    auto local_mesh = Cabana::Grid::createLocalMesh<memory_space>( local_grid );

    // Get the local set of owned cell indices.
    auto owned_cells = local_grid.indexSpace(
        Cabana::Grid::Ghost(), Cabana::Grid::Node(), Cabana::Grid::Local() );


    // Allocate enough space for the case the particles consume the entire
    // local grid.
    int particles_per_cell = particles_per_cell_dim * particles_per_cell_dim *
                             particles_per_cell_dim;
    int num_particles = particles_per_cell * owned_cells.size();
    particles.resize( num_particles );

    // Creation status.
    auto particle_created = Kokkos::View<bool*, memory_space>(
        Kokkos::ViewAllocateWithoutInitializing( "particle_created" ),
        num_particles );

    // Initialize particles.
    int local_num_create = 0;
    Kokkos::parallel_reduce(
        "init_particles_uniform",
        Cabana::Grid::createExecutionPolicy( owned_cells, exec_space ),
        KOKKOS_LAMBDA( const int i, const int j, const int k,
                       int& create_count ) {
            // Compute the owned local cell id.
	    int i_own = i; // - owned_cells.min( Dim::I );
            int j_own = j; //- owned_cells.min( Dim::J );
            int k_own = k; //- owned_cells.min( Dim::K );
            int cell_id = i + (extent+1)*( j + k*(extent+1));

            // Particle coordinate.
            double px[3];
	    double vort[3] ={ vorticity(i_own,j_own,k_own,0), vorticity(i_own,j_own,k_own,1), vorticity(i_own,j_own,k_own,2) };


            // Particle.
            particle_type particle;

            // Local particle id.
            int pid = cell_id; //* particles_per_cell + ip +
                                 // particles_per_cell_dim *
                                 //     ( jp + particles_per_cell_dim * kp );

                        // Set the particle position.
            px[0] = i_own*hp - center; // 0.5 * spacing[Dim::I] +
                                     //ip * spacing[Dim::I] + low_coords[Dim::I];
            px[1] = j_own*hp - center; //0.5 * spacing[Dim::J] +
                                   //  jp * spacing[Dim::J] + low_coords[Dim::J];
            px[2] = k_own*hp - center; //0.5 * spacing[Dim::K] +
  
			//  kp * spacing[Dim::K] + low_coords[Dim::K];

			if( vorticity(i_own,j_own,k_own,0) > 0)
			{


			} 
             // Create a new particle.
             particle_created( pid ) = create_functor( px, vort, particle );
                         
             // If we created a new particle insert it into the list.
             if ( particle_created( pid ) )
             {   
                 particles.setTuple( pid, particle );
                 ++create_count;
              }
                  //  }
        },
        local_num_create );


    filterEmpties( exec_space, local_num_create, particle_created, particles );
//    std::cout << "filter empties" << std::endl;
}
}
#endif // end EXAMPM_PARTICLEINIT_HPP
