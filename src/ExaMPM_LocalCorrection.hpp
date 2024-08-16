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

#ifndef EXAMPM_LOCALCORRECTION_HPP
#define EXAMPM_LOCALCORRECTION_HPP

#include <ExaMPM_MLC_Interp.hpp>
#include <ExaMPM_ProblemManager2.hpp>
#include <Cabana_Grid.hpp>
#include <ExaMPM_GreensFunction.hpp>
#include <Kokkos_Core.hpp>
#include <cmath>
namespace ExaMPM
{
namespace LocalCorrection
{
//---------------------------------------------------------------------------//
// Particle-to-grid.
//
template <class ProblemManagerType, class ExecutionSpace>
void Interpolation( const ExecutionSpace& exec_space, const ProblemManagerType& pm, const int c, const double center, const double cell_size )
{

    // Get the particle data we need.
    // Vort_p is the vorticity vector
    auto u_p = pm.get( Location::Particle(), Field::Velocity() );
    auto x_p = pm.get( Location::Particle(), Field::Position() );
    auto ucorr_p = pm.get( Location::Particle(), Field::Velocity_Correction() );

    // Get the views we need.
    auto u_i = pm.get( Location::Node(), Field::Velocity() );
    auto ucorr_i = pm.get(Location::Node(), Field::Velocity_Correction() );
    //Kokkos::deep_copy(ucorr_i, u_i);
    Kokkos::deep_copy(ucorr_i, 2.0);
    // Build the local mesh.
    auto local_mesh = Cabana::Grid::createLocalMesh<ExecutionSpace>(
        *( pm.mesh()->localGrid() ) );

    //What does L2G do?
    auto l2g = Cabana::Grid::IndexConversion::createL2G(
        *( pm.mesh()->localGrid() ), Cabana::Grid::Node() );

    auto local_nodes = pm.mesh()->localGrid()->indexSpace(
        Cabana::Grid::Ghost(), Cabana::Grid::Node(), Cabana::Grid::Local() );

    MLC_Interp::GridData<3> g( cell_size, center);
//G2P particle interpolation:
   Kokkos::parallel_for(
        "Interpolation_Nbody",
        Kokkos::RangePolicy<ExecutionSpace>( exec_space, 0, pm.numParticle() ),
        KOKKOS_LAMBDA( const int p ) {

	    double xp[3] = { x_p( p, 0 ), x_p( p, 1 ), x_p( p, 2 ) };

            // Update particle velocity.

	    double u_temp[3];

	    MLC_Interp::value( ucorr_i, g,xp, u_temp );

            for(int d = 0; d<3; d++){
                ucorr_p(p, d) = u_temp[d]; 

            }


	});

}


 template <class ProblemManagerType, class ExecutionSpace, class NeighborListType>
 void Correction_NBody( const ExecutionSpace& exec_space, const ProblemManagerType& pm, 
		        const NeighborListType& neigh_list, const int c,
		       	const double center, const double cell_size )
{
    double x[3], xq[3];
    double K[3], u[3]; 
    int i_g[3];
    auto u_p = pm.get( Location::Particle(), Field::Velocity() );
    auto x_p = pm.get( Location::Particle(), Field::Position() );
    auto unbody_p = pm.get( Location::Particle(), Field::Velocity_Nbody() );
    Cabana::deep_copy( unbody_p, 0.0 ); 
        // Build the local mesh.
    auto local_mesh = Cabana::Grid::createLocalMesh<ExecutionSpace>(
        *( pm.mesh()->localGrid() ) );

    auto local_nodes = pm.mesh()->localGrid()->indexSpace(
        Cabana::Grid::Ghost(), Cabana::Grid::Node(), Cabana::Grid::Local() );

    auto  Nbody_corr = KOKKOS_LAMBDA(const int p, const int  q){


	double x[3]  = { x_p( p, 0 ), x_p( p, 1 ), x_p( p, 2 ) };
        double xq[3] = { x_p( q, 0 ), x_p( q, 1 ), x_p( q, 2 ) };
        double u[3]  = { u_p( p, 0 ), u_p( p, 1 ), u_p( p, 2 ) };
        double K[3];

	       //Evaluate Green's Function
            GreensFunction::CalculateK(x,xq,u, K);

	            for(int d = 0; d < 3; d++){
	              unbody_p(p,d) += 1; //K[d];
		    }
                
          };


	  Cabana::neighbor_parallel_for(Kokkos::RangePolicy<ExecutionSpace>( exec_space, 0, pm.numParticle() ), Nbody_corr, neigh_list, Cabana::FirstNeighborsTag(), Cabana::SerialOpTag(), "LocalCorrections" );

/*   for( int p = 0; p < pm.numParticle(); p++){
      int numNeighbor = Cabana::NeighborList<NeighborListType>::numNeighbor( neigh_list, p );
      std::cout << " num_neighbor = " << numNeighbor << std::endl;
      std::cout << "uNbody1 = " << unbody_p(p,0) << " uNbody2 = " << unbody_p(p,1) << " uNbody3 = " << unbody_p(p,2) << std::endl;
   }  
*/
}  
} // end namespace LocalCorrection
} // end namespace ExaMPM

#endif // EXAMPM_LocalCorrection_HPP
