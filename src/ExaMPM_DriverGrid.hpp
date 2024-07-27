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

#ifndef EXAMPM_DRIVERGRID_HPP
#define EXAMPM_DRIVERGRID_HPP

#include <ExaMPM_ProblemManager2.hpp>
#include <ExaMPM_VInterpolation.hpp>

#include <Cabana_Grid.hpp>

#include <Kokkos_Core.hpp>

#include <cmath>

namespace ExaMPM
{
namespace DriverGrid
{
//---------------------------------------------------------------------------//
// Particle-to-grid.
template <class ProblemManagerType, class ExecutionSpace>
void p2g( const ExecutionSpace& exec_space, const ProblemManagerType& pm )
{
    // Get the particle data we need.
    // Vort_p is the vorticity vector
    auto vort_p = pm.get( Location::Particle(), Field::Vorticity() );

    //strength p is a scalar field of ones for testing
    auto strength_p = pm.get( Location::Particle(), Field::Vortx() );
    auto x_p = pm.get( Location::Particle(), Field::Position() );

    // Get the views we need.
    auto vort_i = pm.get( Location::Node(), Field::Vorticity() );
    auto strength_i = pm.get( Location::Node(), Field::Vortx() );

    // Reset write views.
    Kokkos::deep_copy( vort_i, 0.0 );
    Kokkos::deep_copy( strength_i, 0.0 );

    // Create the scatter views we need.
    auto vort_i_sv = Kokkos::Experimental::create_scatter_view( vort_i );
    auto strength_i_sv = Kokkos::Experimental::create_scatter_view( strength_i );

    // Build the local mesh.
    auto local_mesh = Cabana::Grid::createLocalMesh<ExecutionSpace>(
        *( pm.mesh()->localGrid() ) );

    // Loop over particles.
    Kokkos::parallel_for(
        "p2g",
        Kokkos::RangePolicy<ExecutionSpace>( exec_space, 0, pm.numParticle() ),
        KOKKOS_LAMBDA( const int p ) {
            // Get the particle position.
            double x[3] = { x_p( p, 0 ), x_p( p, 1 ), x_p( p, 2 ) };

            // Setup interpolation to the nodes.
            Cabana::Grid::SplineData<double, 2, 3, Cabana::Grid::Node> sd;
            Cabana::Grid::evaluateSpline( local_mesh, x, sd );

	    double vort_p2[3] = { vort_p( p, 0 ), vort_p( p, 1 ), vort_p( p, 2 ) };

	    Cabana::Grid::P2G::value( strength_p( p ), sd, strength_i_sv );
	    Cabana::Grid::P2G::value( vort_p2, sd, vort_i_sv );
        } );

  // Complete local scatter.
  Kokkos::Experimental::contribute( vort_i, vort_i_sv );
  Kokkos::Experimental::contribute( strength_i, strength_i_sv );

 // Complete global scatter. Not sure if needed--uses ScatterReduce::Sum 
pm.scatter( Location::Node() );
}

//---------------------------------------------------------------------------//
// Grid-to-particle.
template <class ProblemManagerType, class ExecutionSpace>
void g2p( const ExecutionSpace& exec_space, const ProblemManagerType& pm )
{
    // Get the particle data we need.
    // Vort_p is the 3D vector
    auto vort_p = pm.get( Location::Particle(), Field::Vorticity() );

    //strength p is a scalar field of ones for testing purposes 
    auto strength_p = pm.get( Location::Particle(), Field::Vortx() );

    //particles postion
    auto x_p = pm.get( Location::Particle(), Field::Position() );

    // Get the views we need.
    auto vort_i = pm.get( Location::Node(), Field::Vorticity() );
    auto strength_i = pm.get( Location::Node(), Field::Vortx() );

    // Build the local mesh.
    auto local_mesh = Cabana::Grid::createLocalMesh<ExecutionSpace>(
        *( pm.mesh()->localGrid() ) );

    // Gather the data we need.
    pm.gather( Location::Node() );

    // Loop over particles.
    Kokkos::parallel_for(
        "g2p",
        Kokkos::RangePolicy<ExecutionSpace>( exec_space, 0, pm.numParticle() ),
        KOKKOS_LAMBDA( const int p ) {
            // Get the particle position.
            double x[3] = { x_p( p, 0 ), x_p( p, 1 ), x_p( p, 2 ) };

            // Setup interpolation from the nodes.
            Cabana::Grid::SplineData<double, 2, 3, Cabana::Grid::Node> sd_i;
            Cabana::Grid::evaluateSpline( local_mesh, x, sd_i );

            // Update particle velocity.
            double vort_temp[3], vort_old[3];
	    double strength_temp = 0; 

	    Cabana::Grid::G2P::value( strength_i, sd_i, strength_temp );
	    Cabana::Grid::G2P::value( vort_i, sd_i, vort_temp );

	    strength_p( p ) = strength_temp;
            for(int d = 0; d<3; ++d){
                vort_p(p, d) = vort_temp[d];

	    }
        } );

}

//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
// Take a time step.
template <class ProblemManagerType, class ExecutionSpace>
void interpVort( const ExecutionSpace& exec_space, const ProblemManagerType& pm,
           const BoundaryCondition& bc )
{
    p2g( exec_space, pm );
    //Grid Output
//    Cabana::Grid::Experimental::BovWriter::writeTimeStep("grid",0,0.0,vort_i);
    g2p( exec_space, pm );
}

//---------------------------------------------------------------------------//

} // end namespace DriverGrid
} // end namespace ExaMPM

#endif // EXAMPM_DriverGrid_HPP
