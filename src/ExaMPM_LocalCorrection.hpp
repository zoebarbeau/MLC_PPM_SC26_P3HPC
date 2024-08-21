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
#include <ExaMPM_GridManager.hpp>
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

 template <class ProblemManagerType, class ExecutionSpace, class NeighborListType, class GridManager>
 void Corrections( const ExecutionSpace& exec_space, const ProblemManagerType& pm,
                        const NeighborListType& Ci_list, const NeighborListType& Pi_list,
                        const GridManager& gridp,
                        const int num_grid, const int extent, const double center, const double h)
{

   //Gridp is the fake grid particle list, get positions and ids	
   auto index = gridp.get(Grid::Index());
   auto gridx = gridp.get(Grid::Position());
   auto id    = gridp.get(Grid::Id());

   //Get vorticity, velocity, and positions of real particles
   auto vorticity_p = pm.get(Location::Particle(), Field::Vorticity());
   auto velocity_p = pm.get(Location::Particle(), Field::Velocity());
   auto velocity_g = pm.get(Location::Node(), Field::Velocity());
   auto positions  = pm.get(Location::Particle(), Field::Position());

   Kokkos::deep_copy( velocity_g, 0.0);

   //Get relevant interpolation quantities 
   MLC_Interp::GridData<3> g( h, center);

   //Iterate over D0 
   Kokkos::parallel_for(
        "Corrections",
        Kokkos::RangePolicy<ExecutionSpace>( exec_space, 0, num_grid ),
        KOKKOS_LAMBDA( const int i ) {

	    //D0 Grid Indices
            int ii = index(i,0);
            int jj = index(i,1);
            int kk = index(i,2);

	    //ith grid positions
            double xi[3] = { gridx(i,0), gridx(i,1), gridx(i,2)};

	    //Offsets for particles by the grid cell
            auto Pi_offset = Pi_list.binOffset(ii,jj,kk);
            auto Pi_size   = Pi_list.binSize(ii,jj,kk);


	    //Get Ci upper and lower bounds
	    // getParticleBin(i) gives the cell/bin associated with the ith grid point
	    // the max and min values of the stencil is built around this bin
            int imin, imax, jmin, jmax, kmin, kmax;
            Ci_list.getStencilCells( Ci_list.getParticleBin( i ), imin,imax, jmin,
                               jmax, kmin, kmax ); 

	    //Iterate over Si 
	    for( int si = ii-1; si <= ii+1; si++)
               for(int sj = jj-1; sj <= jj+1; sj++)
                  for( int sk = kk-1; sk <= kk+1; sk++)
                  { 

		     //Calculate jh
                     double xg[3] = { si*h - center, sj*h - center, sk*h - center};

                     //Iterate over cell stencil of the linked list = Ci
                     for( int pi = imin; pi < imax; pi++)
                        for( int pj = jmin; pj < jmax; pj ++)
                           for( int pk = kmin; pk < kmax; pk ++)
                           {
                                //Get Offset and Size to determine # particles
                                auto Ci_offset = Ci_list.binOffset(pi,pj,pk);
                                auto Ci_size   = Ci_list.binSize(pi,pj,pk);

				//Loop over Ci
                                for( std::size_t r = Ci_offset; r < Ci_offset+Ci_size; r++)
                                {

				     //Get true particle ID in fake particle list	
                                     auto j = Ci_list.getParticle( r );

                                     //Check that it is a real particle vs fake				     
				     if( id(j) == 1 ){


					 //Get Real Particle ID   
					 int p = j - num_grid;   

					 // Get Vorticity and Position 
                                         double vortp[3] = { vorticity_p(p,0), vorticity_p(p,1), vorticity_p(p,2) };
                                         double xp[3]    = { positions(p,0), positions(p,1), positions(p,2) };
                                         double K[3];

					 //Calculate Green's Function
                                         GreensFunction::CalculateK(xg, xp, vortp, K);

					 //Correct Velocity
                                         for(int d = 0; d < 3; d++)
                                             velocity_g(si, sj, sk, d) -= K[d]; 
                                     } 
                                }

                           }

                }


	     //Offsets from linked cell list describing floor(xp/h) = i
             auto offset = Pi_list.binOffset(ii,jj,kk);
             auto size   = Pi_list.binSize(ii,jj,kk);

             for( int r = offset; r < offset+size; r++)
             {

		 auto j = Pi_list.getParticle( r );   

		 //Check for Real Particle vs. Grid Particle 
		 if( id(j) == 1 ){
	            int p = j - num_grid;		 
                    double xp[3] = { positions(p, 0 ), positions( p, 1 ), positions( p, 2 ) };

                    // Update particle velocity.
                    double u_temp[3];

		    //Interpolate from Grid to Particle
		    // g contains cell size information
                    MLC_Interp::value( velocity_g, g, xp, u_temp );

		    //Update RHS
                    for(int d = 0; d<3; d++)
                       velocity_p(p, d) = u_temp[d];

		 }
             }


        });


}

 template <class ProblemManagerType, class ExecutionSpace, class NeighborListType>
 void Interaction_NBody( const ExecutionSpace& exec_space, const ProblemManagerType& pm, 
		        const NeighborListType& neigh_list, const int c,
		       	const double center, const double cell_size )
{
	
    double x[3], xq[3];
    double K[3], vort[3]; 

    //Get vorticity, postion and velocity on particles
    auto u_p = pm.get( Location::Particle(), Field::Velocity() );
    auto x_p = pm.get( Location::Particle(), Field::Position() );
    auto vort_p = pm.get( Location::Particle(), Field::Vorticity() );

    // p is the particle of interest and q is the neighbor particle
    auto  interaction = KOKKOS_LAMBDA(const int p, const int  q){


	// Particle P Location
	double x[3]  = { x_p( p, 0 ), x_p( p, 1 ), x_p( p, 2 ) };

	//Particle Q (Neighbor) Position
        double xq[3] = { x_p( q, 0 ), x_p( q, 1 ), x_p( q, 2 ) };
        double vort[3]  = { vort_p( q, 0 ), vort_p( q, 1 ), vort_p( q, 2 ) };
        double K[3];

        //Evaluate Green's Function with number
        GreensFunction::CalculateK(x, xq, vort, K);


	//Correct Velocity at P with Local Neighbor Interaction at Q
        for(int d = 0; d < 3; d++){
              u_p(p,d) += K[d];
        }
                
     };

          //Find neighbors and calculate interaction for all particles
	  Cabana::neighbor_parallel_for(Kokkos::RangePolicy<ExecutionSpace>( exec_space, 0, pm.numParticle() ), interaction, neigh_list, Cabana::FirstNeighborsTag(), Cabana::SerialOpTag(), "LocalCorrections" );

}  
} // end namespace LocalCorrection
} // end namespace ExaMPM

#endif // EXAMPM_LocalCorrection_HPP
