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
 void Interactions( const ExecutionSpace& exec_space, const ProblemManagerType& pm,
                        const NeighborListType& neigh_list, const NeighborListType& oneGrid_list,
	                const GridManager& gridp,
		       	const int num_grid, const int extent, const double center, const double h)
{

   auto index = gridp.get();
   auto vorticity_p = pm.get(Location::Particle(), Field::Vorticity());
   auto velocity_p = pm.get(Location::Particle(), Field::Velocity());
   auto velocity_g = pm.get(Location::Node(), Field::Velocity());
   auto positions  = pm.get(Location::Particle(), Field::Position());

   Kokkos::deep_copy( velocity_g, 0.0);

   MLC_Interp::GridData<3> g( h, center);

   Kokkos::parallel_for(
        "Interactions",
	Kokkos::RangePolicy<ExecutionSpace>( exec_space, 60, 64 ),
	KOKKOS_LAMBDA( const int i ) {

	    int ii = index(i,0); 
	    int jj = index(i,1);
	    int kk = index(i,2);
            double xi[3] = { ii*h - center, jj*h - center, kk*h - center};
            auto Pi_offset = oneGrid_list.binOffset(ii,jj,kk);
            auto Pi_size   = oneGrid_list.binSize(ii,jj,kk);

            int imin, imax, jmin, jmax, kmin, kmax;
            neigh_list.getStencilCells( neigh_list.getParticleBin( Pi_offset), imin,imax, jmin,
			       jmax, kmin, kmax ); //neigh_list.getParticleBin( Pi_offset ), imin, imax, jmin,

	    std::cout << "num bin = " << neigh_list.numBin( 0 ) << " " << neigh_list.numBin( 1 ) << " " << neigh_list.numBin(2) << std::endl;

	    std::cout << " ii = " << ii << " jj = " << jj << " kk = " << kk << std::endl;
	    std::cout <<  " xi_1 = " << xi[0] << " xi_2 = " << xi[1] << " xi_3 = " << xi[2] << std::endl;
            std::cout << " min " << imin << " " << jmin << " " << kmin << " " << std::endl;
	    std::cout << " max " << imax << " " << jmax << " " << kmax << " " << std::endl;

	    //Iterate over Si
	    for(int si = ii-1; si <= ii+1; si++)
	       for(int sj = jj-1; sj <= jj+1; sj++)
	          for( int sk = kk-1; sk <= kk+1; sk++)
		  {

		     double xg[3] = { si*h - center, sj*h - center, sk*h - center};
//		     std::cout << " si = " << si << " sj = " << sj << " sk = " << sk << std::endl;
//                     std::cout <<  " xg_1 = " << xg[0] << " xg_2 = " << xg[1] << " xg_3 = " << xg[2] << std::endl;

		     int nm_p = 0;
		     int num_neighbors; 
		     //Iterate over cell stencil of the linked list = Ci
		     for( int pi = imin; pi < imax; pi++)
			for( int pj = jmin; pj < jmax; pj ++)
			   for( int pk = kmin; pk < kmax; pk ++)	
                           {
                                //Get Offset and Size to determine # particles
                                auto Ci_offset = neigh_list.binOffset(pi,pj,pk);
                                auto Ci_size   = neigh_list.binSize(pi,pj,pk);
                                nm_p += Ci_size;
                                for( std::size_t p = Ci_offset; p < Ci_offset+Ci_size; p++)
      		                {
                                     double vortp[3] = { vorticity_p(p,0), vorticity_p(p,1), vorticity_p(p,2) };
                                     double xp[3]    = { positions(p,0), positions(p,1), positions(p,2) };
                                     double K[3];
                                     
                                    // std::cout <<  " xp_1 = " << xp[0] << " xp_2 = " << xp[1] << " xp_3 = " << xp[2] << std::endl;
	         	             GreensFunction::CalculateK(xg, xp, vortp, K);

			             for(int d = 0; d < 3; d++)
                                          velocity_g(si, sj, sk, d) -= 1.0; //K[d];

		                }

			   }

		     auto offset = neigh_list.binOffset(ii,jj,kk);
		     std::cout << " num particles in Ci = " << nm_p << std::endl;
                     num_neighbors = Cabana::NeighborList<NeighborListType>::numNeighbor(neigh_list, offset);
		     std::cout << " num neighbors = " << num_neighbors << std::endl;
                     std::cout << " velocity si = " << velocity_g(si,sj,sk,0) << std::endl;
		  }

             //Now Interpolate the Local Corrections Velocity on Sj
	     //

	     

             std::cout << " Pi size = " << Pi_size << std::endl;
	     std::cout << "Pioffset = " << Pi_offset << std::endl;


	     for( std::size_t p = Pi_offset; p < Pi_offset+Pi_size; p++)
             {
             
	         double xp[3] = { positions(p, 0 ), positions( p, 1 ), positions( p, 2 ) };

                 // Update particle velocity.

                 double u_temp[3];
  
                 MLC_Interp::value( velocity_g, g,xp, u_temp );

                 for(int d = 0; d<3; d++)
                     velocity_p(p, d) = u_temp[d];

		 std::cout << "vg = " << velocity_g(ii,jj, kk, 0) << " vp = " << velocity_p(p,0) << " " << u_temp[0] <<std::endl;
		 std::cout <<  " xp_1 = " << xp[0] << " xp_2 = " << xp[1] << " xp_3 = " << xp[2] << std::endl;

             }


	});


}
 template <class ProblemManagerType, class ExecutionSpace, class NeighborListType>
 void Correction_NBody( const ExecutionSpace& exec_space, const ProblemManagerType& pm, 
		        const NeighborListType& neigh_list, const int c,
		       	const double center, const double cell_size )
{
	
    double x[3], xq[3];
    double K[3], vort[3]; 
    int i_g[3];
    auto u_p = pm.get( Location::Particle(), Field::Velocity() );
    auto x_p = pm.get( Location::Particle(), Field::Position() );
    auto vort_p = pm.get( Location::Particle(), Field::Vorticity() );

    // Build the local mesh.
    auto local_mesh = Cabana::Grid::createLocalMesh<ExecutionSpace>(
        *( pm.mesh()->localGrid() ) );

    auto local_nodes = pm.mesh()->localGrid()->indexSpace(
        Cabana::Grid::Ghost(), Cabana::Grid::Node(), Cabana::Grid::Local() );

    auto  Nbody_corr = KOKKOS_LAMBDA(const int p, const int  q){


	double x[3]  = { x_p( p, 0 ), x_p( p, 1 ), x_p( p, 2 ) };
        double xq[3] = { x_p( q, 0 ), x_p( q, 1 ), x_p( q, 2 ) };
        double vort[3]  = { vort_p( p, 0 ), vort_p( p, 1 ), vort_p( p, 2 ) };
        double K[3];

	       //Evaluate Green's Function
            GreensFunction::CalculateK(x, xq, vort, K);

	            for(int d = 0; d < 3; d++){
	              vort_p(p,d) += K[d];
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
