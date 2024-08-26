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

 template <class ProblemManagerType, class ExecutionSpace, class GridManager>
 void Test_F( const ExecutionSpace& exec_space, const ProblemManagerType& pm,
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
   double pi = Kokkos::numbers::pi;
   double fx,fy,fz;
   double fx_e = 0.0,fy_e = 0.0,fz_e = 0.0;

   for(int i = 0; i < extent; i++)
     for(int j = 0; j < extent; j++)
        for(int k = 0; k < extent; k++)
	{

	    double x = i*h - center;
            double y = j*h - center;	    
	    double z = k*h - center;
            for(int d = 0; d < 3; d++)		
               velocity_g(i,j,k,d) = cos(2*pi*x)*sin(4*pi*y)*cos(4*pi*z);
            
	}

   for(int i = 1; i < extent-2; i++)
     for(int j = 1; j < extent-2; j++)
        for(int k = 1; k < extent-2; k++)
        {

            double x = i*h - center;
            double y = j*h - center;
            double z = k*h - center;

            MLC_Interp::f(velocity_g,i,j,k,0,g,fx,fy,fz);
            fx_e += abs(fx - (-2*pi*sin(2*pi*x)*sin(4*pi*y)*cos(4*pi*z) ) )*h;
            fy_e += abs(fy - ( 4*pi*cos(2*pi*x)*cos(4*pi*y)*cos(4*pi*z) ) )*h;
            fz_e += abs(fz - (-4*pi*cos(2*pi*x)*sin(4*pi*y)*sin(4*pi*z) ) )*h;
             
	    std::cout << " fx= " << fx << " fy= " << fy << "fz= " << fz << std::endl;
	    std::cout << " exact x = " << (-2*pi*sin(2*pi*x)*sin(4*pi*y)*cos(4*pi*z) )
		      << " exact y = " << ( 4*pi*cos(2*pi*x)*cos(4*pi*y)*cos(4*pi*z) )
		      << " exact z = " << (-4*pi*cos(2*pi*x)*sin(4*pi*y)*sin(4*pi*z) )
		      << std::endl;
	
	}

   std::cout << " x error = " << fx_e << " y error = " << fy_e << " z error = " << fz_e << std::endl;

}

 template <class ProblemManagerType, class ExecutionSpace, class GridManager>
 void Test_F2( const ExecutionSpace& exec_space, const ProblemManagerType& pm,
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
   double pi = Kokkos::numbers::pi;
   double fxx,fyy,fzz,fxy,fxz,fyz;
   double fxx_e = 0.0,fyy_e = 0.0,fzz_e = 0.0;
   double fxy_e = 0.0, fxz_e = 0.0, fyz_e = 0.0;

   for(int i = 0; i < extent; i++)
     for(int j = 0; j < extent; j++)
        for(int k = 0; k < extent; k++)
        {

            double x = i*h - center;
            double y = j*h - center;
            double z = k*h - center;
            for(int d = 0; d < 3; d++)
               velocity_g(i,j,k,d) = cos(2*pi*x)*sin(4*pi*y)*cos(4*pi*z);

        }

   for(int i = 1; i < extent-2; i++)
     for(int j = 1; j < extent-2; j++)
        for(int k = 1; k < extent-2; k++)
        {

            double x = i*h - center;
            double y = j*h - center;
            double z = k*h - center;

            MLC_Interp::f2(velocity_g,i,j,k,0,g,fxx,fyy,fzz,fxy,fxz,fyz);
	    fxx_e += abs(fxx - (-4*pi*pi*cos(2*pi*x)*sin(4*pi*y)*cos(4*pi*z) ) )*h;
            fyy_e += abs(fyy - (-16*pi*pi*cos(2*pi*x)*sin(4*pi*y)*cos(4*pi*z) ) )*h;
            fzz_e += abs(fzz - (-16*pi*pi*cos(2*pi*x)*sin(4*pi*y)*cos(4*pi*z) ) )*h;
            fxy_e += abs(fxy - (-8*pi*pi*sin(2*pi*x)*cos(4*pi*y)*cos(4*pi*z)  ) )*h;
	    fxz_e += abs(fxz - ( 8*pi*pi*sin(2*pi*x)*sin(4*pi*y)*sin(4*pi*z)  ) )*h;
	    fyz_e += abs(fyz - (-16*pi*pi*cos(2*pi*x)*cos(4*pi*y)*sin(4*pi*z) ) )*h;

  /*          std::cout << " fxx = " << fxx << " fyy = " << fyy << " fzz = " << fzz 
		      << " fxy = " << fxy << " fxz = " << fxz << " fyz = " << fyz << std::endl;

            std::cout << " exact xx = " << (-4*pi*pi*cos(2*pi*x)*sin(4*pi*y)*cos(4*pi*z) )
                      << " exact yy = " << (-16*pi*pi*cos(2*pi*x)*sin(4*pi*y)*cos(4*pi*z) )
		      << " exact zz = " << (-16*pi*pi*cos(2*pi*x)*sin(4*pi*y)*cos(4*pi*z) )
		      << " exact xy = " << (-8*pi*pi*sin(2*pi*x)*cos(4*pi*y)*cos(4*pi*z)  )
		      << " exact xz = " << (8*pi*pi*sin(2*pi*x)*sin(4*pi*y)*sin(4*pi*z)  )
		      << " exact yz = " << (-16*pi*pi*cos(2*pi*x)*cos(4*pi*y)*sin(4*pi*z) )
                      << std::endl; */



        }

   std::cout << " x error = " << fxx_e << " y error = " << fyy_e << " z error = " << fzz_e  
	     << " xy error = " << fxy_e << " xz error = " << fxz_e << " yz error = " << fyz_e << std::endl;

}

 template <class ProblemManagerType, class ExecutionSpace, class GridManager>
 void Test_MLC_Interp( const ExecutionSpace& exec_space, const ProblemManagerType& pm,
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
   double pi = Kokkos::numbers::pi;

   for(int i = 0; i < extent; i++)
     for(int j = 0; j < extent; j++)
        for(int k = 0; k < extent; k++)
        {

            double x = i*h - center;
            double y = j*h - center;
            double z = k*h - center;
            for(int d = 0; d < 3; d++)
               velocity_g(i,j,k,d) = cos(2*pi*x)*sin(4*pi*y)*cos(4*pi*z);

//	    std::cout << " velocity g = " << velocity_g(i,j,k,0) << std::endl;

        }

        double error = 0, exact;
        for(int p = 0; p < pm.numParticle(); p++)
        {

	    double u_temp[3];
	    double xp[3] = { positions(p,0), positions(p,1), positions(p,2) };
	    MLC_Interp::HarmonicValue( velocity_g, g, xp, u_temp );

	    for( int d = 0; d < 3; d++)
	    {	    
	       velocity_p(p,d) = u_temp[d];
	       exact = cos( 2*pi*xp[0] )*sin( 4*pi*xp[1] )*cos( 4*pi*xp[2] );

	       error += abs( exact - velocity_p(p,d) );

	       std::cout << " exact " << exact << " velocity = " << velocity_p(p,d) << std::endl;

            }

	    std::cout << " error = " << error << std::endl;

	    std::cout << " error by particle = " << error/(3*pm.numParticle() ) << std::endl;
	}


}
 template <class ProblemManagerType, class ExecutionSpace, class GridManager>
 void Test_F3( const ExecutionSpace& exec_space, const ProblemManagerType& pm,
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
   double pi = Kokkos::numbers::pi;
   double fxxy,fxxz,fyyx,fyyz,fzzx, fzzy, fxyz;
   double fxxy_e = 0.0,fxxz_e = 0.0,fyyx_e = 0.0;
   double fyyz_e = 0.0, fzzy_e = 0.0, fzzx_e = 0.0, fxyz_e = 0.0;

   for(int i = 0; i < extent; i++)
     for(int j = 0; j < extent; j++)
        for(int k = 0; k < extent; k++)
        {

            double x = i*h - center;
            double y = j*h - center;
            double z = k*h - center;
            for(int d = 0; d < 3; d++)
               velocity_g(i,j,k,d) = cos(2*pi*x)*sin(4*pi*y)*cos(4*pi*z);

        }

   for(int i = 1; i < extent-2; i++)
     for(int j = 1; j < extent-2; j++)
        for(int k = 1; k < extent-2; k++)
        {

            double x = i*h - center;
            double y = j*h - center;
            double z = k*h - center;

            MLC_Interp::f3(velocity_g,i,j,k,0,g,fxxy,fxxz, fyyx, fyyz, fzzx, fzzy, fxyz);
            fxxy_e += abs(fxxy - (-16*pi*pi*pi*cos(2*pi*x)*cos(4*pi*y)*cos(4*pi*z) ) )*h*h*h;
            fxxz_e += abs(fxxz - (16*pi*pi*pi*cos(2*pi*x)*sin(4*pi*y)*sin(4*pi*z) ) )*h*h*h;
            fyyx_e += abs(fyyx - (32*pi*pi*pi*sin(2*pi*x)*sin(4*pi*y)*cos(4*pi*z) ) )*h*h*h;
            fyyz_e += abs(fyyz - (64*pi*pi*pi*cos(2*pi*x)*sin(4*pi*y)*sin(4*pi*z)  ) )*h*h*h;
            fzzx_e += abs(fzzx - (32*pi*pi*pi*sin(2*pi*x)*sin(4*pi*y)*cos(4*pi*z)  ) )*h*h*h;
            fzzy_e += abs(fzzy - (-64*pi*pi*pi*cos(2*pi*x)*cos(4*pi*y)*cos(4*pi*z) ) )*h*h*h;
	    fxyz_e += abs(fxyz - (32*pi*pi*pi*sin(2*pi*x)*cos(4*pi*y)*sin(4*pi*z)  ) )*h*h*h;

 /*           std::cout << " fxxy = " << fxxy << " fxxz = " << fxxz << " fyyx = " << fyyx
                      << " fyyz = " << fyyz << " fzzx = " << fzzx << " fzzy = " << fzzy << " fxyz = " << fxyz << std::endl;

            std::cout << " exact xxy = " << (-16*pi*pi*pi*cos(2*pi*x)*cos(4*pi*y)*cos(4*pi*z) )
                      << " exact xxz = " << (16*pi*pi*pi*cos(2*pi*x)*sin(4*pi*y)*sin(4*pi*z) )
                      << " exact yyx = " << (32*pi*pi*pi*sin(2*pi*x)*sin(4*pi*y)*cos(4*pi*z) )
                      << " exact yyz = " << (64*pi*pi*pi*cos(2*pi*x)*sin(4*pi*y)*sin(4*pi*z)  )
                      << " exact zzx = " << (32*pi*pi*pi*sin(2*pi*x)*sin(4*pi*y)*cos(4*pi*z)  )
                      << " exact zzy = " << (-64*pi*pi*pi*cos(2*pi*x)*cos(4*pi*y)*cos(4*pi*z) )
		      << " exact xyz = " << (32*pi*pi*pi*sin(2*pi*x)*cos(4*pi*y)*sin(4*pi*z)  ) 
                      << std::endl;  */



        }

   std::cout << " xxy error = " << fxxy_e << " xxz error = " << fxxz_e << " yyx error = " << fyyx_e
             << " yyz error = " << fyyz_e << " zzx error = " << fzzx_e << " zzy error = " << fzzy_e << " xyz error = " << fxyz_e << std::endl;

}

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
 void Deposition( const ExecutionSpace& exec_space, const ProblemManagerType& pm, const NeighborListType& Ci_list,
                        const GridManager& gridp, const int num_grid, const int extent, const double center, const double h)
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
   auto F          = pm.get(Location::Node(), Field::F() );

   Kokkos::deep_copy( F, 0.0);
   Kokkos::deep_copy( velocity_g, 0.0);

   //Get relevant interpolation quantities 
   MLC_Interp::GridData<3> g( h, center);

   //Iterate over D0 
   Kokkos::parallel_for(
        "Depositions",
        Kokkos::RangePolicy<ExecutionSpace>( exec_space, 0, num_grid ),
        KOKKOS_LAMBDA( const int i ) {

            //D0 Grid Indices
            int ii = index(i,0);
            int jj = index(i,1);
            int kk = index(i,2);

            //ith grid positions
            double xi[3] = { gridx(i,0), gridx(i,1), gridx(i,2)};

            //Get Ci upper and lower bounds
            // getParticleBin(i) gives the cell/bin associated with the ith grid point
            // the max and min values of the stencil is built around this bin
            int imin, imax, jmin, jmax, kmin, kmax;
            Ci_list.getStencilCells( Ci_list.getParticleBin( i ), imin,imax, jmin,
                               jmax, kmin, kmax );

            //Iterate over cell stencil of the linked list = Ci
            for( int pi = imin; pi < imax; pi++)
                 for( int pj = jmin; pj < jmax; pj ++)
                      for( int pk = kmin; pk < kmax; pk ++)
                      {
                            //Get Offset and Size to determine # particles
                             auto Ci_offset = Ci_list.binOffset(pi,pj,pk);
                             auto Ci_size   = Ci_list.binSize(pi,pj,pk);
                             //Calculate jh
                             double xg[3] = { pi*h - center, pj*h - center, pk*h - center};

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
                                             velocity_g(pi, pj, pk, d) += K[d];
                                     }
                            }

                      }

         
	         // Calculate 2nd order Laplacian of each velocity component 
	         double F_temp[3] = {0.0, 0.0, 0.0};
                 MLC_Interp::L7(velocity_g,ii,jj,kk,g,F_temp);
                 
		 //Set F
		 for(int d = 0; d < 3; d++)
		    F(ii,jj,kk,d) += F_temp[d];
	     


	});

}

template <class ProblemManagerType, class ExecutionSpace, class NeighborListType, class GridManager>
 void TestConvolution( const ExecutionSpace& exec_space, const ProblemManagerType& pm, const NeighborListType& Ci_list,
                        const GridManager& gridp, const int num_grid, const int extent, const double center, const double h)
{

    auto velocity_g = pm.get(Location::Node(), Field::Velocity());
    auto positions  = pm.get(Location::Particle(), Field::Position());
    auto F          = pm.get(Location::Node(), Field::F() );
 
    Kokkos::deep_copy(velocity_g, 0.0);

    //Iterate over D	
    for(int i = 0; i < extent; i++)
       for(int j = 0; j < extent; j++)
          for( int k = 0; j < extent; k++)
          {

	      double xg[3] = { i*h - center, j*h - center, k*h - center };	  
	      //iterate over D0	  
              for(int i0 = 0; i0 < extent; i++)
                for(int j0 = 0; j0 < extent; j++)
                   for( int k = 0; j0 < extent; k++)
                   {

                         double x0[3] = { i0*h - center, j0*h - center, k0*h - center };
			 double r = pow( pow( x0[0]-xg[0], 2.0) + pow( x0[1]-xg[1], 2.0) + pow( x0[2]-xg[2], 2.0) );

			 if( r < pow(10, -9.0) )
		         {
		             for(int d = 0; d < 3; d++)		 
			        velocity_g(i,j,k,d) += 0.0;
                         }else{



                             for(int d = 0; d < 3; d++)
			        velocity_g(i,j,k,d) += F(i0,j0,k0,d)*1.0/(4*Kokkos::numbers::pi*r);


			 }
 
                    }




	  }


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

	     //Interpolate Particles where floor(xp/h) == i
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
                    MLC_Interp::HarmonicValue( velocity_g, g, xp, u_temp );

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
