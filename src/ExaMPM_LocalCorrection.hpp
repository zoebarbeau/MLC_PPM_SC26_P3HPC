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


// Get Vorticity and Position
void test_greens( )
{
       double vortp[3] = { 1, 1, -1 };
       double xq[3]    = { 0,0,0 };
       double xp[3]    = { 0.5, 0.5, 0.5};
       double K[3];
       //Calculate Green's Function
       GreensFunction::Calculate_qK(xp, xq, vortp, K);

       std::cout << " K = " << K[0] << "  " << K[1] << " " << K[2] << std::endl;

}
 template <class ProblemManagerType, class ExecutionSpace, class GridManager>
 void Test_L27( const ExecutionSpace& exec_space, const ProblemManagerType& pm,
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
   double L[3], Error[3]={0.0,0.0,0.0};
   double L_exact;

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

   double errormax = 0;
   for(int i = 1; i < extent-1 ; i++)
     for(int j = 1; j < extent-1; j++)
        for(int k = 1; k < extent-1; k++)
        {

            double x = i*h - center;
            double y = j*h - center;
            double z = k*h - center;

            MLC_Interp::L27(velocity_g,i,j,k,g,L);
            L_exact = (-4*pi*pi*cos(2*pi*x)*sin(4*pi*y)*cos(4*pi*z) ) + (-16*pi*pi*cos(2*pi*x)*sin(4*pi*y)*cos(4*pi*z) )
	              + (-16*pi*pi*cos(2*pi*x)*sin(4*pi*y)*cos(4*pi*z) ); 

	    for(int d = 0; d < 3; d++)
	       Error[d] += (std::abs(L[d] - L_exact))*h*h*h;
        // std::cout << " EXACT= " << L_exact << " Real = " << L[0] << std::endl;
        // std::cout << " i = " << i << " j = " << j << " k = " << k << std::endl;

	   if( errormax < std::abs(L[0] - L_exact) )
           {

	      errormax = std::abs(L[0] - L_exact);
	   }		   


        }

   std::cout << " laplacian error = " << Error[0] << " " << Error[1] << " " << Error[2] << std::endl;
   std::cout << " error max = " << errormax << std::endl;
}
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
	    fxx_e += std::abs(fxx - (-4*pi*pi*cos(2*pi*x)*sin(4*pi*y)*cos(4*pi*z) ) )*h;
            fyy_e += std::abs(fyy - (-16*pi*pi*cos(2*pi*x)*sin(4*pi*y)*cos(4*pi*z) ) )*h;
            fzz_e += std::abs(fzz - (-16*pi*pi*cos(2*pi*x)*sin(4*pi*y)*cos(4*pi*z) ) )*h;
            fxy_e += std::abs(fxy - (-8*pi*pi*sin(2*pi*x)*cos(4*pi*y)*cos(4*pi*z)  ) )*h;
	    fxz_e += std::abs(fxz - ( 8*pi*pi*sin(2*pi*x)*sin(4*pi*y)*sin(4*pi*z)  ) )*h;
	    fyz_e += std::abs(fyz - (-16*pi*pi*cos(2*pi*x)*cos(4*pi*y)*sin(4*pi*z) ) )*h;

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

template <class ExecutionSpace, class ProblemManagerType, class GridManagerType, class LocalGridType>
void update_GridList(const ExecutionSpace& exec_space, const LocalGridType& cgrid,
                       const ProblemManagerType& pm, const GridManagerType& gridp, const int num_p
                      ,const int num_D0, const int extent, const double h, const double center)
{
        auto index = gridp.get( Grid::Index());
        auto positions = gridp.get( Grid::Position());
        auto id        = gridp.get( Grid::Id());
	auto x         = pm.get(Location::Particle(), Field::Position() );
        //Add fake grid particles and their positions
        // The fake particles are the grid D0

       Cabana::Grid::grid_parallel_for(
        "find numParticles", exec_space, cgrid, Cabana::Grid::Ghost(),
        Cabana::Grid::Node(),
        KOKKOS_LAMBDA( const int i, const int j, const int k )
        {

	            if( (i > 0 && j > 0 && k > 0) && ( i < extent && j < extent && k < extent) )
		    {
                       int particle_counter = (i-1) + (extent-1)*((j-1) + (k-1)*(extent-1));
                       index( particle_counter, 0 ) = i;
                       index( particle_counter, 1 ) = j;
                       index( particle_counter, 2 ) = k;

                       positions( particle_counter, 0) = i*h-center;
                       positions( particle_counter, 1) = j*h-center;
                       positions( particle_counter, 2) = k*h-center;

                       //Fake particle denoted by an ID of 100
                       id( particle_counter ) =  100;
		    }
         });

        //Add real particles and their positions

        std::cout << " add real particles" << std::endl;

        Kokkos::parallel_for(
        "add_real_particles",
        Kokkos::RangePolicy<ExecutionSpace>( exec_space, num_D0, num_D0+num_p ),
        KOKKOS_LAMBDA( const int p )
        {


                int it = p - num_D0;
                double xp[3] = { x(it,0), x(it,1), x(it,2) };
                int i = floor( (xp[0] + center) / h);
                int j = floor( (xp[1] + center) / h);
                int k = floor( (xp[2] + center) / h);
               
                index( p, 0 ) = i;
                index( p, 1 ) = j;
                index( p, 2 ) = k;

                positions( p, 0) = xp[0];
                positions( p, 1) = xp[1];
                positions( p, 2) = xp[2];
                Kokkos::printf( " p %d x %f y %f z %f", p, xp[0], xp[1], xp[2] );
                //Real particle denoted by an ID of 1
                id( p ) = 1;
        });
}

template <class ProblemManagerType, class ExecutionSpace, class NeighborListType, class GridManager>
 void Deposition( const ExecutionSpace& exec_space, const ProblemManagerType& pm, const NeighborListType& Pi_list,
                  const NeighborListType& Ci_list, const GridManager& gridp, const int num_grid, 
		  const int extent, const double center, const double h)
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
   auto velx       = pm.get(Location::Node(), Field::velx() );
   auto Fx         = pm.get(Location::Node(), Field::Fx() );

   Kokkos::deep_copy( velx, 0.0);
   Kokkos::deep_copy( F, 0.0);
   Kokkos::deep_copy( velocity_g, 0.0);
   Kokkos::deep_copy(Fx, 0.0);
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

            // getParticleBin(i) gives the cell/bin associated with the ith grid point
            // the max and min values of the stencil is built around this bin
            int imin, imax, jmin, jmax, kmin, kmax;
            Ci_list.getStencilCells( Ci_list.getParticleBin( i ), imin,imax, jmin,
                               jmax, kmin, kmax );



   //Reset Ci to 0
            for( int ci = imin; ci < imax; ci++)
                for( int cj = jmin; cj < jmax; cj ++)
                   for( int ck = kmin; ck < kmax; ck ++)
                   {
                          for(int d = 0; d < 3; d++)
                              velocity_g(ci,cj,ck,d) = 0.0;

                   }
    
	    auto offset = Pi_list.binOffset(ii,jj,kk);
            auto size   = Pi_list.binSize(ii,jj,kk);
	    // Iterate over Ci
            for( int ci = imin; ci < imax; ci++)
                 for( int cj = jmin; cj < jmax; cj ++)
                      for( int ck = kmin; ck < kmax; ck ++)
                      {
			  //Calculate jh
                          double xg[3] = { ci*h - center, cj*h - center, ck*h - center};    

                          //Loop over Ci
                          for( std::size_t r = offset; r < offset+size; r++)
                          {

                              //Get true particle ID in fake particle list       
                              auto j = Pi_list.getParticle( r );

			      //Check that it is a real particle vs fake
                              if( id(j) == 1 ){

                                   //Get Real Particle ID
                                   int p = j - num_grid;
 
                                   // Get Vorticity and Position
                                   double vortp[3] = { vorticity_p(p,0), vorticity_p(p,1), vorticity_p(p,2) };
                                   double xp[3]    = { positions(p,0), positions(p,1), positions(p,2) };
                                   double K[3];

                                   //Calculate Green's Function
                                   GreensFunction::Calculate_qK(xg, xp, vortp, K);

                                   //Correct Velocity
                                   for(int d = 0; d < 3; d++)
                                        velocity_g(ci, cj, ck, d) += K[d];
                                   velx(ci,cj,ck,0) +=K[0];
                                        if(vortp[0] > 0){
                 //                          Kokkos::printf(" vortp %f velocity %f \n",vortp[0], velocity_g(ci,cj,ck,0) );
                                        }

                                  }
                          }

                       

                  }

	         for( int c0i = imin+1; c0i < imax-1; c0i++)
                   for( int c0j = jmin+1; c0j < jmax-1; c0j++)
                      for( int c0k = kmin+1; c0k < kmax-1; c0k++)
                      {

	                 // Calculate 2nd order Laplacian of each velocity component 
	                 double F_temp[3] = {0.0, 0.0, 0.0};
                         MLC_Interp::L27(velocity_g,c0i, c0j, c0k,g,F_temp);
                  
		    
                         //Set F
   		         for(int d = 0; d < 3; d++)
		            F(c0i,c0j,c0k,d) += F_temp[d];
                         Fx(c0i,c0j,c0k,0) += F_temp[0];
			 
		      }



	});

             pm.save_v( "Deposition_V",1,0);
             pm.save_F( "Initial_F",1,0);
}

template <class ProblemManagerType, class ExecutionSpace, class NeighborListType, class GridManager>
 void TestConvolution( const ExecutionSpace& exec_space, const ProblemManagerType& pm, const NeighborListType& Ci_list,
                        const GridManager& gridp, const int num_grid, const int extent, const double center, const double h)
{

    auto velocity_g = pm.get(Location::Node(), Field::Velocity());
    auto positions  = pm.get(Location::Particle(), Field::Position());
    auto F          = pm.get(Location::Node(), Field::F() );
    auto velx       = pm.get(Location::Node(), Field::velx() );
    auto Fx         = pm.get(Location::Node(), Field::Fx() );
    Kokkos::deep_copy(velocity_g, 0.0);
    //Get relevant interpolation quantities 
    MLC_Interp::GridData<3> g( h, center);

    //Iterate over D	
    for(int i = 0; i < extent+1; i++)
       for(int j = 0; j < extent+1; j++)
          for( int k = 0; k < extent+1; k++)
          { 

	      double xg[3] = { i*h - center, j*h - center, k*h - center };	  
	      //iterate over D0	  
              for(int i0 = 1; i0 < extent; i0++)
                for(int j0 = 1; j0 <= extent; j0++)
                   for( int k0 = 1; k0 <= extent; k0++)
                   {

                         double x0[3] = { i0*h - center, j0*h - center, k0*h - center };
			 double r = pow( pow( x0[0]-xg[0], 2.0) + pow( x0[1]-xg[1], 2.0) + pow( x0[2]-xg[2], 2.0), 0.5 );

			 if( r < pow(10, -8.0) )
		         {
		             for(int d = 0; d < 3; d++)		 
			        velocity_g(i,j,k,d) += 0.0;
                         }else{

                             for(int d = 0; d < 3; d++)
			        velocity_g(i,j,k,d) -= F(i0,j0,k0,d)*1.0/(4.0*Kokkos::numbers::pi*r);
                                 
			       if ( std::abs( F(i0,j0,k0,0)*1.0/(4.0*Kokkos::numbers::pi*r) ) > 0 
			            || std::abs( F(i0,j0,k0,1)*1.0/(4.0*Kokkos::numbers::pi*r) ) > 0
				    || std::abs( F(i0,j0,k0,2)*1.0/(4.0*Kokkos::numbers::pi*r) ) > 0  ){
		/*		std::cout << "r " << r << std::endl;
			        std::cout << " G " << 1.0/(4.0*Kokkos::numbers::pi*r) << std::endl;
                                std::cout << " F" << F(i0,j0,k0,0) << std::endl;
				std::cout << " add " << F(i0,j0,k0,0)*1.0/(4.0*Kokkos::numbers::pi*r) << std::endl;
				std::cout << "velocity = " << velocity_g(i,j,k,0) << std::endl; */
                                } 
			 }
 
                    }

           }


    //Print out Velocity
      for( int i = 1; i < extent; i++)
          for( int j = 1; j < extent; j++)
             for( int k = 1; k < extent; k++)
             {

                double xgg[3] = {i*h - center, j*h - center, k*h - center };
		double F_temp[3] = {0.0, 0.0, 0.0};
                MLC_Interp::L27(velocity_g,i,j,k,g,F_temp);
               velx(i,j,k,0) = velocity_g(i,j,k,0);
	       Fx(i,j,k,0) = F_temp[0];   

             }

              pm.save_v( "Convolution_V",1,0);
             pm.save_F( "Laplacian_V",1,0);

}
 template <class ProblemManagerType, class ExecutionSpace, class NeighborListType, class GridManager>
 void Corrections( const ExecutionSpace& exec_space, const ProblemManagerType& pm,
                        const NeighborListType& Ci_list, const NeighborListType& Pi_list, 
                        const GridManager& gridp,
                        const int num_grid, const int extent, const double center, const double h, const double hp)
{

   //Gridp is the fake grid particle list, get positions and ids	
   auto index = gridp.get(Grid::Index());
   auto gridx = gridp.get(Grid::Position());
   auto id    = gridp.get(Grid::Id());

   //Get vorticity, velocity, and positions of real particles
   auto vorticity_p = pm.get(Location::Particle(), Field::Vorticity());
   auto velocity_p = pm.get(Location::Particle(), Field::Velocity());
   auto velocity_corr = pm.get(Location::Node(), Field::Velocity_Corr());
   auto velocity_g = pm.get(Location::Node(), Field::Velocity());
   auto positions  = pm.get(Location::Particle(), Field::Position());
   auto velx       = pm.get(Location::Node(), Field::velx() );
   auto Fx         = pm.get(Location::Node(), Field::Fx() );
   auto advect_vort = pm.get(Location::Particle(), Field::Vorticity_Advect() );
//   Kokkos::deep_copy( velx, 0.0);
   Kokkos::deep_copy( velocity_corr, 0.0);
   //Get relevant interpolation quantities 
   MLC_Interp::GridData<3> g( h, center);
   
   pm.save_v( "Precorrection_V",1,0);
   //Iterate over D0 
   Kokkos::parallel_for(
        "Corrections",
        Kokkos::RangePolicy<ExecutionSpace>( exec_space, 0, num_grid ),
        KOKKOS_LAMBDA( const int i ) {

	    //D0 Grid Indices
            int ii = index(i,0);
            int jj = index(i,1);
            int kk = index(i,2);
	    int idd = id( i );

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

	     auto offset = Pi_list.binOffset(ii,jj,kk);
             auto size   = Pi_list.binSize(ii,jj,kk);

             //Interpolate Particles where floor(xp/h) == i
             for( int r = offset; r < offset+size; r++)
             {

                 auto j = Pi_list.getParticle( r );

                 //Check for Real Particle vs. Grid Particle
                 if( id(j) == 1 ){
                    int p = j - num_grid;

                    for( int si = ii-1; si <= ii+1; si++)
                       for(int sj = jj-1; sj <= jj+1; sj++)
                           for( int sk = kk-1; sk <= kk+1; sk++)
                           {

           			  for(int d = 0; d < 3; d++)
                                     velocity_corr(si,sj,sk,d) = velocity_g(si,sj,sk,d);

                           }

                 }
             }

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
                                         GreensFunction::Calculate_qK(xg, xp, vortp, K);

					 //Correct Velocity
                                         for(int d = 0; d < 3; d++)
                                     	    velocity_corr(si, sj, sk, d) -= K[d]; 

					    velx(si,sj,sk,0) -=K[0];

                                     } 
                                }

                           }

                }



	     //Interpolate Particles where floor(xp/h) == i
             for( int r = offset; r < offset+size; r++)
             {

		 auto j = Pi_list.getParticle( r );   

		 //Check for Real Particle vs. Grid Particle 
		 if( id(j) == 1 ){
	            int p = j - num_grid;		 
                    double xp[3] = { positions(p, 0 ), positions( p, 1 ), positions( p, 2 ) };
                    Kokkos::printf(" xp %f yp %f zp %f \n ", xp[0],xp[1],xp[2]);
                     // Update particle velocity.
                    double u_temp[3];
                    double x_plus[3], x_minus[3], u_plus[3], u_minus[3];
		     // grid index closest to the particle
                    int ip = floor( (xp[0]+g.center) / g.cell_size );
                    int jp = floor( (xp[1]+g.center) / g.cell_size );
                    int kp = floor( (xp[2]+g.center) / g.cell_size );
    //grid position

		    for( int d = 0; d < 3; d++)
	            {
                       x_plus[d] = xp[d] + 0.5*hp*vorticity_p(p,d);
		       x_minus[d] = xp[d] - 0.5*hp*vorticity_p(p,d);
		       
		    }

                   for( int si = ip-1; si <= ip+1; si++)
                     for(int sj = jp-1; sj <= jp+1; sj++)
                       for( int sk = kp-1; sk <= kp+1; sk++)
                       {

		         Kokkos::printf(" velocity_corr = %f \n ", velocity_corr(si,sj,sk,0));
            
                       }

                    //Interpolate from Grid to Particle
		    // g contains cell size information
                    MLC_Interp::HarmonicValue( velocity_corr, g, xp, u_temp );
                    MLC_Interp::HarmonicValue( velocity_corr, g, x_plus, u_plus );
		    MLC_Interp::HarmonicValue( velocity_corr, g, x_minus, u_minus );

                    //Update RHS
                    for(int d = 0; d<3; d++)
                       velocity_p(p, d) = u_temp[d];
                       Kokkos::printf("velocity_p = %f \n ", u_temp[0]);
                     //Iterate over cell stencil of the linked list = Ci
/*                     for( int pi = imin; pi < imax; pi++)
                        for( int pj = jmin; pj < jmax; pj ++)
                           for( int pk = kmin; pk < kmax; pk ++)
                           {
                                //Get Offset and Size to determine # particles
                                auto Ci_offset = Ci_list.binOffset(pi,pj,pk);
                                auto Ci_size   = Ci_list.binSize(pi,pj,pk);

                                //Loop over Ci
                                for( std::size_t m = Ci_offset; m < Ci_offset+Ci_size; m++)
                                {

                                     //Get true particle ID in fake particle list
                                     auto y = Ci_list.getParticle( m );

                                     //Check that it is a real particle vs fake
                                     if( id(y) == 1 ){


                                         //Get Real Particle ID
                                         int q = y - num_grid;

                                         // Get Vorticity and Position
                                         double vortq[3] = { vorticity_p(q,0), vorticity_p(q,1), vorticity_p(q,2) };
                                         double xq[3]    = { positions(q,0), positions(q,1), positions(q,2) };
                                         double K[3],K_minus[3], K_plus[3];

                                         //Calculate Green's Function
                                         GreensFunction::CalculateK(x_plus, xq, vortq, K_plus);
					 GreensFunction::CalculateK(x_minus, xq, vortq, K_minus);
					 GreensFunction::CalculateK(xp, xq, vortq, K);

                                         //Correct Velocity
                                         for(int d = 0; d < 3; d++)
				         {
                                             u_plus[d]  += K_plus[d];
				             u_minus[d] += K_minus[d];
					     velocity_p(p,d) += K[d];

				         }

                                     }
                                }

                           }
*/
		    
                    for( int d = 0; d < 3; d++)
		       advect_vort(p,d) = (u_plus[d] - u_minus[d] ) / hp;
             
		    
		 }
             }


        });


     pm.save_v( "Post_Correction_V",1,0.0);    
        

}

 template <class ProblemManagerType, class ExecutionSpace, class NeighborListType>
 void Interaction_NBody( const ExecutionSpace& exec_space, const ProblemManagerType& pm, 
		        const NeighborListType& neigh_list, const int c,
		       	const double center, const double cell_size, const double hp )
{
	
    double x[3], xq[3];
    double K[3], vort[3]; 

    //Get vorticity, postion and velocity on particles
    auto u_p = pm.get( Location::Particle(), Field::Velocity() );
    auto x_p = pm.get( Location::Particle(), Field::Position() );
    auto vort_p = pm.get( Location::Particle(), Field::Vorticity() );
    auto advect_vorticity = pm.get( Location::Particle(), Field::Vorticity_Advect() );
    // p is the particle of interest and q is the neighbor particle
    auto  interaction = KOKKOS_LAMBDA(const int p, const int  q){


	// Particle P Location
	double x[3]  = { x_p( p, 0 ), x_p( p, 1 ), x_p( p, 2 ) };

	//Particle Q (Neighbor) Position
        double xq[3] = { x_p( q, 0 ), x_p( q, 1 ), x_p( q, 2 ) };
        double vort[3]  = { vort_p( q, 0 ), vort_p( q, 1 ), vort_p( q, 2 ) };
        double K[3], K_plus[3], K_minus[3];
	double xp_minus[3], xp_plus[3];

	for(int d = 0; d < 3; d++)
	{
           xp_minus[d] = x[d] - 0.5*hp*vort_p(p,d);
	   xp_plus[d]  = x[d] + 0.5*hp*vort_p(p,d); 

	}

        //Evaluate Green's Function with number
        GreensFunction::Calculate_qK(x, xq, vort, K);
	GreensFunction::Calculate_qK(xp_minus, xq, vort, K_minus);
	GreensFunction::Calculate_qK(xp_plus, xq, vort, K_plus);


	//Correct Velocity at P with Local Neighbor Interaction at Q
        for(int d = 0; d < 3; d++){
              u_p(p,d) += K[d];
	      advect_vorticity(p,d) += ( K_plus[d] - K_minus[d] )/ hp;

/*	std::cout << " interpolation " << std::endl;
        std::cout << " u = " << u_p(p,0) << " v = " << u_p(p,1) << " w = " << u_p(p,2) << std::endl;
        std::cout << " x = " << x[0] << " y = " << x[1] << " z = " << x[2] << std::endl; */

        }

        Kokkos::printf(" p %d, u %f v %f w %f \n", u_p(p,0), u_p(p,1), u_p(p,2) );
                
     };

          //Find neighbors and calculate interaction for all particles
	  Cabana::neighbor_parallel_for(Kokkos::RangePolicy<ExecutionSpace>( exec_space, 0, pm.numParticle() ), interaction, neigh_list, Cabana::FirstNeighborsTag(), Cabana::SerialOpTag(), "LocalCorrections" );

        Kokkos::parallel_for(
        "print_velocity",
        Kokkos::RangePolicy<ExecutionSpace>( exec_space, 0, pm.numParticle() ),
        KOKKOS_LAMBDA( const int i ) {

            Kokkos::printf(" p %d, u %f v %f w %f \n",i,u_p(i,0), u_p(i,1), u_p(i,2) );

        });
}  
} // end namespace LocalCorrection
} // end namespace ExaMPM

#endif // EXAMPM_LocalCorrection_HPP
