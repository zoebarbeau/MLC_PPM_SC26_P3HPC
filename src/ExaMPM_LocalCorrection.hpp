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
#include "FFTWLGFConvolution.H"
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
       GreensFunction::Calculate_qK(xp, xq, vortp,K,0.5,0.5);

       std::cout << " K = " << K[0] << "  " << K[1] << " " << K[2] << std::endl;

}
// template <class ProblemManagerType, class ExecutionSpace, class GridManager>

/* void Test_L27( const ExecutionSpace& exec_space, const ProblemManagerType& pm,
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


  Kokkos::parallel_for("Copy 1D to 3D", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {extent, extent, extent}),
     KOKKOS_LAMBDA(const int i, const int j, const int k) {

            double x = i*h - center;
            double y = j*h - center;
            double z = k*h - center;
            for(int d = 0; d < 3; d++)
               velocity_g(i,j,k,d) = cos(0.5*x)*sin(y)*cos(0.25*z);

        }
     
   double errormax = 0;

     Kokkos::parallel_for("Copy 1D to 3D", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({1, 1, 1}, {extent-1, extent-1, extent-1}),
     KOKKOS_LAMBDA(const int i, const int j, const int k) {
            double x = i*h - center;
            double y = j*h - center;
            double z = k*h - center;

            MLC_Interp::L27(velocity_g,i,j,k,g,L);
            L_exact =  (-0.5*0.5*cos(0.5*x)*sin(y)*cos(0.25*z) ) + (-1*cos(0.5*x)*sin(y)*cos(0.25*z) )
	              + (-0.25*0.25*cos(0.5*x)*sin(y)*cos(0.25*z) ); 

	    for(int d = 0; d < 3; d++)
	       Error[d] += (std::abs(L[d] - L_exact))*h*h*h;
        // std::cout << " EXACT= " << L_exact << " Real = " << L[0] << std::endl;
        // std::cout << " i = " << i << " j = " << j << " k = " << k << std::endl;

	   if( errormax < std::abs(L[0] - L_exact) )
           {

	      errormax = std::abs(L[0] - L_exact);
	   }		   


        }
    }
   std::cout << " laplacian error = " << Error[0] << " " << Error[1] << " " << Error[2] << std::endl;
   std::cout << " error max = " << errormax << std::endl;
}
*/
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

        double er = 0, exact;
        for(int p = 0; p < pm.numParticle(); p++)
        {

	    double u_temp[3];
	    double xp[3] = { positions(p,0), positions(p,1), positions(p,2) };
	    MLC_Interp::HarmonicValue( velocity_g, g, xp, u_temp );

	    for( int d = 0; d < 3; d++)
	    {	    
	       velocity_p(p,d) = u_temp[d];
	       exact = cos( 2*pi*xp[0] )*sin( 4*pi*xp[1] )*cos( 4*pi*xp[2] );

	       er += abs( exact - velocity_p(p,d) );

	       std::cout << " exact " << exact << " velocity = " << velocity_p(p,d) << std::endl;

            }

	    std::cout << " error = " << er << std::endl;

	    std::cout << " error by particle = " << er/(3*pm.numParticle() ) << std::endl;
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

                  //     Kokkos::printf(" counter %d i %d j %d k %d x %f y %f z %f \n)", particle_counter,i,j,k,positions(particle_counter,0),
                  //            positions(particle_counter,1),positions(particle_counter,2) );
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

                positions( p, 0) = xp[0];//+0.5*h;
                positions( p, 1) = xp[1];//+0.5*h;
                positions( p, 2) = xp[2];//+0.5*h;
            //    Kokkos::printf( " p %d x %f y %f z %f \n", p, xp[0], xp[1], xp[2] );
                //Real particle denoted by an ID of 1
                id( p ) = 1;
        });
}

template <class ProblemManagerType, class ExecutionSpace, class NeighborListType, class GridManager>
 void Deposition( const ExecutionSpace& exec_space, const ProblemManagerType& pm, const NeighborListType& Pi_list,
                  const NeighborListType& Ci_list, const GridManager& gridp, const int num_grid, 
		  const int extent, const double center, const double h,const double hp,const int corr_radius)
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

   int sz = corr_radius*2 +1;
//   Kokkos::View<double*****> vel_loc("local_velocity",num_grid,sz,sz,sz,3);
// Kokkos::deep_copy( vel_loc, 0.0);
   //Iterate over D0 
   Kokkos::parallel_for(
        "Depositions",
        Kokkos::RangePolicy<ExecutionSpace>( exec_space,0,num_grid),
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
            bool particlefound = false;
            int counter = 0;

            double vel_loc[9][9][9][3]={0};
      
   //Reset Ci to 0
/*            for( int ci = imin; ci <= imax; ci++)
                for( int cj = jmin; cj <= jmax; cj ++)
                   for( int ck = kmin; ck <= kmax; ck ++)
                   {
                          for(int d = 0; d < 3; d++)
                              vel_loc[ci-imin][cj-jmin][ck-kmin][d] = 0.0;

                   }
 */

//          assert(imax - imin <= 9 && jmax - jmin <= 9 && kmax - kmin <= 9);   
	    auto offset = Pi_list.binOffset(ii,jj,kk);
            auto size   = Pi_list.binSize(ii,jj,kk);
	    // Iterate over Ci
            for( int ci = imin; ci < imax; ci++)
                 for( int cj = jmin; cj < jmax; cj ++)
                      for( int ck = kmin; ck < kmax; ck ++){
			  //Calculate jh
                          double xg[3] = { ci*h - center, cj*h - center, ck*h - center};    

                          //Loop over Ci
                          for( std::size_t r = offset; r < offset+size; r++)
                          {    
                             /*  if(size > 1){
                                 Kokkos::printf("offset %d size %d \n ", offset, size);
                               } */
                              //Get true particle ID in fake particle list       
                              auto j = Pi_list.getParticle( r );

			      //Check that it is a real particle vs fake
                              if( id(j) == 1 ){
                                   counter++;
                                   //Get Real Particle ID
                                   int p = j - num_grid;
 
                                   // Get Vorticity and Position
                                   double vortp[3]  = { vorticity_p( p, 0 ), vorticity_p( p, 1 ), vorticity_p( p, 2 ) };                                  
                                   double xp[3]    = { positions(p,0), positions(p,1), positions(p,2) };
                                   double K[3];

                                   //Calculate Green's Function
                                   GreensFunction::Calculate_qK(xg, xp, vortp, K,h,corr_radius);
                                   int ip = std::floor((xp[0]+center)/h); int jp = std::floor((xp[1]+center)/h);
                                   int kp = std::floor((xp[2]+center)/h);
                                   //Correct Velocity
                                   for(int d = 0; d < 3; d++){
                   

                                         vel_loc[ci-imin][cj-jmin][ck-kmin][d] += K[d]; 
                                   }

                                   
//                                       Kokkos::printf(" velocity %f x %f y %f z %f i %d \n", K[0],xg[0],xg[1],xg[2],i);
                                 
                                        velx(ci,cj,ck,0) +=K[0];


                                  }
                          


                         }

                  }

                
             if( counter > 0 ){
                 
	         for( int c0i = imin+1; c0i < imax-1; c0i++)
                   for( int c0j = jmin+1; c0j < jmax-1; c0j++)
                      for( int c0k = kmin+1; c0k < kmax-1; c0k++)
                      {

                
                         double xg0[3] = { c0i*h - center, c0j*h - center, c0k*h - center};

	                 // Calculate 2nd order Laplacian of each velocity component 
	                 double F_temp[3] = {0.0, 0.0, 0.0};
                         double u_face[3] = {0.0,0.0,0.0};
                         double u_corner[3] = {0.0,0.0,0.0};
                         double u_edge[3] = {0.0,0.0,0.0};

                        

                         for(int si = c0i-1; si <= c0i+1; si++)
                            for(int sj = c0j-1; sj <= c0j+1; sj++)
                               for(int sk = c0k-1; sk <= c0k+1; sk++)
                                  {


                                      
                                      int s1 = si-c0i;
                                      int s2 = sj-c0j;
                                      int s3 = sk-c0k;
                                      int  s = abs(s1) + abs(s2) + abs(s3);
                                      int li = si-imin;
                                      int lj = sj-jmin;
                                      int lk = sk-kmin;

 
                                      if( s == 1)
                                      {
 
                                         for(int d = 0; d < 3; d++)
                                           u_face[d] += vel_loc[li][lj][lk][d];


                                      }else if( s == 2)
                                      {

                                         for(int d = 0; d < 3; d++)
                                           u_edge[d] += vel_loc[li][lj][lk][d];

                                      }else if( s == 3)
                                      {

                                         for(int d = 0; d < 3; d++)
                                            u_corner[d] += vel_loc[li][lj][lk][d];


                                      }


                                  }



//                            MLC_Interp::L27(vel_loc,c0i-imin, c0j-jmin, c0k-kmin,g,F_temp,i);

		
                         //Set F
   		         for(int d = 0; d < 3; d++){

                            double result = ( vel_loc[c0i-imin][c0j-jmin][c0k-kmin][d]*-128.0/30.0 + u_corner[d]*1.0/30.0 + 
                                              u_edge[d]*1.0/10.0 + 7.0/15.0*u_face[d]) / pow( g.cell_size, 2.0 );

                            Kokkos::atomic_add(&F(c0i,c0j,c0k,d), result);
//		            F(c0i,c0j,c0k,d) += ( vel_loc[c0i-imin][c0j-jmin][c0k-kmin][d]*-128.0/30.0 + u_corner[d]*1.0/30.0 + u_edge[d]*1.0/10.0 + 7.0/15.0*u_face[d]) /pow( g.cell_size, 2.0); ///F_temp[d];
                         }

                            double result = ( vel_loc[c0i-imin][c0j-jmin][c0k-kmin][0]*-128.0/30.0 + u_corner[0]*1.0/30.0 +
                                              u_edge[0]*1.0/10.0 + 7.0/15.0*u_face[0]) / pow( g.cell_size, 2.0 );

                            Kokkos::atomic_add(&Fx(c0i,c0j,c0k,0), result);
                          
//                                Fx(c0i,c0j,c0k,0) +=( vel_loc[c0i-imin][c0j-jmin][c0k-kmin][0]*-128.0/30.0 + u_corner[0]*1.0/30.0 + u_edge[0]*1.0/10.0 + 7.0/15.0*u_face[0]) /(h*h); // F_temp[0];

//                         if(abs(Fx(c0i,c0j,c0k,0) ) > 0 )
//                         Kokkos::printf("c0i %d c0j %d c0z %d Fx %f Fy %f Fz %f \n", c0i,c0j,c0k,F_temp[0],F_temp[1],F_temp[2]);

                          }

                      }

  //              }
/*              for( int ci = imin; ci < imax; ci++)
                  for( int cj = jmin; cj < jmax; cj ++)
                   for( int ck = kmin; ck < kmax; ck ++)
                   {
                          for(int d = 0; d < 3; d++)
                              vel_loc[ci][cj][ck][d] = 0.0;

                   }
*/             

	});


//          pm.save_F("Fy_", 1, 0);

//        int N = extent;
//        Kokkos::printf("velocity added");
/*        Kokkos::parallel_for("Copy 1D to 3D", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {N,N,N}),
        KOKKOS_LAMBDA(const int i, const int j, const int k) {

           if( abs(F(i,j,k,0)) > 1e-6)
           Kokkos::printf("i %d j %d z %d Fx %f Fy %f Fz %f \n", i,j,k,F(i,j,k,0),F(i,j,k,1),F(i,j,k,2));

       });
*/
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

    int i,j,k;
    double delta = 2*h;
       i = floor((0.875-center)/h); j = i; k = i;
//     Kokkos::parallel_for("Copy 1D to 3D", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {extent+1, extent+1, extent+1}),
//        KOKKOS_LAMBDA(const int i, const int j, const int k) {

	      double xg[3] = { i*h - center, j*h - center, k*h - center };	  
              double loc = pow( pow(xg[0]-0.875, 2.0) + pow(xg[1]-0.875,2.0) + pow(xg[2]-0.875,2.0) , 0.5);
              if( loc < pow(10,-6) ){
	      //iterate over D0	  
              for(int i0 = 1; i0 <= extent; i0++)
                for(int j0 = 1; j0 <= extent; j0++)
                   for( int k0 = 1; k0 <= extent; k0++)
                   {

                         double x0[3] = { i0*h - center, j0*h - center, k0*h - center };
			 double r = pow( pow( x0[0]-xg[0], 2.0) + pow( x0[1]-xg[1], 2.0) + pow( x0[2]-xg[2], 2.0), 0.5 );
                //         double r = pow( pow( xg[0], 2.0) + pow( xg[1], 2.0) + pow( xg[2], 2.0), 0.5 );
			 if( r < delta )
		         {
		             for(int d = 0; d < 3; d++)		 
			        velocity_g(i,j,k,d) += F(i0,j0,k0,d)*(1.0/(32.0*Kokkos::numbers::pi*delta )*(-3.0*pow(r/delta, 4.0) + 10.0*pow(r/delta,2.0) - 7.0) - 1.0/(4.0*Kokkos::numbers::pi*delta) );
                         }else{

                             for(int d = 0; d < 3; d++)
			        velocity_g(i,j,k,d) -= F(i0,j0,k0,d)*1.0/(4.0*Kokkos::numbers::pi*r);
                                velx(i,j,k,0) = velocity_g(i,j,k,0);  
			       if ( std::abs( F(i0,j0,k0,0)*1.0/(4.0*Kokkos::numbers::pi*r) ) > 0 
			            || std::abs( F(i0,j0,k0,1)*1.0/(4.0*Kokkos::numbers::pi*r) ) > 0
				    || std::abs( F(i0,j0,k0,2)*1.0/(4.0*Kokkos::numbers::pi*r) ) > 0  ){

 //                                   Kokkos::printf(" r %f x %f y %f z %f F %f i %d j %d k %d \n", r, x0[0], x0[1],x0[2],F(i0,j0,k0,0),i0,j0,k0);
                                } 
			 }
 
                    }

                    double r = pow( pow( xg[0]-0.5-0.15*h, 2.0) + pow( xg[1]-0.5-0.15*h, 2.0) + pow( xg[2]-0.5-0.15*h, 2.0), 0.5 );
                    Kokkos::printf( " x %f y %f z %f velx %f exact %f \n", xg[0],xg[1],xg[2],velocity_g(i,j,k,0),(xg[2] - 0.5-0.15*h) / (4*Kokkos::numbers::pi*pow(r, 3.0) ) );

                }

//         });

             pm.save_v( "Convolution_V",1,0);
             pm.save_F( "Laplacian_V",1,0);

}
 template <class ProblemManagerType, class ExecutionSpace, class NeighborListType, class GridManager>
 void Corrections( const ExecutionSpace& exec_space, const ProblemManagerType& pm,
                        const NeighborListType& Ci_list, const NeighborListType& Pi_list, 
                        const GridManager& gridp,
                        const int num_grid, const int extent, const double center, const double h, const double hp, const int corr_radius)
{

   //Gridp is the fake grid particle list, get positions and ids	
   auto index = gridp.get(Grid::Index());
   auto gridx = gridp.get(Grid::Position());
   auto id    = gridp.get(Grid::Id());

   //Get vorticity, velocity, and positions of real particles
   auto vorticity_p = pm.get(Location::Particle(), Field::Vorticity());
   auto velocity_p  = pm.get(Location::Particle(), Field::Velocity());
   auto velocity_corr = pm.get(Location::Node(), Field::Velocity_Corr());
   auto velocity_g = pm.get(Location::Node(), Field::Velocity());
   auto positions  = pm.get(Location::Particle(), Field::Position());
   auto velx       = pm.get(Location::Node(), Field::velx() );
   auto Fx         = pm.get(Location::Node(), Field::Fx() );
   auto advect_vort = pm.get(Location::Particle(), Field::Vorticity_Advect() );
   Kokkos::deep_copy( velocity_corr, 0.0);
   //Get relevant interpolation quantities 
   MLC_Interp::GridData<3> g( h, center);
   
   pm.save_v( "Precorrection_V",1,0);
   //Iterate over D0
   Kokkos::View<double*****> vel_loc("local_velocity",num_grid,3,3,3,3); 
   Kokkos::parallel_for(
        "Corrections",
        Kokkos::RangePolicy<ExecutionSpace>( exec_space,0,num_grid ),
        KOKKOS_LAMBDA( const int i ) {

	    //D0 Grid Indices
            int ii = index(i,0);
            int jj = index(i,1);
            int kk = index(i,2);
	    int idd = id( i );

 
	    //Offsets for particles by the grid cell
            auto Pi_offset = Pi_list.binOffset(ii,jj,kk);
            auto Pi_size   = Pi_list.binSize(ii,jj,kk);

	    //Get Ci upper and lower bounds
	    // getParticleBin(i) gives the cell/bin associated with the ith grid point
	    // the max a
	    // nd min values of the stencil is built around this bin
            int imin, imax, jmin, jmax, kmin, kmax;
            Ci_list.getStencilCells( Ci_list.getParticleBin( i ), imin,imax, jmin,
                               jmax, kmin, kmax ); 

	     auto offset = Pi_list.binOffset(ii,jj,kk);
             auto size   = Pi_list.binSize(ii,jj,kk);

              for( int si = ii-1; si <= ii+1; si++)
                  for(int sj = jj-1; sj <= jj+1; sj++)
                      for( int sk = kk-1; sk <= kk+1; sk++)
                       {

           			  for(int d = 0; d < 3; d++)
                                     vel_loc(i,si-ii+1,sj-jj+1,sk-kk+1,d) = velocity_g(si,sj,sk,d);

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
                                         GreensFunction::Calculate_qK(xg, xp, vortp, K,h,corr_radius );

					 //Correct Velocity
                                         for(int d = 0; d < 3; d++)
                                     	    vel_loc(i,si-ii+1,sj-jj+1,sk-kk+1,d) -= K[d];

//                                          Kokkos::printf("Kx %f xg %f yg %f zg %f i %d \n", K[0],xg[0],xg[1],xg[2],i); 

                                     } 
                                }

                           }

                   }

     //        std::cout << " completed loop " << std::endl;

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

                    //Interpolate from Grid to Particle
              	    // g contains cell size information
                    MLC_Interp::HarmonicValue( vel_loc, g, xp, u_temp,i );
                    MLC_Interp::HarmonicValue( vel_loc, g, x_plus, u_plus,i );
	            MLC_Interp::HarmonicValue( vel_loc, g, x_minus, u_minus,i );
                    //Update RHS
                    for(int d = 0; d < 3; d++){
                       velocity_p(p,d) = u_temp[d];
                       advect_vort(p,d) = ( u_plus[d] - u_minus[d] ) / hp;
                    }
                   }		    
		 }
             


        });


//     pm.save_v( "Post_Correction_V",1,0.0);    
        

}

 template <class ProblemManagerType, class ExecutionSpace, class NeighborListType>
 void Interaction_NBody( const ExecutionSpace& exec_space, const ProblemManagerType& pm, 
		        const NeighborListType& neigh_list, const int c,
		       	const double center, const double cell_size, const double hp, const int corr_radius )
{
	
    double x[3], xq[3];
    double K[3], vort[3]; 

    //Get vorticity, postion and velocity on particles
    auto u_p = pm.get( Location::Particle(), Field::Velocity() );
    auto x_p = pm.get( Location::Particle(), Field::Position() );
    auto vort_p = pm.get( Location::Particle(), Field::Vorticity() );
    auto advect_vorticity = pm.get( Location::Particle(), Field::Vorticity_Advect() );
    auto velx       = pm.get(Location::Node(), Field::velx() );
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
        GreensFunction::Calculate_qK(x, xq, vort, K,cell_size,corr_radius);
	GreensFunction::Calculate_qK(xp_minus, xq, vort, K_minus,cell_size,corr_radius);
	GreensFunction::Calculate_qK(xp_plus, xq, vort, K_plus,cell_size,corr_radius);


	//Correct Velocity at P with Local Neighbor Interaction at Q
        for(int d = 0; d < 3; d++)
        {
              u_p(p,d) += K[d];
	      advect_vorticity(p,d) += ( K_plus[d] - K_minus[d] )/ hp;
        }

                
     };

          //Find neighbors and calculate interaction for all particles
	  Cabana::neighbor_parallel_for(Kokkos::RangePolicy<ExecutionSpace>( exec_space, 0, pm.numParticle() ), interaction, neigh_list, Cabana::FirstNeighborsTag(), Cabana::SerialOpTag(), "LocalCorrections" );

/*      std::cout << " num particles " << pm.numParticle() << std::endl;

        Kokkos::parallel_for(
        "print_velocity",
        Kokkos::RangePolicy<ExecutionSpace>( exec_space, 0, pm.numParticle() ),
        KOKKOS_LAMBDA( const int i ) {

          double x[3]  = { x_p( i, 0 )-0.5, x_p( i, 1 )-0.5, x_p( i, 2 )-0.5 };
 
          double v_exact[3], v_error[3];
          double r =  sqrt(x[1]*x[1] + x[2]*x[2]+x[0]*x[0]);
          double R = 0.25;
          double R2 = 0.25*0.25;
          double U = 1;
          double R3 = 0.25*0.25*0.25;
          double xz = x[0]*x[2];
          double yz = x[1]*x[2];
          if( r < (0.25-1e-8) ){

            v_exact[0] = -1.5*xz/R2*U;
            v_exact[1] = -1.5*yz/R2*U;
            v_exact[2] = -1.5*( 1.0 + x[2]*x[2]/R2 -  2.0*r*r/R2  ) *U;
          }else{

            v_exact[0] = -1.5 * (  xz / pow( r, 5.0) )*R3*U;
            v_exact[1] = -1.5 * (  yz / pow( r, 5.0) )*R3*U;
            v_exact[2] = U*( 1.0 + R3 / (2.0*pow( r, 3.0 ) ) ) - U*1.5*R3 / (  pow( r, 5.0 ) ) * x[2]*x[2] ;
            
     }




            for( int d = 0; d < 3; d++){
              v_error[d] = v_exact[d] - u_p(i,d); 
            }
                 
  //        Kokkos::printf(" p %d, u %f v %f w %f  uex %f vex %f wex %f x %f y %f z %f \n",i,u_p(i,0),u_p(i,1), u_p(i,2), v_exact[0], v_exact[1], v_exact[2],x[0],x[1],x[2]);
//             Kokkos::printf(" u_p %e advection vorticity %e \n", u_p(i,0),advect_vorticity(i,0) );

        });
*/

} 


 template <class ProblemManagerType, class ExecutionSpace,class LocalGridType>
 void Error_V( const ExecutionSpace& exec_space, const ProblemManagerType& pm,
                        const int extent, const double h, const double hp,const LocalGridType& local_grid )
{

  int N = extent;
  auto velocity_g = pm.get(Location::Node(), Field::Velocity());    
  auto u_p = pm.get( Location::Particle(), Field::Velocity() );
  auto x_p = pm.get( Location::Particle(), Field::Position() );

  double max_final = 0, L2_final = 0;


/*   Kokkos::parallel_reduce("Copy 1D to 3D", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {N,N,N}),
        KOKKOS_LAMBDA(const int i, const int j, const int k, double& L2_g) {
          int index_f = i * N * N + j * N + k;
          double x[3] = {i*h-0.5,j*h-0.5,k*h-0.5};
          double v_exact[3],v_error[3];
          double r =  sqrt(x[1]*x[1] + x[2]*x[2]+x[0]*x[0]);
          double R = 0.25;
          double R2 = 0.25*0.25;
          double U = 1;
          double R3 = 0.25*0.25*0.25;
          double xz = x[0]*x[2];
          double yz = x[1]*x[2];
          if( r < (0.25-1e-10) ){

            v_exact[0] = -1.5*xz/R2*U;
            v_exact[1] = -1.5*yz/R2*U;
            v_exact[2] = -1.5*( 1.0 + x[2]*x[2]/R2 -  2.0*r*r/R2  ) *U;
          }else{

            v_exact[0] = -1.5 * (  xz / pow( r, 5.0) )*R3*U;
            v_exact[1] = -1.5 * (  yz / pow( r, 5.0) )*R3*U;
            v_exact[2] = U*( 1.0 + R3 / (2.0*pow( r, 3.0 ) ) ) - U*1.5*R3 / (  pow( r, 5.0 ) ) * x[2]*x[2] ;

         }, sum);
*/

     // Get the local set of owned cell indices.
 auto owned_cells = local_grid.indexSpace(  Cabana::Grid::Ghost(), Cabana::Grid::Node(), Cabana::Grid::Local() );

  Kokkos::parallel_reduce(
        "L2Grid",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {N,N,N}),
        KOKKOS_LAMBDA( const int i, const int j, const int k, double& L2_error) {
          int index_f = i * N * N + j * N + k;
          double x[3] = {i*h-0.5,j*h-0.5,k*h-0.5};
          double v_exact[3],v_error[3],v_magn; 
          double r =  sqrt(x[1]*x[1] + x[2]*x[2]+x[0]*x[0]);
          double R = 0.25;
          double R2 = 0.25*0.25;
          double U = 1;
          double R3 = 0.25*0.25*0.25;
          double xz = x[0]*x[2];
          double yz = x[1]*x[2];
          if( r <( 0.25 - 1e-10) ){

            v_exact[0] = -1.5*xz/R2*U;
            v_exact[1] = -1.5*yz/R2*U;
            v_exact[2] = -1.5*( 1.0 + x[2]*x[2]/R2 -  2.0*r*r/R2  ) *U;
          }else{

            v_exact[0] = -1.5 * (  xz / pow( r, 5.0) )*R3*U;
            v_exact[1] = -1.5 * (  yz / pow( r, 5.0) )*R3*U;
            v_exact[2] = U*( 1.0 + R3 / (2.0*pow( r, 3.0 ) ) ) - U*1.5*R3 / (  pow( r, 5.0 ) ) * x[2]*x[2] ;

          }

           for(int d = 0; d < 3; d++)
             v_error[d] = v_exact[d] - velocity_g(i,j,k,d);
             
           v_magn = sqrt( v_error[0]*v_error[0] + v_error[1]*v_error[1] + v_error[2]*v_error[2] );
//           if( v_magn > max_error )
//               max_error = v_magn;

           for( int d = 0; d < 3; d++)
              L2_error += ( pow(v_exact[d] - velocity_g(i,j,k,d), 2.0))*h*h*h;
       },L2_final);

    L2_final = sqrt(L2_final);
    std::cout << "L2 GRID = " << std::setprecision(12) << L2_final << std::endl;
//  Kokkos::printf(" L2 grid error %f max grid %f \n", L2_final);


   double L2_pfinal=0.0;
   double vm, maxp_final = 0;
  const size_t numP = pm.numParticle();
   Kokkos::parallel_reduce("reduce particles", Kokkos::RangePolicy<ExecutionSpace>( exec_space, 0, pm.numParticle() ), KOKKOS_LAMBDA( const int i, double& L2_p){

       double x[3]  = { x_p( i, 0 )-0.5, x_p( i, 1 )-0.5, x_p( i, 2 )-0.5 };

          double v_exact[3], v_error[3],vm;
          double r =  sqrt(x[1]*x[1] + x[2]*x[2]+x[0]*x[0]);
          double R = 0.25;
          double R2 = 0.25*0.25;
          double U = 1;
          double R3 = 0.25*0.25*0.25;
          double xz = x[0]*x[2];
          double yz = x[1]*x[2];
          if( r < (0.25-1e-10) ){

            v_exact[0] = -1.5*xz/R2*U;
            v_exact[1] = -1.5*yz/R2*U;
            v_exact[2] = -1.5*( 1.0 + x[2]*x[2]/R2 -  2.0*r*r/R2  ) *U;
          }else{

            v_exact[0] = -1.5 * (  xz / pow( r, 5.0) )*R3*U;
            v_exact[1] = -1.5 * (  yz / pow( r, 5.0) )*R3*U;
            v_exact[2] = U*( 1.0 + R3 / (2.0*pow( r, 3.0 ) ) ) - U*1.5*R3 / (  pow( r, 5.0 ) ) * x[2]*x[2] ;

           }

          for( int d = 0; d < 3; d++){
              v_error[d] = v_exact[d] - u_p(i,d); 
              L2_p += pow(v_exact[d] - u_p(i,d), 2.0)*hp*hp*hp;
          }
          
//        vm = sqrt( v_error[0]*v_error[0] + v_error[1]*v_error[1] + v_error[2]*v_error[2] );
          
//        if( vm > maxp )
//           maxp = vm;
     
    
//           Kokkos::printf("p %d u %e v %e w %e x %e y %e z %e \n ", i,u_p(i,0),u_p(i,1),u_p(i,2),x_p(i,0),x_p(i,1),x_p(i,2) );

     },L2_pfinal);

     L2_pfinal = sqrt(L2_pfinal);

//     Kokkos::printf(" L2 particle error %f  \n", L2_pfinal);
std::cout << "L2 P = " << std::setprecision(12) << L2_pfinal << std::endl;
}

 template <class ProblemManagerType, class ExecutionSpace>
 void ConvFFTW( const ExecutionSpace& exec_space, const ProblemManagerType& pm,
                        const int extent, const double h, const int d )
{


    int N = extent; 
    std::vector<double> dataIn(N*N*N,0.);
    std::vector<double> dataOut(N*N*N,0.); 

    auto F = pm.get(Location::Node(), Field::F() );
    auto velocity_g = pm.get(Location::Node(), Field::Velocity());
    auto velx       = pm.get(Location::Node(), Field::velx() );
    Kokkos::deep_copy(velx,0.0);
    //3D point charge field
    for(int i = 0; i < N; i++){
      for(int j = 0; j < N; j++){
        for(int k = 0; k < N; k++){
             // index for 1D F
          int index_f = i * N * N + j * N + k;
           dataIn[index_f] = F(i, j, k,d);
         }
      }
    }  

   std::string file="/g/g16/barbeau2/CPU/MLC_PPM/LatticeGreensFunction/exec/G_128_Octant"; 

   LGFConvolution LGFConv(file,h,extent);

   LGFConv.convolveWithG(dataIn,dataOut);

   double U = 1.0;
   if(d == 2){

     Kokkos::parallel_for("Copy 1D to 3D", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {N+1,N+1,N+1}),
         KOKKOS_LAMBDA(const int i, const int j, const int k) {

              int index_f = i * N * N + j * N + k;
              velocity_g(i,j,k,d) = U;
              velx(i,j,k,0)       = U;

    });

 }



   Kokkos::parallel_for("Copy 1D to 3D", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {N,N,N}),
        KOKKOS_LAMBDA(const int i, const int j, const int k) {
          int index_f = i * N * N + j * N + k;
          double x[3] = {i*h-0.5,j*h-0.5,k*h-0.5};
          double v_exact[3],v_error[3];      
          double r =  sqrt(x[1]*x[1] + x[2]*x[2]+x[0]*x[0]);
          double R = 0.25;
          double R2 = 0.25*0.25;
          double U = 1;
          double R3 = 0.25*0.25*0.25;
          double xz = x[0]*x[2];
          double yz = x[1]*x[2];
          if( r < (0.25-1e-10) ){

            v_exact[0] = -1.5*xz/R2*U;
            v_exact[1] = -1.5*yz/R2*U;
            v_exact[2] = -1.5*( 1.0 + x[2]*x[2]/R2 -  2.0*r*r/R2  ) *U;
          }else{

            v_exact[0] = -1.5 * (  xz / pow( r, 5.0) )*R3*U;
            v_exact[1] = -1.5 * (  yz / pow( r, 5.0) )*R3*U;
            v_exact[2] = U*( 1.0 + R3 / (2.0*pow( r, 3.0 ) ) ) - U*1.5*R3 / (  pow( r, 5.0 ) ) * x[2]*x[2] ;

         }

     //     std::cout << v_exact[2] << std::endl;
          if(d == 2 ){
            velocity_g(i,j,k,d) += dataOut[index_f];
            velx(i,j,k,0)  += (dataOut[index_f]);
          }else if(d == 1){

            velocity_g(i,j,k,d) = dataOut[index_f];
            velx(i,j,k,0)  = dataOut[index_f]; 
            
          }else if(d == 0){

            velocity_g(i,j,k,d) = dataOut[index_f];
            velx(i,j,k,0) =  dataOut[index_f];
          }
   });

   std::stringstream ss;
   ss << d << "_Velocity";
   pm.save_v( ss.str(),1,0.0);   

}    
 template <class ProblemManagerType, class ExecutionSpace, class NeighborListType>
 void Test_ReadBack( const ExecutionSpace& exec_space, const ProblemManagerType& pm,
                        const NeighborListType& neigh_list, const int c,const int extent,
                        const double center, const double h, const double hp, const int corr_radius )
{
    
     double *output = new double[extent * extent * extent]; 
     auto velx = pm.get(Location::Node(), Field::velx());
     auto velocity_g = pm.get(Location::Node(), Field::Velocity());
     int DIM  = 0;
    
     Kokkos::parallel_for("Copy 4D to 1D", Kokkos::MDRangePolicy<Kokkos::Rank<4>>({0, 0, 0, 0}, {extent, extent, extent, 1}),
      KOKKOS_LAMBDA(const int i, const int j, const int k, const int m) {
            int index = i * extent * extent + j * extent + k;
            double x[3] = {i*h-center, j*h-center,k*h-center};
            double r = pow( pow(x[0],2.0) + pow(x[1], 2.0) + pow(x[2],2.0) , 0.5);

            if( r < 6 ){
             output[index] = x[2];
            }else{
      
             output[index] = 0;

            }
      });

     for(int i0 = 0; i0 < (extent)*(extent)*(extent); i0++){

            
            std::cout << " output = \n" << output[i0] << std::endl;

     }
     Kokkos::parallel_for("Copy 1D to 3D", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {extent, extent, extent}),
     KOKKOS_LAMBDA(const int i, const int j, const int k) {
       int index = i * extent * extent + j * extent + k;
       velocity_g(i, j, k,DIM) = output[index];

       double x[3] = {i*h-center, j*h-center,k*h-center};
       double r = pow( pow(x[0],2.0) + pow(x[1], 2.0) + pow(x[2],2.0) , 0.5);

       velx(i,j,k,0) = output[index];
       Kokkos::printf( " velocity %f r %f x %f y %f z %f \n", velocity_g(i,j,k,DIM),r,x[0],x[1],x[2] );
     });
     //
 }
} // end namespace LocalCorrection
} // end namespace ExaMPM

#endif // EXAMPM_LocalCorrection_HPP
