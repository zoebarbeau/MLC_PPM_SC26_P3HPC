/****************************************************************************
 * Copyright (c) 2018-2020 by the MLC authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the MLC library. MLC is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef MLC_LOCALCORRECTION_HPP
#define MLC_LOCALCORRECTION_HPP

#include <MLC_MLC_Interp.hpp>
#include <MLC_ProblemManager2.hpp>
#include <Cabana_Grid.hpp>
#include <MLC_GreensFunction.hpp>
#include <MLC_GridManager.hpp>
#include <Kokkos_Core.hpp>
#include <cmath>
#include <counters.hpp>
//#include "FFTWLGFConvolution.H"
namespace MLC
{
namespace LocalCorrection
{
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

   Kokkos::deep_copy( F, 0.0);
   Kokkos::deep_copy( velocity_g, 0.0);
   //Get relevant interpolation quantities 
   MLC_Interp::GridData<3> g( h, center);
   PerfCounters counters("DepositionCounters");
   
   int sz = corr_radius*2 +1;
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

	    // This gives the stencil defining which grid points interact with the particles
	    // in the cell associated with grid point i
	    // This stencil is based on the correction radius
	    // Here the bounds of the correction radius are returned
            Ci_list.getStencilCells( Ci_list.getParticleBin( i ), imin,imax, jmin,
                               jmax, kmin, kmax );
            bool particlefound = false;
            int counter = 0;

	    //This array stores the velocity approximation for grid point i 
	    //which will be used to calculate the Laplacian
            double vel_loc[9][9][9][3]={0};
      

	    //Pi defines the number of particles within the cell of grid point i
	    //offset and size can be used to iterate over the particles in a bin/cells/correction radius
	    auto offset = Pi_list.binOffset(ii,jj,kk);
            auto size   = Pi_list.binSize(ii,jj,kk);


             //For each particle in Pi, we approximate its velocity contribution 
	     //within the stencil defined by the correction radius
             for( std::size_t r = offset; r < offset+size; r++)
             {

		     auto j = Pi_list.getParticle( r );

                     counter++;
                     int p = j ;//- num_grid;
                     double vortp[3]  = { vorticity_p( p, 0 ), vorticity_p( p, 1 ), vorticity_p( p, 2 ) };
                     double xp[3]    = { positions(p,0), positions(p,1), positions(p,2) };


            	    // Iterate over Ci and calculate the velocity contribution between Ci and the particles in Pi
                      for( int ci = imin; ci < imax; ci++)
                         for( int cj = jmin; cj < jmax; cj ++)
                              for( int ck = kmin; ck < kmax; ck ++){
        			  //Calculate jh
                                  double xg[3] = { ci*h - center, cj*h - center, ck*h - center};    
                                  double K[3];
                                  //Calculate Green's Function
                                  GreensFunction::Calculate_qK(xg, xp, vortp, K, hp, corr_radius);
                                  for(int d = 0; d < 3; d++){
        
                                     vel_loc[ci-imin][cj-jmin][ck-kmin][d] += K[d]; 
                                  }
        

                         }

                  }

                
             if( counter > 0 ){

		 // Now on C0i (Ci-1), the laplacian of the velocity contribution is calculated 
		 // The laplacian of the local velocity contributions are added to generate a global laplacian of velocity 
		 // approximation that the FFT will act on
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


                      			 
                         //This is the L27 laplacian calculation
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




		
                         //Set F
   		         for(int d = 0; d < 3; d++){

                            double result = ( vel_loc[c0i-imin][c0j-jmin][c0k-kmin][d]*-128.0/30.0 + u_corner[d]*1.0/30.0 + 
                                              u_edge[d]*1.0/10.0 + 7.0/15.0*u_face[d]) / (g.cell_size*g.cell_size);

			    //Global acculmation of the answer
                            Kokkos::atomic_add(&F(c0i,c0j,c0k,d), result);
                         }

                          }

                      }


	});

}
 template <class ProblemManagerType, class ExecutionSpace, class NeighborListType, class GridManager>
 void Corrections( const ExecutionSpace& exec_space, const ProblemManagerType& pm,
                        const NeighborListType& Ci_list, const NeighborListType& Pi_list, const NeighborListType& Neigh_list,
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

	    //Get Ci upper and lower bounds to define the correction radius
	    // getParticleBin(i) gives the cell/bin associated with the ith grid point
	    // the max and min values of the stencil is built around this bin
            int imin, imax, jmin, jmax, kmin, kmax;
            Ci_list.getStencilCells( Ci_list.getParticleBin( i ), imin,imax, jmin,
                               jmax, kmin, kmax ); 

        
	     auto offset = Pi_list.binOffset(ii,jj,kk);
             auto size   = Pi_list.binSize(ii,jj,kk);

             // Range of Local Correction
             double vel_loc[3][3][3][3]={0};

	     //The correction is local to a grid point i so the grid of corrected velocity is reset to 0 for each point
	     //This array of corrected velocity is the far field contribution to velocity and is interpolated to the particles
              for( int si = ii-1; si <= ii+1; si++){
                  for(int sj = jj-1; sj <= jj+1; sj++){
                      for( int sk = kk-1; sk <= kk+1; sk++)
                       {

           			  for(int d = 0; d < 3; d++)
				  vel_loc[si-ii+1][sj-jj+1][sk-kk+1][d] = velocity_g(si,sj,sk,d);
				  

                                  

                        }
                    }
                }
        

        // The correction radius is iterated over
	// For each bin of the correction radius, the particles and their contribution are calculated
        for(int pi = imin; pi < imax; pi++)
            for(int pj = jmin; pj < jmax; pj++)
                for(int pk = kmin; pk < kmax; pk++) {
                    
                    auto Ci_offset = Neigh_list.binOffset(pi, pj, pk);
                    auto Ci_size   = Neigh_list.binSize(pi, pj, pk);
                    
		    // Find all particles in Ci
                    for(std::size_t r = Ci_offset; r < Ci_offset + Ci_size; r++) {
                        auto j = Neigh_list.getParticle(r);
                        
                            int p = j; //! - num_grid;
      
      			    
                            double vortp[3] = {vorticity_p(p,0), vorticity_p(p,1), vorticity_p(p,2)};
                            double xp[3] = {positions(p,0), positions(p,1), positions(p,2)};
                            
                            //3x3x3 stencil around grid point si where corrected velocity is assessed
			    //the stencil is later used for interpolation
                            for(int si = 0; si < 3; si++) {
                                for(int sj = 0; sj < 3; sj++) {
                                    for(int sk = 0; sk < 3; sk++) {
                                        double K[3];
                                        double xg[3] = { (ii-1+si)*h-center,(jj-1+sj)*h-center,(kk-1+sk)*h-center};
                                        GreensFunction::Calculate_qK_MatVec_Fused(xg,xp, vortp, K, hp);
					for(int d = 0; d<3; d++)
                                        vel_loc[si][sj][sk][d] -= K[d];
                                    }
                                }
                            }
			    
                        }
                    }


	     //For each particle in Pi, associated with grid point i, the corrected velocity is interpolated to the particle
	     //Interpolate Particles where floor(xp/h) == i
             for( int r = offset; r < offset+size; r++)
             {


              	    auto j = Pi_list.getParticle( r );   

		    //Check for Real Particle vs. Grid Particle 
	            int p = j; // - num_grid;		 
                    double xp[3] = { positions(p, 0 ), positions( p, 1 ), positions( p, 2 ) };
                     // Update particle velocity.
                    double u_temp[3];
                    double x_plus[3], x_minus[3], u_plus[3], u_minus[3];
                    //grid position
		    // We also want the velocity at x+ and x- so we can approximate the vorticty term
		    for( int d = 0; d < 3; d++)
	            {
                       x_plus[d] = xp[d] + 0.5*hp*vorticity_p(p,d);
		       x_minus[d] = xp[d] - 0.5*hp*vorticity_p(p,d);
		       
		    }

		    //Here we do a harmonic interpolation of the corrected velocity to particles
		    //this separates the near and far field
                    MLC_Interp::HarmonicValue_local( vel_loc, g, xp, u_temp );
                    MLC_Interp::HarmonicValue_local( vel_loc, g, x_plus, u_plus );
                    MLC_Interp::HarmonicValue_local( vel_loc, g, x_minus, u_minus );

                    //Update RHS
		    //Here we update the RHS of the particles with the far field contribution
                    for(int d = 0; d < 3; d++){
                       velocity_p(p,d) = u_temp[d];
                       advect_vort(p,d) = ( u_plus[d] - u_minus[d] ) / hp;
                    }
		 }
             


        });


        

}


 template <class ExecutionSpace, class NeighborListType,class VortSlice, class PosSlice, class USlice, class AdvectVortSlice>
 void Interaction_NBody( const ExecutionSpace& exec_space, PosSlice x_p, USlice u_p, VortSlice vort_p, AdvectVortSlice advect_vorticity,
		        NeighborListType& neigh_list, const int c,
		       	const double center, const double cell_size, const double hp, const int corr_radius, const int numP )
{
	
    PerfCounters counters("InteractionsCounters");

    //The interactions step calculates the interactions between the particles in grid cell i and the correction radius
    //This is streamlined by use of the neighbor list
    auto  interaction = KOKKOS_LAMBDA(const int p, const int  q){

         double x[3]  = { x_p( p, 0 ), x_p( p, 1 ), x_p( p, 2 ) };
         double xp_minus[3], xp_plus[3];

	 //We also need the particle-particle interaction at x+ and x- to approximate the vorticity advection term

         for(int d = 0; d < 3; d++)
         {
                xp_minus[d] = x[d] - 0.5*hp*vort_p(p,d);
                xp_plus[d]  = x[d] + 0.5*hp*vort_p(p,d);

         }


	//Particle Q (Neighbor) Position
        double xq[3] = { x_p( q, 0 ), x_p( q, 1 ), x_p( q, 2 ) };
        double vort[3]  = { vort_p( q, 0 ), vort_p( q, 1 ), vort_p( q, 2 ) };
        double K[3], K_plus[3], K_minus[3];

        //Evaluate Green's Function with number
/*      GreensFunction::Calculate_qK_MatVec_Fused(x, xq, vort, K,hp);
	GreensFunction::Calculate_qK_MatVec_Fused(xp_minus, xq, vort, K_minus,hp);
	GreensFunction::Calculate_qK_MatVec_Fused(xp_plus, xq, vort, K_plus,hp);
*/

	//Here we calculate the interaction
        GreensFunction::Calculate_qK(x, xq, vort, K,hp,corr_radius);
        GreensFunction::Calculate_qK(xp_minus, xq, vort, K_minus,hp,corr_radius);
        GreensFunction::Calculate_qK(xp_plus, xq, vort, K_plus,hp,corr_radius);


        //The near field contribution is added
        for(int d = 0; d < 3; d++)
        {
             u_p(p,d) += K[d];
             advect_vorticity(p,d) += ( K_plus[d] - K_minus[d] )/ hp;
        }

                
     };

          //Find neighbors and calculate interaction for all particles
	  Cabana::neighbor_parallel_for(Kokkos::RangePolicy<ExecutionSpace>( 0,numP ), interaction, neigh_list, Cabana::FirstNeighborsTag(), Cabana::SerialOpTag(), "LocalInteractions" );

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


   // 2. Check intermediate error values
   double max_error_component = 0;
   Kokkos::parallel_reduce("Check errors",
       Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0}, {N,N,N}),
       KOKKOS_LAMBDA(const int i, const int j, const int k, double& max_err) {
           double x[3] = {i*h-0.5, j*h-0.5, k*h-0.5};
           int index_f = i * N * N + j * N + k;
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
   
   
           // ... your v_exact calculation ...
           for(int d = 0; d < 3; d++) {
               double err = fabs(v_exact[d] - velocity_g(i,j,k,d));
               if(err > max_err) max_err = err;
           }
       }, Kokkos::Max<double>(max_error_component));
   
      std::cout << "Max error component: " << max_error_component << std::endl;
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
   
              for( int d = 0; d < 3; d++)
                 L2_error += ( pow(v_exact[d] - velocity_g(i,j,k,d), 2.0))*h*h*h;
          },L2_final);
   
       L2_final = sqrt(L2_final);
       std::cout << "L2 GRID = " << std::setprecision(12) << L2_final << std::endl;
   
   
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
             
   
        },L2_pfinal);
   
        L2_pfinal = sqrt(L2_pfinal);
   
std::cout << "L2 P = " << std::setprecision(12) << L2_pfinal << std::endl;
}

template <class ProblemManagerType, class ExecutionSpace, class NeighborListType, class GridManager>
void Corrections_Split(
    const ExecutionSpace& exec_space,
    const ProblemManagerType& pm,
    const NeighborListType& Ci_list,
    const NeighborListType& Pi_list,
    const NeighborListType& Neigh_list,
    const GridManager& gridp,
    const int num_grid,
    const int extent,
    const double center,
    const double h,
    const double hp,
    const double corr_radius)
{
    using MemSpace = typename ExecutionSpace::memory_space;

    auto index         = gridp.get(Grid::Index());
    auto vorticity_p   = pm.get(Location::Particle(), Field::Vorticity());
    auto velocity_p    = pm.get(Location::Particle(), Field::Velocity());
    auto velocity_corr = pm.get(Location::Node(), Field::Velocity_Corr());
    auto velocity_g    = pm.get(Location::Node(), Field::Velocity());
    auto positions     = pm.get(Location::Particle(), Field::Position());
    auto advect_vort   = pm.get(Location::Particle(), Field::Vorticity_Advect());

    Kokkos::deep_copy(velocity_corr, 0.0);

    MLC_Interp::GridData<3> g(h, center);

    const double inv_hp  = 1.0 / hp;
    const double half_hp = 0.5 * hp;
    constexpr int stencil = 3;

    // Temporary corrected stencil per grid cell:
    // [grid cell][si][sj][sk][component]
    Kokkos::View<double*****, MemSpace> corr_stencil(
        "corr_stencil", num_grid, stencil, stencil, stencil, 3);

    // Build corrected local stencil for each grid cell
    Kokkos::parallel_for(
        "Corrections_BuildStencil",
        Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_grid),
        KOKKOS_LAMBDA(const int i)
        {
            const int ii = index(i, 0);
            const int jj = index(i, 1);
            const int kk = index(i, 2);

            const auto Pi_offset = Pi_list.binOffset(ii, jj, kk);
            const auto Pi_size   = Pi_list.binSize(ii, jj, kk);

            // No particles in this cell => nothing to interpolate later
            if (Pi_size == 0) return;

            int imin, imax, jmin, jmax, kmin, kmax;
            Ci_list.getStencilCells(Ci_list.getParticleBin(i),
                                    imin, imax, jmin, jmax, kmin, kmax);

            // Precompute stencil node coordinates once
            double gx[stencil], gy[stencil], gz[stencil];
            for (int s = 0; s < stencil; ++s) {
                gx[s] = (ii - 1 + s) * h - center;
                gy[s] = (jj - 1 + s) * h - center;
                gz[s] = (kk - 1 + s) * h - center;
            }

            // Optional short-circuit: check whether any correction particles exist
            bool has_corr_particles = false;
            for (int pi = imin; pi < imax && !has_corr_particles; ++pi) {
                for (int pj = jmin; pj < jmax && !has_corr_particles; ++pj) {
                    for (int pk = kmin; pk < kmax && !has_corr_particles; ++pk) {
                        if (Neigh_list.binSize(pi, pj, pk) > 0)
                            has_corr_particles = true;
                    }
                }
            }

            // Start from base grid velocity
            for (int si = 0; si < stencil; ++si) {
                const int i_idx = ii - 1 + si;
                for (int sj = 0; sj < stencil; ++sj) {
                    const int j_idx = jj - 1 + sj;
                    for (int sk = 0; sk < stencil; ++sk) {
                        const int k_idx = kk - 1 + sk;

                        corr_stencil(i, si, sj, sk, 0) = velocity_g(i_idx, j_idx, k_idx, 0);
                        corr_stencil(i, si, sj, sk, 1) = velocity_g(i_idx, j_idx, k_idx, 1);
                        corr_stencil(i, si, sj, sk, 2) = velocity_g(i_idx, j_idx, k_idx, 2);
                    }
                }
            }

            if (!has_corr_particles) return;

            // Apply Green's function corrections
            for (int pi = imin; pi < imax; ++pi) {
                for (int pj = jmin; pj < jmax; ++pj) {
                    for (int pk = kmin; pk < kmax; ++pk) {

                        const auto Ci_offset = Neigh_list.binOffset(pi, pj, pk);
                        const auto Ci_size   = Neigh_list.binSize(pi, pj, pk);

                        for (std::size_t r = Ci_offset; r < Ci_offset + Ci_size; ++r) {
                            const auto p = Neigh_list.getParticle(r);

                            const double vortp_x = vorticity_p(p, 0);
                            const double vortp_y = vorticity_p(p, 1);
                            const double vortp_z = vorticity_p(p, 2);

                            const double xp_x = positions(p, 0);
                            const double xp_y = positions(p, 1);
                            const double xp_z = positions(p, 2);

                            const double xp_arr[3]    = {xp_x, xp_y, xp_z};
                            const double vortp_arr[3] = {vortp_x, vortp_y, vortp_z};

                            for (int si = 0; si < stencil; ++si) {
                                const double xg_x = gx[si];
                                for (int sj = 0; sj < stencil; ++sj) {
                                    const double xg_y = gy[sj];
                                    for (int sk = 0; sk < stencil; ++sk) {
                                        const double xg_z = gz[sk];

                                        const double xg_arr[3] = {xg_x, xg_y, xg_z};
                                        double K[3];

                                        // Keep this call exactly as-is
                                        GreensFunction::Calculate_qK_MatVec_Fused(
                                            xg_arr, xp_arr, vortp_arr, K, hp);

                                        corr_stencil(i, si, sj, sk, 0) -= K[0];
                                        corr_stencil(i, si, sj, sk, 1) -= K[1];
                                        corr_stencil(i, si, sj, sk, 2) -= K[2];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        });


    // Interpolation Step
    Kokkos::parallel_for(
        "Corrections_Interpolate",
        Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_grid),
        KOKKOS_LAMBDA(const int i)
        {
            const int ii = index(i, 0);
            const int jj = index(i, 1);
            const int kk = index(i, 2);

            const auto Pi_offset = Pi_list.binOffset(ii, jj, kk);
            const auto Pi_size   = Pi_list.binSize(ii, jj, kk);

            if (Pi_size == 0) return;

            // Local stencil for interpolation only
            double vel_loc[stencil][stencil][stencil][3];

            for (int si = 0; si < stencil; ++si) {
                for (int sj = 0; sj < stencil; ++sj) {
                    for (int sk = 0; sk < stencil; ++sk) {
                        vel_loc[si][sj][sk][0] = corr_stencil(i, si, sj, sk, 0);
                        vel_loc[si][sj][sk][1] = corr_stencil(i, si, sj, sk, 1);
                        vel_loc[si][sj][sk][2] = corr_stencil(i, si, sj, sk, 2);
                    }
                }
            }

            for (int r = Pi_offset; r < Pi_offset + Pi_size; ++r) {
                const auto p = Pi_list.getParticle(r);

                const double px = positions(p, 0);
                const double py = positions(p, 1);
                const double pz = positions(p, 2);

                const double wx = vorticity_p(p, 0);
                const double wy = vorticity_p(p, 1);
                const double wz = vorticity_p(p, 2);

                double xp[3] = {px, py, pz};
                double x_plus[3] = {
                    px + half_hp * wx,
                    py + half_hp * wy,
                    pz + half_hp * wz
                };
                double x_minus[3] = {
                    px - half_hp * wx,
                    py - half_hp * wy,
                    pz - half_hp * wz
                };

                {
                    double u_temp[3];
                    MLC_Interp::HarmonicValue_local(vel_loc, g, xp, u_temp);
                    velocity_p(p, 0) = u_temp[0];
                    velocity_p(p, 1) = u_temp[1];
                    velocity_p(p, 2) = u_temp[2];
                }

                double u_plus[3];
                MLC_Interp::HarmonicValue_local(vel_loc, g, x_plus, u_plus);

                double u_minus[3];
                MLC_Interp::HarmonicValue_local(vel_loc, g, x_minus, u_minus);

                advect_vort(p, 0) = (u_plus[0] - u_minus[0]) * inv_hp;
                advect_vort(p, 1) = (u_plus[1] - u_minus[1]) * inv_hp;
                advect_vort(p, 2) = (u_plus[2] - u_minus[2]) * inv_hp;
            }
        });
}
 // (3) Full optimized kernel
template <class ExecutionSpace, class NeighborListType, class VortSlice,
          class PosSlice, class USlice, class AdvectVortSlice>
void Interaction_NBody_Split(
    const ExecutionSpace& exec_space,
    PosSlice x_p,
    USlice u_p,
    VortSlice vort_p,
    AdvectVortSlice advect_vorticity,
    NeighborListType& neigh_list,
    const double hp,
    const int numP)
{
    const double inv_4pi    = 0.07957747154594767;
    const double delta      = 0.5 * hp;
    const double delta2     = delta * delta;
    const double delta3inv  = 1.0 / (delta * delta2);
    const double near_const = 0.125 * inv_4pi * delta3inv;
    const double hp_inv     = 1.0 / hp;
    const double half_hp    = 0.5 * hp;

    auto eval_biot_savart = KOKKOS_LAMBDA(
        const double dx, const double dy, const double dz,
        const double uq0, const double uq1, const double uq2,
        const double delta2_in,
        const double inv_4pi_in,
        const double near_const_in,
        double& out0, double& out1, double& out2)
    {
        const double r2 = dx * dx + dy * dy + dz * dz;

        if (r2 >= delta2_in)
        {
            const double rinv  = 1.0 / Kokkos::sqrt(r2);
            const double r3inv = rinv * rinv * rinv;
            const double c     = inv_4pi_in * r3inv;

            out0 = c * (dz * uq1 - dy * uq2);
            out1 = c * (dx * uq2 - dz * uq0);
            out2 = c * (dy * uq0 - dx * uq1);
        }
        else if (r2 > 1e-24)
        {
            const double c = (-12.0 * r2 / delta2_in + 20.0) * near_const_in;

            out0 = c * (dz * uq1 - dy * uq2);
            out1 = c * (dx * uq2 - dz * uq0);
            out2 = c * (dy * uq0 - dx * uq1);
        }
        else
        {
            out0 = 0.0;
            out1 = 0.0;
            out2 = 0.0;
        }
    };


    // Updated Particle Velocity
    auto interaction_u = KOKKOS_LAMBDA(const int p, const int q)
    {
        const double xp0 = x_p(p, 0);
        const double xp1 = x_p(p, 1);
        const double xp2 = x_p(p, 2);

        const double xq0 = x_p(q, 0);
        const double xq1 = x_p(q, 1);
        const double xq2 = x_p(q, 2);

        const double uq0 = vort_p(q, 0);
        const double uq1 = vort_p(q, 1);
        const double uq2 = vort_p(q, 2);

        double b0, b1, b2;
        eval_biot_savart(
            xp0 - xq0, xp1 - xq1, xp2 - xq2,
            uq0, uq1, uq2,
            delta2, inv_4pi, near_const,
            b0, b1, b2);

        u_p(p, 0) += b0;
        u_p(p, 1) += b1;
        u_p(p, 2) += b2;
    };

    Cabana::neighbor_parallel_for(
        Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, numP),
        interaction_u,
        neigh_list,
        Cabana::FirstNeighborsTag(),
        Cabana::SerialOpTag(),
        "Interaction_NBody_u_only");

    // Advectd Vorticity Update
    auto interaction_adv = KOKKOS_LAMBDA(const int p, const int q)
    {
        const double xp0 = x_p(p, 0);
        const double xp1 = x_p(p, 1);
        const double xp2 = x_p(p, 2);

        const double vp0 = vort_p(p, 0);
        const double vp1 = vort_p(p, 1);
        const double vp2 = vort_p(p, 2);

        const double xpp0 = xp0 + half_hp * vp0;
        const double xpp1 = xp1 + half_hp * vp1;
        const double xpp2 = xp2 + half_hp * vp2;

        const double xpm0 = xp0 - half_hp * vp0;
        const double xpm1 = xp1 - half_hp * vp1;
        const double xpm2 = xp2 - half_hp * vp2;

        const double xq0 = x_p(q, 0);
        const double xq1 = x_p(q, 1);
        const double xq2 = x_p(q, 2);

        const double uq0 = vort_p(q, 0);
        const double uq1 = vort_p(q, 1);
        const double uq2 = vort_p(q, 2);

        double ap0, ap1, ap2;
        eval_biot_savart(
            xpp0 - xq0, xpp1 - xq1, xpp2 - xq2,
            uq0, uq1, uq2,
            delta2, inv_4pi, near_const,
            ap0, ap1, ap2);

        double am0, am1, am2;
        eval_biot_savart(
            xpm0 - xq0, xpm1 - xq1, xpm2 - xq2,
            uq0, uq1, uq2,
            delta2, inv_4pi, near_const,
            am0, am1, am2);

        advect_vorticity(p, 0) += (ap0 - am0) * hp_inv;
        advect_vorticity(p, 1) += (ap1 - am1) * hp_inv;
        advect_vorticity(p, 2) += (ap2 - am2) * hp_inv;
    };

    Cabana::neighbor_parallel_for(
        Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, numP),
        interaction_adv,
        neigh_list,
        Cabana::FirstNeighborsTag(),
        Cabana::SerialOpTag(),
        "Interaction_NBody_adv_only");
}
// (3) Optimized deposition with TeamPolicy but has atomics. This takes the same time as the original one but better register pressure
template <class ProblemManagerType, class ExecutionSpace, class NeighborListType, class GridManager>
void Deposition_TeamPolicy_optimized(
    const ExecutionSpace& exec_space,
    const ProblemManagerType& pm,
    const NeighborListType& Pi_list,
    const NeighborListType& Ci_list,
    const GridManager& gridp,
    const int num_grid,
    const int extent,
    const double center,
    const double h,
    const double hp,
    const int corr_radius)
{
    auto index       = gridp.get(Grid::Index());
    auto vorticity_p = pm.get(Location::Particle(), Field::Vorticity());
    auto positions   = pm.get(Location::Particle(), Field::Position());
    auto F           = pm.get(Location::Node(), Field::F());

    Kokkos::deep_copy(F, 0.0);

    const int sz = 2 * corr_radius + 1;
    const double h2_inv = 1.0 / (h * h);

    const double w_center = -128.0 / 30.0;
    const double w_corner =  1.0 / 30.0;
    const double w_edge   =  1.0 / 10.0;
    const double w_face   =  7.0 / 15.0;

    using team_policy = Kokkos::TeamPolicy<ExecutionSpace>;
    using member_type = typename team_policy::member_type;

    // One team per grid cell
    team_policy policy(exec_space, num_grid, Kokkos::AUTO());

    Kokkos::parallel_for(
        "Deposition_TeamPolicy",
        policy,
        KOKKOS_LAMBDA(const member_type& team)
        {
            const int i = team.league_rank();

            const int ii = index(i, 0);
            const int jj = index(i, 1);
            const int kk = index(i, 2);

            const auto offset = Pi_list.binOffset(ii, jj, kk);
            const auto size   = Pi_list.binSize(ii, jj, kk);
            if (size == 0) return;

            int imin, imax, jmin, jmax, kmin, kmax;
            Ci_list.getStencilCells(
                Ci_list.getParticleBin(i),
                imin, imax, jmin, jmax, kmin, kmax);

            const int nx = imax - imin;
            const int ny = jmax - jmin;
            const int nz = kmax - kmin;

            // Guard if runtime stencil and corr_radius disagree
            if (nx > sz || ny > sz || nz > sz) return;

            // Only interior nodes can deposit a 27-point Laplacian
            if (nx < 3 || ny < 3 || nz < 3) return;

            // Number of interior stencil nodes
            const int n_interior = (nx - 2) * (ny - 2) * (nz - 2);

            // Parallelize over interior stencil nodes
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, n_interior),
                [&](const int flat)
                {
                    // Map flat -> local interior index (li, lj, lk)
                    const int li = 1 + flat / ((ny - 2) * (nz - 2));
                    const int rem = flat % ((ny - 2) * (nz - 2));
                    const int lj = 1 + rem / (nz - 2);
                    const int lk = 1 + rem % (nz - 2);

                    const int c0i = imin + li;
                    const int c0j = jmin + lj;
                    const int c0k = kmin + lk;

                    // Accumulate Laplacian pieces directly, component by component.
                    double center_val[3] = {0.0, 0.0, 0.0};
                    double u_face[3]     = {0.0, 0.0, 0.0};
                    double u_edge[3]     = {0.0, 0.0, 0.0};
                    double u_corner[3]   = {0.0, 0.0, 0.0};

                    // Loop over 3x3x3 neighborhood around current interior node
                    for (int di = -1; di <= 1; ++di) {
                        const int ci = c0i + di;
                        for (int dj = -1; dj <= 1; ++dj) {
                            const int cj = c0j + dj;
                            for (int dk = -1; dk <= 1; ++dk) {
                                const int ck = c0k + dk;

                                const int m = (di == 0 ? 0 : 1)
                                                    + (dj == 0 ? 0 : 1)
                                                    + (dk == 0 ? 0 : 1);

                                const double xg_arr[3] = {
                                    ci * h - center,
                                    cj * h - center,
                                    ck * h - center
                                };

                                double accum[3] = {0.0, 0.0, 0.0};

                                // Sum Green's-function contributions from all particles in this cell
                                for (std::size_t r = offset; r < offset + size; ++r) {
                                    const auto p = Pi_list.getParticle(r);

                                    const double xp_arr[3] = {
                                        positions(p, 0), positions(p, 1), positions(p, 2)
                                    };
                                    const double vortp_arr[3] = {
                                        vorticity_p(p, 0), vorticity_p(p, 1), vorticity_p(p, 2)
                                    };

                                    double K[3];
                                    GreensFunction::Calculate_qK_MatVec_Fused(
                                        xg_arr, xp_arr, vortp_arr, K, hp);

                                    accum[0] += K[0];
                                    accum[1] += K[1];
                                    accum[2] += K[2];
                                }

                                if (m == 0) {
                                    center_val[0] = accum[0];
                                    center_val[1] = accum[1];
                                    center_val[2] = accum[2];
                                } else if (m == 1) {
                                    u_face[0] += accum[0];
                                    u_face[1] += accum[1];
                                    u_face[2] += accum[2];
                                } else if (m == 2) {
                                    u_edge[0] += accum[0];
                                    u_edge[1] += accum[1];
                                    u_edge[2] += accum[2];
                                } else { // m == 3
                                    u_corner[0] += accum[0];
                                    u_corner[1] += accum[1];
                                    u_corner[2] += accum[2];
                                }
                            }
                        }
                    }

                    for (int d = 0; d < 3; ++d) {
                        const double result =
                            (center_val[d] * w_center +
                             u_corner[d]   * w_corner +
                             u_edge[d]     * w_edge +
                             u_face[d]     * w_face) * h2_inv;

                        Kokkos::atomic_add(&F(c0i, c0j, c0k, d), result);
                    }
                });
        });
}

// (2) Deposition with no atomic but takes more time than the original
template <class ProblemManagerType, class ExecutionSpace, class NeighborListType, class GridManager>
void Deposition_Colored(
    const ExecutionSpace& exec_space,
    const ProblemManagerType& pm,
    const NeighborListType& Pi_list,
    const NeighborListType& Ci_list,
    const GridManager& gridp,
    const int num_grid,
    const int extent,
    const double center,
    const double h,
    const double hp,
    const int corr_radius)
{
    auto index       = gridp.get(Grid::Index());
    auto vorticity_p = pm.get(Location::Particle(), Field::Vorticity());
    auto positions   = pm.get(Location::Particle(), Field::Position());
    auto F           = pm.get(Location::Node(), Field::F());

    Kokkos::deep_copy(F, 0.0);

    const int period = 2 * corr_radius - 1;

    auto deposition_one_color = KOKKOS_LAMBDA(
        const int i,
        const int cx, const int cy, const int cz,
        const int period_in)
    {
        const int ii = index(i,0);
        const int jj = index(i,1);
        const int kk = index(i,2);

        if ((ii % period_in) != cx || (jj % period_in) != cy || (kk % period_in) != cz)
            return;

        auto offset = Pi_list.binOffset(ii,jj,kk);
    auto size   = Pi_list.binSize(ii,jj,kk);
    if (size == 0) return;

    int imin, imax, jmin, jmax, kmin, kmax;
    Ci_list.getStencilCells(Ci_list.getParticleBin(i),
                            imin, imax, jmin, jmax, kmin, kmax);

    const int sz = 2 * corr_radius + 1;
    if ((imax - imin) > sz || (jmax - jmin) > sz || (kmax - kmin) > sz)
        return;

    // still local, but only for one active color at a time
    double vel_loc[9][9][9][3] = {0.0};   // valid for corr_radius = 4

    double gx[9], gy[9], gz[9];
    for (int s = 0; s < sz; ++s) {
        gx[s] = (imin + s) * h - center;
        gy[s] = (jmin + s) * h - center;
        gz[s] = (kmin + s) * h - center;
    }

    // build local stencil once
    for (std::size_t r = offset; r < offset + size; ++r) {
        const auto p = Pi_list.getParticle(r);

        const double xp_arr[3] = {
            positions(p,0), positions(p,1), positions(p,2)
        };
        const double vortp_arr[3] = {
            vorticity_p(p,0), vorticity_p(p,1), vorticity_p(p,2)
        };

        for (int li = 0; li < (imax - imin); ++li) {
            for (int lj = 0; lj < (jmax - jmin); ++lj) {
                for (int lk = 0; lk < (kmax - kmin); ++lk) {
                    const double xg_arr[3] = { gx[li], gy[lj], gz[lk] };
                    double K[3];

                    GreensFunction::Calculate_qK_MatVec_Fused(
                        xg_arr, xp_arr, vortp_arr, K, hp);

                    vel_loc[li][lj][lk][0] += K[0];
                    vel_loc[li][lj][lk][1] += K[1];
                    vel_loc[li][lj][lk][2] += K[2];
                }
            }
        }
    }

    const double h2_inv = 1.0 / (h * h);
    const double w_center = -128.0 / 30.0;
    const double w_corner =  1.0 / 30.0;
    const double w_edge   =  1.0 / 10.0;
    const double w_face   =  7.0 / 15.0;

    // deposit WITHOUT atomics because this color is non-overlapping
    for (int li = 1; li < (imax - imin) - 1; ++li) {
        for (int lj = 1; lj < (jmax - jmin) - 1; ++lj) {
            for (int lk = 1; lk < (kmax - kmin) - 1; ++lk) {

                const int c0i = imin + li;
                const int c0j = jmin + lj;
                const int c0k = kmin + lk;

                for (int d = 0; d < 3; ++d) {
                    const double center_val = vel_loc[li][lj][lk][d];

                    const double u_face =
                        vel_loc[li+1][lj][lk][d] + vel_loc[li-1][lj][lk][d] +
                        vel_loc[li][lj+1][lk][d] + vel_loc[li][lj-1][lk][d] +
                        vel_loc[li][lj][lk+1][d] + vel_loc[li][lj][lk-1][d];

                    const double u_edge =
                        vel_loc[li+1][lj+1][lk][d] + vel_loc[li+1][lj-1][lk][d] +
                        vel_loc[li-1][lj+1][lk][d] + vel_loc[li-1][lj-1][lk][d] +
                        vel_loc[li+1][lj][lk+1][d] + vel_loc[li+1][lj][lk-1][d] +
                        vel_loc[li-1][lj][lk+1][d] + vel_loc[li-1][lj][lk-1][d] +
                        vel_loc[li][lj+1][lk+1][d] + vel_loc[li][lj+1][lk-1][d] +
                        vel_loc[li][lj-1][lk+1][d] + vel_loc[li][lj-1][lk-1][d];

                    const double u_corner =
                        vel_loc[li+1][lj+1][lk+1][d] + vel_loc[li+1][lj+1][lk-1][d] +
                        vel_loc[li+1][lj-1][lk+1][d] + vel_loc[li+1][lj-1][lk-1][d] +
                        vel_loc[li-1][lj+1][lk+1][d] + vel_loc[li-1][lj+1][lk-1][d] +
                        vel_loc[li-1][lj-1][lk+1][d] + vel_loc[li-1][lj-1][lk-1][d];

                    const double result =
                        (center_val * w_center +
                         u_corner   * w_corner +
                         u_edge     * w_edge +
                         u_face     * w_face) * h2_inv;

                    F(c0i,c0j,c0k,d) += result;   // no atomic
                }
            }
        }
    }
    };

    for (int cx = 0; cx < period; ++cx) {
        for (int cy = 0; cy < period; ++cy) {
            for (int cz = 0; cz < period; ++cz) {
                Kokkos::parallel_for(
                    "Deposition_Colored_Phase",
                    Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_grid),
                    KOKKOS_LAMBDA(const int i) {
                        deposition_one_color(i, cx, cy, cz, period);
                    });
                Kokkos::fence();
            }
        }
    }
}

    //Get vorticity, postion and velocity on
} // end namespace LocalCorrection
} // end namespace MLC

#endif // MLC_LocalCorrection_HPP
