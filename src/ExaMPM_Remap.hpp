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

#ifndef EXAMPM_REMAP_HPP
#define EXAMPM_REMAP_HPP

#include <ExaMPM_MLC_Interp.hpp>
#include <ExaMPM_ProblemManager2.hpp>
#include <Cabana_Grid.hpp>
#include <ExaMPM_GridManager.hpp>
#include <Kokkos_Core.hpp>
#include <cmath>
#include <vector>
#include <array>
namespace ExaMPM
{
namespace Remap
{
//---------------------------------------------------------------------------//
// Particle-to-grid.
//
struct RemapInitFunc
{
    template <class ParticleType>
    KOKKOS_INLINE_FUNCTION bool operator()( const double x[3],
		                            const double vort[3],
                                            ParticleType& p ) const
    {

       double vort_magn = pow( pow( vort[0] , 2.0) +
	                       pow( vort[1] , 2.0) +
		               pow( vort[2] , 2.0), 0.5);

<<<<<<< HEAD

       if ( vort_magn > pow(10,-6.0) ) 
=======
       if ( vort_magn > pow(10,-4.0) ) 
>>>>>>> origin/FFTX_GreensFunction
       {
   //        std::cout << " vorticity magn " << std::endl;
	   for(int d = 0; d < 3; d++)    
	   {  
              Cabana::get<0>( p, d ) = vort[d]; 
	      Cabana::get<1>( p, d ) = 0.0;
              Cabana::get<2>( p, d ) = x[d];
<<<<<<< HEAD
	      Cabana::get<3>( p, d ) = 0.0;
           }

	   Kokkos::printf( "vorticity %f  x %f y %f z %f \n", vort[0], Cabana::get<2>(p,0), Cabana::get<2>(p,1), Cabana::get<2>(p,2) );
=======
           }

>>>>>>> origin/FFTX_GreensFunction
	   return true;

       }
       
       return false;
	      
   }
};
<<<<<<< HEAD
KOKKOS_INLINE_FUNCTION void W44_Weight(double W44[3], double x_g[3], double x_p[3], double hg, double hp)
{
    int n = 10;	
    double a[10], b[10], g[10];
=======
void W44_Weight(double W44[3], double x_g[3], double x_p[3], double hg, double hp)
{
    int n = 10;	
    double a[n], b[n], g[n];
>>>>>>> origin/FFTX_GreensFunction
    double ratio = pow(hp/hg, 3.0);
    double d[3];
    int DIM = 3;

    //Coefficients for interpolation
    a[0] = 1.0; a[1] = 0; a[2] = -5.0/4.0;
    a[3] = 0.0; a[4] = 1.0/4.0; a[5] = -100.0/3.0;
    a[6] = 455.0/4.0; a[7] = -295.0/2.0;
    a[8] = 345.0/4.0; a[9]= -115.0/6.0;

    b[0] = -199.0; b[1] = 5485.0/4.0; b[2] = -32975.0/8.0;
    b[3] = 28425.0/4.0; b[4] = -61953.0/8.0;
    b[5] = 33175/6.0; b[6] = -20685.0/8.0;
    b[7] = 3055.0/4.0; b[8] = -1035.0/8.0;
    b[9] = 115.0/12.0;

    g[0] = 5913.0; g[1] = -89235.0/4.0;
    g[2] = 297585.0/8.0; g[3] = -143895.0/4.0;
    g[4] = 177871.0/8.0; g[5] = -54641.0/6.0;
    g[6] = 19775.0/8.0; g[7] = -1715.0/4.0;
    g[8]  = 345.0/8.0; g[9] = -23.0/12.0;

     for(int j =0; j < DIM; j++)
     {

           //difference between particle location and stencil location
           d[j] = ( std::abs((x_g[j] - x_p[j]) / hg ) );

<<<<<<< HEAD
	 //  Kokkos::printf( " dj %f \n", d[j]);
=======
>>>>>>> origin/FFTX_GreensFunction
           W44[j] = 0.0;

           //Generate W44 based on distance d
           if((d[j] < 1.0)){

             for(int p = (n-1); p > -1; p--){
                 W44[j] = W44[j]*d[j] + a[p];
             }

           } else if( (d[j] >= 1.0) && (d[j] < 2.0) ){


            for(int p = (n-1); p > -1; p--){
                  W44[j] = W44[j]*d[j] + b[p];
              }
           } else if( (d[j] >= 2.0) && (d[j] < 3.0) ){


               for(int p = (n-1); p > -1; p--){
                  W44[j] = W44[j]*d[j] + g[p];
               }

           } else{

             W44[j] = 0.0;
           }

     }	   


}

template <class ProblemManagerType, class ExecutionSpace, class NeighborListType>
void W44( const ExecutionSpace& exec_space, ProblemManagerType& pm,
                        const NeighborListType& W44_list, const double center, const double h, const double hp)
{

   //Get vorticity, velocity, and positions of real particles
   auto vorticity_p = pm.get(Location::Particle(), Field::Vorticity());
   auto velocity_p = pm.get(Location::Particle(), Field::Velocity());
   auto vorticity_g = pm.get(Location::Node(), Field::Vorticity_hp());
   auto positions  = pm.get(Location::Particle(), Field::Position());
   double ratio = pow(h/hp, 3.0);
   Kokkos::deep_copy(vorticity_g, 0.0);

   std::cout << " ratio " << ratio << std::endl;

   //Iterate over D0 
   Kokkos::parallel_for(
        "W44",
        Kokkos::RangePolicy<ExecutionSpace>( exec_space, 0, pm.numParticle() ),
        KOKKOS_LAMBDA( const int p ) {
	  
     
            //Get Particle Position
	    double xp[3] = { positions(p, 0 ), positions( p, 1 ), positions( p, 2 ) };

	    //Get W44 upper and lower bounds
	    // getParticleBin(i) gives the cell/bin associated with the particle of interest
	    // The max and min values of the stencil is built around this bin
            int imin, imax, jmin, jmax, kmin, kmax;
            W44_list.getStencilCells( W44_list.getParticleBin( p ), imin,imax, jmin,
                                      jmax, kmin, kmax ); 


            //Iterate over cell stencil of the linked list = Ci
            for( int i = imin; i < imax; i++)
                for( int j = jmin; j < jmax; j ++)
                    for( int k = kmin; k < kmax; k ++)
                     {

			  double weights[3];   
                          double xg[3] = {i*h - center, j*h - center, k*h - center};
         //               Kokkos::printf(" xp %f xg %f yp %f yg %f zp %f zg %f \n", xp[0],xp[1],xp[2],xg[0],xg[1],xg[2]);
			  //Calculate Weights
                          W44_Weight(weights, xg, xp, h, hp);
			  if( std::abs(vorticity_p(p,0) * ratio * ( weights[0]*weights[1]*weights[2] )) > 0.001){
                                  Kokkos::printf(" pre VORT %f \n",vorticity_g(i,j,k,0));
                          }

                          for(int d = 0; d < 3; d++)
			     vorticity_g(i,j,k,d) += vorticity_p(p,d) * ratio * ( weights[0]*weights[1]*weights[2] );
			  if( std::abs(vorticity_p(p,0) * ratio * ( weights[0]*weights[1]*weights[2] )) > 0.001){
		         	  Kokkos::printf(" addition %f VORT %f \n", vorticity_p(p,0) * ratio * ( weights[0]*weights[1]*weights[2] ),vorticity_g(i,j,k,0));
				  Kokkos::printf("i %d j %d k %d numParticle %d \n ", i, j, k, p);
                                  Kokkos::printf(" xp %f yp %f zp %f \n", xp[0], xp[1], xp[2]);
                          }

                     }



        });


       pm.Resize_Remap( exec_space, RemapInitFunc());
          

       std::cout << "resized remap" << std::endl;
          

}

template <class ProblemManagerType, class ExecutionSpace, class NeighborListType>
void Test_Remap( const ExecutionSpace& exec_space, ProblemManagerType& pm,
                        const NeighborListType& W44_list, const double center, const double h, const double hp)
{
          
	  std::cout << " pre-remap num p " << pm.numParticle() << std::endl;

          W44( exec_space, pm, W44_list, center, hp, hp);
          auto vorticity_p = pm.get(Location::Particle(), Field::Vorticity());
          auto velocity_p = pm.get(Location::Particle(), Field::Velocity());
          auto vorticity_g = pm.get(Location::Node(), Field::Vorticity_hp());
          auto positions  = pm.get(Location::Particle(), Field::Position());
 
	  std::cout << " post-remap num p " << pm.numParticle() << std::endl;

	  for(int p = 0; p < pm.numParticle(); p++)
	  {

            std::cout << "x = " << positions(p,0) << " y = "
		      << positions(p,1) << " z = " 
		      << positions(p,2) << std::endl;

	    std::cout << " vort " << vorticity_p(p,0) << " " << vorticity_p(p,1) << " "
		      << vorticity_p(p,2) << std::endl;

	  }
	  
}
template <class ProblemManagerType, class ExecutionSpace, class NeighborListType>
void Test_Remap_Particles( const ExecutionSpace& exec_space, ProblemManagerType& pm,
                        const NeighborListType& W44_list, const double center, const double h, const double hp)
{

	auto vorticity_g = pm.get(Location::Node(), Field::Vorticity_hp());

	Kokkos::deep_copy(vorticity_g, 0.0);

	int i = 1, j = 1, k = 1;

	Kokkos::parallel_for(
        "one",
        Kokkos::RangePolicy<ExecutionSpace>( exec_space, 0, 3 ),
        KOKKOS_LAMBDA( const int p ) {

	   if( p == 1 ){
             for(int d = 0; d < 3; d++)
	     {
 	       vorticity_g(4,4,4,d) += 1.0;
               vorticity_g(5,5,5,d) += 1.0;
	     }
	     Kokkos::printf("vorticity 1 1 1 %f \n ", vorticity_g(4,4,4,0) );
	   }
	   });   
//orticity_g(0,0,0,0) = 1.0;
//td::cout << vorticity_g(i,j,k,0) << " vorticity 1 1 1 " << std::endl;
        Kokkos::printf(" OG Position %f \n ", i*hp - center );
//        std::cout << "OG position " << i*hp - center << std::endl;
	pm.Resize_Remap( exec_space, RemapInitFunc());
        auto vorticity_p = pm.get(Location::Particle(), Field::Vorticity());
        auto velocity_p = pm.get(Location::Particle(), Field::Velocity());
        auto positions  = pm.get(Location::Particle(), Field::Position());

//	std::cout << "particle size" << pm.numParticle() << std::endl;
        Kokkos::printf(" particle size %d \n ", pm.numParticle() );

	Kokkos::parallel_for(
        "one_2",
        Kokkos::RangePolicy<ExecutionSpace>( exec_space, 0, pm.numParticle() ),
        KOKKOS_LAMBDA( const int p ) {
            Kokkos::printf(" vorticity %f %f %f \n", vorticity_p(p,0),vorticity_p(p,1),vorticity_p(p,2) );
            Kokkos::printf(" velocity %f %f %f \n", velocity_p(p,0),velocity_p(p,1),velocity_p(p,2) );	    
	    Kokkos::printf(" positions %f %f %f \n", positions(p,0),positions(p,1),positions(p,2) );
//   	    std::cout << " vorticity " << vorticity_p(p,0) << " " << vorticity_p(p,1) << " " << vorticity_p(p,2) << std::endl;
//   std::cout << " velocity "  << velocity_p(p,0)  << " " << velocity_p(p,1)  << " " << velocity_p(p,2)  << std::endl;
//   std::cout << " positions " << positions(p,0)   << " " << positions(p,1)   << " " << positions(p,2)   << std::endl;

	});
}

} // end namespace REMAP
} // end namespace ExaMPM

#endif // EXAMPM_REMAP_HPP
