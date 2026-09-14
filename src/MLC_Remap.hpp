/* This file is written by the authors specifically for MLC */
#ifndef MLC_REMAP_HPP
#define MLC_REMAP_HPP

#include <MLC_MLC_Interp.hpp>
#include <MLC_ProblemManager2.hpp>
#include <Cabana_Grid.hpp>
#include <MLC_GridManager.hpp>
#include <Kokkos_Core.hpp>
#include <cmath>
#include <vector>
#include <array>
namespace MLC
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

       if ( vort_magn > pow(10,-4.0) ) 
       {
	   for(int d = 0; d < 3; d++)    
	   {  
              Cabana::get<0>( p, d ) = vort[d]; 
	      Cabana::get<1>( p, d ) = 0.0;
              Cabana::get<2>( p, d ) = x[d];

	   }   
	   return true;

       }
       
       return false;
	      
   }
};
KOKKOS_INLINE_FUNCTION void W44_Weight(double W44[3], double x_g[3], double x_p[3], double hg, double hp)
{
    int n = 10;	
    double a[10], b[10], g[10];
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
                          W44_Weight(weights, xg, xp, h, hp);
                          for(int d = 0; d < 3; d++)
			     vorticity_g(i,j,k,d) += vorticity_p(p,d) * ratio * ( weights[0]*weights[1]*weights[2] );
                     }



        });


       pm.Resize_Remap( exec_space, RemapInitFunc());
          

       std::cout << "resized remap" << std::endl;
          

}
} // end namespace REMAP
} // end namespace MLC

#endif // MLC_REMAP_HPP
