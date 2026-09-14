/* This file is written by the authors specifically for MLC */
#ifndef MLC_RK4_HPP
#define MLC_RK4_HPP

#include <MLC_ProblemManager2.hpp>
#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>
#include <cmath>
namespace MLC
{
namespace RK4
{

template <class ProblemManagerType, class ExecutionSpace>
void updateP( const ExecutionSpace& exec_space, const ProblemManagerType& pm)
{

    auto x_p = pm.get( Location::Particle(), Field::Position() );
    auto x_p0 = pm.get( Location::Particle0(), Field::Position() );

    auto vort_p = pm.get( Location::Particle(), Field::Vorticity() );
    auto vort_p0 = pm.get( Location::Particle0(), Field::Vorticity() );

    auto advect_vorticity = pm.get( Location::Particle(), Field::Vorticity_Advect() );
    auto u_p = pm.get( Location::Particle(), Field::Velocity() );

    Kokkos::parallel_for(
        "updateQn",
        Kokkos::RangePolicy<ExecutionSpace>( exec_space, 0, pm.numParticle() ),
        KOKKOS_LAMBDA( const int p ) {

           for(int d = 0; d < 3; d++){


             x_p(p,d)   = x_p0(p,d);
             vort_p(p,d) = vort_p0(p,d);


           }


//          Kokkos::printf(" x_p %f vort_p %f Qn \n", x_p(p,0),vort_p(p,0) );


        });

}
template <class ProblemManagerType, class ExecutionSpace>
void updateQn( const ExecutionSpace& exec_space, const ProblemManagerType& pm)
{

    auto x_p = pm.get( Location::Particle(), Field::Position() );
    auto x_p0 = pm.get( Location::Particle0(), Field::Position() );
    auto x_pk = pm.get( Location::ParticleK(), Field::Position() );

    auto vort_p = pm.get( Location::Particle(), Field::Vorticity() );
    auto vort_p0 = pm.get( Location::Particle0(), Field::Vorticity() );    
    auto vort_pk = pm.get( Location::ParticleK(), Field::Vorticity() );
  
    auto advect_vorticity = pm.get( Location::Particle(), Field::Vorticity_Advect() );
    auto u_p = pm.get( Location::Particle(), Field::Velocity() );
       
    Cabana::deep_copy(u_p,0.0);
    Cabana::deep_copy(advect_vorticity,0.0);

    Kokkos::parallel_for(
        "updateQn",
        Kokkos::RangePolicy<ExecutionSpace>( exec_space, 0, pm.numParticle() ),
        KOKKOS_LAMBDA( const int p ) {

           for(int d = 0; d < 3; d++){


             x_p(p,d)   += x_p0(p,d);
             vort_p(p,d) += vort_p0(p,d);

           
           } 


//          Kokkos::printf(" x_p %f vort_p %f Qn \n", x_p(p,0),vort_p(p,0) );


        });

}


template <class ProblemManagerType, class ExecutionSpace>
void increment( const ExecutionSpace& exec_space, const ProblemManagerType& pm, double dt, int i, double k_multiply, double k_increment)
{

    auto x_p = pm.get( Location::Particle(), Field::Position() );
    auto x_p0 = pm.get( Location::Particle0(), Field::Position() );
    auto x_pk = pm.get( Location::ParticleK(), Field::Position() );

    auto vort_p = pm.get( Location::Particle(), Field::Vorticity() );
    auto vort_p0 = pm.get( Location::Particle0(), Field::Vorticity() );
    auto vort_pk = pm.get( Location::ParticleK(), Field::Vorticity() );

    auto advect_vorticity = pm.get( Location::Particle(), Field::Vorticity_Advect() );
    auto u_p = pm.get( Location::Particle(), Field::Velocity() );
    auto u_p0 = pm.get( Location::Particle0(), Field::Velocity() );

     Kokkos::parallel_for(
         "increment",
         Kokkos::RangePolicy<ExecutionSpace>( exec_space, 0, pm.numParticle() ),
         KOKKOS_LAMBDA( const int p ) {

            for(int d = 0; d < 3; d++){


             x_p(p,d)    =x_p0(p,d)+k_multiply*dt*u_p(p,d);
             vort_p(p,d) = vort_p0(p,d)+k_multiply*dt*advect_vorticity(p,d);
             x_pk(p,d)   += k_increment*dt*u_p(p,d);
             vort_pk(p,d) += k_increment*dt*advect_vorticity(p,d);
            

           }
     });


     
     if( i == 3 ){


         Kokkos::parallel_for(
         "final",
         Kokkos::RangePolicy<ExecutionSpace>( exec_space, 0, pm.numParticle() ),
         KOKKOS_LAMBDA( const int p ) {

            for(int d = 0; d < 3; d++){


             x_p0(p,d)   += x_pk(p,d);
             vort_p0(p,d) += vort_pk(p,d);
             u_p0(p,d)   = u_p(p,d);
             x_p(p,d)    = x_p0(p,d); 
             vort_p(p,d) = vort_p0(p,d);
             
             x_pk(p,d)    = 0.0;
             vort_pk(p,d) = 0.0;
             
             advect_vorticity(p,d) = 0.0;



           }

        });





     }



   

}


} // end namespace LocalCorrection
} // end namespace MLC

#endif // MLC_LocalCorrection_HPP
