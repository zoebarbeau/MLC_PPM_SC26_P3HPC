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

#ifndef EXAMPM_SOLVER2_HPP
#define EXAMPM_SOLVER2_HPP

#include <ExaMPM_BoundaryConditions.hpp>
#include <ExaMPM_Mesh.hpp>
#include <ExaMPM_ProblemManager2.hpp>
#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>
#include <ExaMPM_LocalCorrection.hpp>
#include <ExaMPM_GridManager.hpp>
#include <ExaMPM_Remap.hpp>
#include <memory>
#include <string>
#include <ExaMPM_Remap.hpp>
#include <ExaMPM_RK4.hpp>
#include <mpi.h>
#include "FFTXLGFConvolution.H"
namespace ExaMPM
{
//---------------------------------------------------------------------------//
class SolverBase
{
  public:
    virtual ~SolverBase() = default;
    virtual void solve( const double t_final, const int write_freq, const double center, const int c, const double cell_size, const double hp ) = 0;
};

//---------------------------------------------------------------------------//
template <class MemorySpace, class ExecutionSpace>
class Solver : public SolverBase
{
  public:
    using ListType = Cabana::LinkedCellList<MemorySpace,double>;	 
    
    template <class InitFunc>
    Solver( MPI_Comm comm, const Kokkos::Array<double, 6>& global_bounding_box,
            const std::array<int, 3>& global_num_cell,
	    const std::array<int, 3>& pgrid_num_cell,
            const std::array<bool, 3>& periodic,
            const Cabana::Grid::BlockPartitioner<3>& partitioner,
            const int halo_cell_width, const InitFunc& create_functor,
            const int particles_per_cell, const double cell_size, const double hp,
            const double center, BoundaryCondition& bc )
        : _dt( 0.001 )
        , _time( 0.0 )
        , _step( 0 )
        , _bc( bc )
        , _halo_min( 0 )
	, _center( center)
    {

	 //Width of the domain
         extent = global_num_cell[0];
         extentp = pgrid_num_cell[0];
        _mesh = std::make_shared<Mesh<MemorySpace>>(
            global_bounding_box, global_num_cell, periodic, partitioner,
            halo_cell_width, _halo_min, comm );

	_pmesh = std::make_shared<Mesh<MemorySpace>>(
            global_bounding_box, pgrid_num_cell, periodic, partitioner,
            halo_cell_width, _halo_min, comm );

        _bc.min = _mesh->minDomainGlobalNodeIndex();
        _bc.max = _mesh->maxDomainGlobalNodeIndex();


        _pm = std::make_shared<ProblemManager<MemorySpace>>(
            ExecutionSpace(), _mesh, _pmesh, create_functor, particles_per_cell,
	    cell_size, _center, hp, extent,extentp);

	double grid_min[3] = { 0,
                               0,
                               0 };
        double grid_max[3] = { 1,
                               1,
                               1 };

        L = grid_max[0] - grid_min[0];

        double grid_delta[3] = {cell_size, cell_size, cell_size};

        double pgrid_delta[3] = {hp,hp,hp};

	auto positions = _pm->get( Location::Particle(), Field::Position() );
        // 
        corr_radius = 4;
        //Real Particle Lists
	//5x5x5 linked cell stencil
        _neigh_list = std::make_shared<Cabana::LinkedCellList<MemorySpace,double>>(positions,0, _pm->numParticle(),grid_delta,grid_min,grid_max,corr_radius*cell_size, 1.0/4.0);

	//1x1x1 linked cell stencil
        _oneGrid_list = std::make_shared<Cabana::LinkedCellList<MemorySpace,double>>(positions,0, _pm->numParticle(),grid_delta,grid_min,grid_max);

	//7x7 W44 stencil linked list
	//
	//

	double third = 1.0 / 3.0;
	_W44_list = std::make_shared<Cabana::LinkedCellList<MemorySpace,double>>(positions,0, _pm->numParticle(), pgrid_delta, grid_min, grid_max, 3*hp, third);

	//Define the number of points contained in D0 and D
	num_D0 = (global_num_cell[0] + 1 - 2)*(global_num_cell[1] + 1 - 2)*(global_num_cell[2] + 1 - 2);
        num_D  = (global_num_cell[0]+1)*(global_num_cell[1]+1)*(global_num_cell[2]+1);

	int nump = _pm->numParticle();
	//Fake Grid Particle Lists
	_gridp = std::make_shared<GridManager<MemorySpace>>(ExecutionSpace(),*(_mesh->localGrid()),positions,nump,num_D0, extent,cell_size,center);
	LocalCorrection::update_GridList(ExecutionSpace(),*(_mesh->localGrid()),*_pm, *_gridp, nump, num_D0, extent, cell_size, center);

	auto gridpositions = _gridp->get( Grid::Position() );
     //   std::cout << "get pos nump " << nump << " numDO " << num_D0 << " cellsize " << cell_size <<" GLOBAL NUM CELL " << global_num_cell[0] << std::endl;
	// 5x5x5 grid particle list

        _Ci_grid_list = std::make_shared<Cabana::LinkedCellList<MemorySpace,double>>(gridpositions,0, nump+num_D0,grid_delta,grid_min,grid_max,corr_radius*cell_size, 1.0/4.0);
       
        //1x1x1 grid particle list
        _Pi_grid_list = std::make_shared<Cabana::LinkedCellList<MemorySpace,double>>(gridpositions,0, nump+num_D0,grid_delta,grid_min,grid_max,cell_size, 1.0);

	MPI_Comm_rank( comm, &_rank );
    }

    void solve( const double t_final, const int write_freq, const double center, const int c, const double cell_size, const double hp )
    {   


        ConvolutionFFTX<ExecutionSpace> FFTXConv(extent,center,cell_size);
        std::string file="/global/homes/z/zbarbeau/ExaMPM/LatticeGreensFunction/exec/G_256_Octant";

        FFTXConv.read_LGF_file(file);
        // Output initial state.
//       outputParticles();
        
       _time = 0;
       _dt   =0.0001953125;
       double multiply[4]  = {0.5,0.5,1.0,0.0};
       double increment[4] = {1.0/6.0,1.0/3.0,1.0/3.0,1.0/6.0};
       while( _time <1* _dt ){

//          _pm->initRK4();
//          RK4::updateP(ExecutionSpace(),*_pm);
          for(int i = 0; i < 1; i++){

             Kokkos::Timer timer;
             LocalCorrection::Deposition(ExecutionSpace(), *_pm, *_Pi_grid_list,*_Ci_grid_list,*_gridp,num_D0,extent,center,cell_size,hp,corr_radius);
             Kokkos::fence();
             double timeD = timer.seconds();
	     timer.reset();

             FFTXConv.compute_convolution(ExecutionSpace(), *_pm);
	     Kokkos::fence();
	     double timeF = timer.seconds();
             std::stringstream name;
             name << 2 << "_velocity";
             const std::string prefix = name.str();
             Cabana::Grid::Experimental::BovWriter::writeTimeStep(ExecutionSpace(),prefix,1,0,  *(_pm->_velx));

             timer.reset();
             LocalCorrection::Corrections(ExecutionSpace(), *_pm, *_Ci_grid_list,*_Pi_grid_list,*_gridp,num_D0,
                     extent,center,cell_size, hp, corr_radius);
             Kokkos::fence();
             double timeCorr = timer.seconds();
             timer.reset();
             LocalCorrection::Interaction_NBody(ExecutionSpace(), *_pm, *_neigh_list, c, center, cell_size, hp, corr_radius );  
             Kokkos::fence();
             double timeInt = timer.seconds();
          
//             RK4::increment(ExecutionSpace(), *_pm, _dt, i, multiply[i], increment[i]);
 
	     if( i == 0 ){

                LocalCorrection::Error_V( ExecutionSpace(), *_pm, extent, cell_size, hp,*(_mesh->localGrid()));

	     }
             std::cout << timeD << " "<< timeF << " " << timeCorr << " " << timeInt << " " << std::endl;


       }

         _time += _dt; 
         _step += 1;
//         outputParticles(); 
     }
     

     LocalCorrection::Error_V( ExecutionSpace(), *_pm, extent, cell_size, hp,*(_mesh->localGrid()));
        
//         LocalCorrection::Error_V2( ExecutionSpace(), *_pm, extent, cell_size, hp);
      
/*       for(int d = 0; d < 3; d++){

          Remap::VelG_Error( ExecutionSpace(),*_pm,*_W44_list,center,hp,hp,d);

       }

*/



    }


    void outputParticles()
    {

    	    // Prefer HDF5 output over Silo. Only output if one is enabled.
#ifdef Cabana_ENABLE_HDF5
        Cabana::Experimental::HDF5ParticleOutput::HDF5Config h5_config;
        Cabana::Experimental::HDF5ParticleOutput::writeTimeStep(
            h5_config,"single_particle", _pmesh->localGrid()->globalGrid().comm(),
            _step, _time, _pm->numParticle(),
            _pm->get( Location::Particle0(), Field::Position() ),
            _pm->get( Location::Particle0(), Field::Vorticity() ),
            _pm->get( Location::Particle0(), Field::Velocity() ));
#else
#ifdef Cabana_ENABLE_SILO
        Cabana::Grid::Experimental::SiloParticleOutput::writeTimeStep(
            "remap", _pmesh->localGrid()->globalGrid(), _step, _time,
            _pm->get( Location::Particle(), Field::Position() ),
            _pm->get( Location::Particle(), Field::Vorticity() ));

#else
        if ( _rank == 0 )
            std::cout << "No particle output enabled in Cabana. Add "
                         "Cabana_REQUIRE_HDF5=ON or Cabana_REQUIRE_SILO=ON to "
                         "the Cabana build if needed.";
#endif
#endif
    }

  private:
    double _dt;
    double _time;
    int _step, corr_radius;
    BoundaryCondition _bc;
    int _halo_min;
    std::shared_ptr<Mesh<MemorySpace>> _mesh, _pmesh;
    std::shared_ptr<ProblemManager<MemorySpace>> _pm;
    std::shared_ptr<GridManager<MemorySpace>> _gridp;
    std::shared_ptr<Cabana::LinkedCellList<MemorySpace,double>> _neigh_list, _Ci_grid_list, _W44_list;
    std::shared_ptr<Cabana::LinkedCellList<MemorySpace,double>> _oneGrid_list, _Pi_grid_list;
    int _rank;
    int num_D0;
    int num_D;
    int extent,extentp;
    double _center;
    double L;
};

//---------------------------------------------------------------------------//
// Creation method.
template <class InitFunc>
std::shared_ptr<SolverBase>
createSolver( const std::string& exec_space, MPI_Comm comm,
              const Kokkos::Array<double, 6>& global_bounding_box,
              const std::array<int, 3>& global_num_cell,
	      const std::array<int, 3>& pgrid_num_cell,
              const std::array<bool, 3>& periodic,
              const Cabana::Grid::BlockPartitioner<3>& partitioner,
              const int halo_cell_width, const InitFunc& create_functor,
              const int particles_per_cell, const double cell_size, const double hp, const double center,
	      BoundaryCondition& bc)
{
    if ( 0 == exec_space.compare( "serial" ) ||
         0 == exec_space.compare( "Serial" ) ||
         0 == exec_space.compare( "SERIAL" ) )
    {
#ifdef KOKKOS_ENABLE_SERIAL
        return std::make_shared<
            ExaMPM::Solver<Kokkos::HostSpace, Kokkos::Serial>>(
            comm, global_bounding_box, global_num_cell, pgrid_num_cell, periodic, partitioner,
            halo_cell_width, create_functor, particles_per_cell, cell_size, hp, center, bc );
#else
        throw std::runtime_error( "Serial Backend Not Enabled" );
#endif
    }
    else if ( 0 == exec_space.compare( "openmp" ) ||
              0 == exec_space.compare( "OpenMP" ) ||
              0 == exec_space.compare( "OPENMP" ) )
    {
#ifdef KOKKOS_ENABLE_OPENMP
        return std::make_shared<
            ExaMPM::Solver<Kokkos::HostSpace, Kokkos::OpenMP>>(
            comm, global_bounding_box, global_num_cell,pgrid_num_cell, periodic, partitioner,
            halo_cell_width, create_functor, particles_per_cell, cell_size, hp, center, bc );
#else
        throw std::runtime_error( "OpenMP Backend Not Enabled" );
#endif
    }
    else if ( 0 == exec_space.compare( "cuda" ) ||
              0 == exec_space.compare( "Cuda" ) ||
              0 == exec_space.compare( "CUDA" ) )
    {
#ifdef KOKKOS_ENABLE_CUDA
        return std::make_shared<
            ExaMPM::Solver<Kokkos::CudaSpace, Kokkos::Cuda>>(
            comm, global_bounding_box, global_num_cell, pgrid_num_cell, periodic, partitioner,
            halo_cell_width, create_functor, particles_per_cell, cell_size, hp, center, bc );
#else
        throw std::runtime_error( "CUDA Backend Not Enabled" );
#endif
    }
    else if ( 0 == exec_space.compare( "hip" ) ||
              0 == exec_space.compare( "Hip" ) ||
              0 == exec_space.compare( "HIP" ) )
    {
#ifdef KOKKOS_ENABLE_HIP
        return std::make_shared<ExaMPM::Solver<Kokkos::Experimental::HIPSpace,
                                               Kokkos::Experimental::HIP>>(
            comm, global_bounding_box, global_num_cell, pgrid_num_cell, periodic, partitioner,
            halo_cell_width, create_functor, particles_per_cell, cell_size, hp, center, bc );
#else
        throw std::runtime_error( "HIP Backend Not Enabled" );
#endif
    }
    else
    {
        throw std::runtime_error( "invalid backend" );
        return nullptr;
    }
}

//---------------------------------------------------------------------------//

} // end namespace ExaMPM

#endif // end EXAMPM_SOLVER2_HPP
