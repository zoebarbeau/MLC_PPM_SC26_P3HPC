#include <Cabana_Core.hpp>
#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>
#include <mpi.h>
#include <memory.h>

#include <array>
#include <cmath>

//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//
void initgrid(const double cell_size, const int ppc, const int halo_size,
               const std::string& exec_space, const double vorticity )
{
    // Domain
    Kokkos::Array<double, 6> global_box = { -0.75, -0.75, -0.75, 0.75, 0.75, 0.75 };
    double center = 0.75;
    int c = 2;

    // Compute the number of cells in each direction. The user input must
    // squarely divide the domain.
    std::array<int, 3> global_num_cell = {
        static_cast<int>( 1.5 / cell_size ),
        static_cast<int>( 1.5 / cell_size ),
        static_cast<int>( 1.5 / cell_size ) };

    double grid_min[3] = { -0.75, -0.75, -0.75 };
    double grid_max[3] = { 0.75, 0.75, 0.75 };
    double grid_delta[3] = { cell_size, cell_size, cell_size };
    // Due to the 2D nature of the problem we will only partition in Y. The
    // behavior of the fluid will be to largely just run out in X and Z with
    // little movement in Y.
    int comm_size;
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );
    std::array<int, 3> ranks_per_dim = { 1, comm_size, 1 };
    Cabana::Grid::ManualBlockPartitioner<3> partitioner( ranks_per_dim );

 /*   //Determine Execution and Memory space
    if ( 0 == exec_space.compare( "serial" ) ||
         0 == exec_space.compare( "Serial" ) ||
         0 == exec_space.compare( "SERIAL" ) )
    {
     #ifdef KOKKOS_ENABLE_SERIAL
        using ExecutionSpace = Kokkos::Serial;
	using MemorySpace    = Kokkos::HostSpace;
     #else
        throw std::runtime_error( "Serial Backend Not Enabled" );
     #endif
    }
    else if ( 0 == exec_space.compare( "openmp" ) ||
              0 == exec_space.compare( "OpenMP" ) ||
              0 == exec_space.compare( "OPENMP" ) )
    {
    #ifdef KOKKOS_ENABLE_OPENMP
	using ExecutionSpace = Kokkos::OpenMP;
	using MemorySpace    = Kokkos::HostSpace;
    #else
        throw std::runtime_error( "OpenMP Backend Not Enabled" );
    #endif
    }
    else if ( 0 == exec_space.compare( "cuda" ) ||
              0 == exec_space.compare( "Cuda" ) ||
              0 == exec_space.compare( "CUDA" ) )
    {
    #ifdef KOKKOS_ENABLE_CUDA
	using ExecutionSpace = Kokkos::Cuda;
	using MemorySpace    = Kokkos::CudaSpace;
    #else
        throw std::runtime_error( "CUDA Backend Not Enabled" );

    #endif
    }
    */

    using ExecutionSpace = Kokkos::Serial;
    using MemorySpace    = Kokkos::HostSpace;
    //Declare Particle Struct 
    using particle_members =  Cabana::MemberTypes<double[3], double[3], double[3],int>;
    using particle_list = Cabana::AoSoA<particle_members, MemorySpace>;
    particle_list particles("A", 432);
    auto positions = Cabana::slice<0>( particles );
    auto velocity  = Cabana::slice<1>( particles );
    auto v_nBody   = Cabana::slice<2>( particles );
    auto ids       = Cabana::slice<3>( particles );

    //Position particles in the middle of a cell
    double x[3], r;
    int particle_counter = 0;

    for ( int p = 0; p < ppc; ++p )
        for ( int i = 0; i < global_num_cell[0]; ++i )
            for ( int j = 0; j < global_num_cell[1]; ++j )
                for ( int k = 0; k < global_num_cell[2]; ++k, ++particle_counter )
                {

		    x[0] = grid_min[0] + grid_delta[0] * ( 0.5 + i );
		    x[1] = grid_min[1] + grid_delta[1] * ( 0.5 + j );
		    x[2] = grid_min[2] + grid_delta[2] * ( 0.5 + k ); 
		    //Spherical

		  /*  r = pow( pow(x[0], 2.0) + pow(x[1],2.0) + pow(x[2], 2.0),  0.5);

                    if ( (r <= 0.5) ){ */
                  
		   //    particle_counter += 1;
                       positions( particle_counter, 0 ) = x[0];
                       positions( particle_counter, 1 ) = x[1];
                       positions( particle_counter, 2 ) = x[2];
		       velocity(  particle_counter, 0)  = 1.0;
		       velocity(  particle_counter, 1)  = 2.0;
		       velocity(  particle_counter, 2)  = 3.0;
		       v_nBody(  particle_counter, 0)  = 0.0;
                       v_nBody(  particle_counter, 1)  = 0.0;
                       v_nBody(  particle_counter, 2)  = 0.0;
                       ids( particle_counter        )  = particle_counter;
		  //}
                }   

      int num_p = particle_counter;
      using ListType = Cabana::LinkedCellList<MemorySpace,double>;
      ListType neigh_list(positions,0,num_p,grid_delta,grid_min,grid_max,2*cell_size, 0.5);

      double xq[3], u[3], K[3];
      auto  Nbody_corr = KOKKOS_LAMBDA(const int p, const int  q){


      /*      x = { positions( p, 0 ), positions( p, 1 ), positions( p, 2 ) };
            xq = { positions( q, 0 ), positions( q, 1 ), positions( q, 2 ) };
            u = { velocity( p, 0 ), velocity( p, 1 ), velocity( p, 2 ) };
           
               //Evaluate Green's Function
            GreensFunction::CalculateK(x,xq,u, K);
      */
                    for(int d = 0; d < 3; d++){
                      v_nBody(p,d) += velocity(q,d); //K[d];
                    }
                
          };

          Cabana::neighbor_parallel_for(Kokkos::RangePolicy<ExecutionSpace>( ExecutionSpace(), 0, num_p), Nbody_corr, neigh_list, Cabana::FirstNeighborsTag(), Cabana::SerialOpTag(), "LocalCorrections" );


     for( int p = 0; p < num_p; p++){

	  int i = static_cast<int>( positions(p,0) / cell_size );
	  int j = static_cast<int>( positions(p,1) / cell_size );
	  int k = static_cast<int>( positions(p,2) / cell_size );

          std::cout << " i = " << i << " j = " << j << " k = " << k << std::endl; 
	  int num_n =
            Cabana::NeighborList<ListType>::numNeighbor( neigh_list, p );
          std::cout << "num neighbors = " << num_n << std::endl;	  
	  std::cout << "vnBody1 = " << v_nBody(p,0) << "vnBody2 = " << v_nBody(p,1) << "vnBody3 = " << v_nBody(p,2) << std::endl;  

     }
}

//---------------------------------------------------------------------------//
int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );

    Kokkos::initialize( argc, argv );

    // check inputs and write usage
    if ( argc < 5 )
    {
        std::cerr << "Usage: ./init_grid cell_size parts_per_cell_size "
                     "halo_cells exec_space vorticity\n";
        std::cerr << "\nwhere cell_size       edge length of a computational "
                     "cell (domain is unit cube)\n";
        std::cerr
            << "      parts_per_cell  particles per cell in each direction\n";
        std::cerr << "      halo_cells      number of halo cells\n";
        std::cerr << "      dt              time step size\n";
        std::cerr << "      t_end           simulation end time\n";
        std::cerr
            << "      write_freq      number of steps between output files\n";
        std::cerr << "      exec_space      execute with: serial, openmp, "
                     "cuda, hip\n";
        std::cerr << "\nfor example: ./init_grid 0.05 2 0 serial 1\n";
        Kokkos::finalize();
        MPI_Finalize();
        return 0;
    }

    // cell size
    double cell_size = std::atof( argv[1] );

    // particles per cell in a dimension
    int ppc = std::atoi( argv[2] );

    // number of halo cells.
    int halo_size = std::atoi( argv[3] );

    // execution space
    std::string exec_space( argv[4] );

    double vorticity = std::atof( argv[5] );

    // run the problem.
    initgrid( cell_size, ppc, halo_size,
              exec_space, vorticity );

    Kokkos::finalize();

    MPI_Finalize();

    return 0;
}

//---------------------------------------------------------------------------//
