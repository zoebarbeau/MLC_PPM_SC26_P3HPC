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

    int cells = global_num_cell[0]*global_num_cell[1]*global_num_cell[2]*ppc;
    int nodes = (global_num_cell[0]+1)*(global_num_cell[1]+1)*(global_num_cell[2]+1);

    std::cout << " cells = " << cells << " nodes = " << nodes << std::endl;
    double grid_min[3] = { -0.75, -0.75, -0.75 };
    double grid_max[3] = { 0.75, 0.75, 0.75 };
    double grid_delta[3] = { cell_size, cell_size, cell_size };
    std::array<bool, 3> periodic = { false, false, false };


    std::array<int, 3> num_cell = global_num_cell;

    // Create global mesh bounds.
    std::array<double, 3> global_low_corner = { global_box[0],
                                                global_box[1],
                                                global_box[2] };
    std::array<double, 3> global_high_corner = { global_box[3],
                                                 global_box[4],
                                                 global_box[5] };
    std::cout << "declare grid highs and lows " << std::endl;
    // Due to the 2D nature of the problem we will only partition in Y. The
    // behavior of the fluid will be to largely just run out in X and Z with
    // little movement in Y.
    int comm_size;
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );
    std::array<int, 3> ranks_per_dim = { 1, comm_size, 1 };
    Cabana::Grid::ManualBlockPartitioner<3> partitioner( ranks_per_dim );

    using ExecutionSpace = Kokkos::Serial;
    using MemorySpace    = Kokkos::HostSpace;

    //Declare Particle Struct 
    using particle_members =  Cabana::MemberTypes<double[3], double[3], double[3],int, int[3]>;
    using particle_list = Cabana::AoSoA<particle_members, MemorySpace>;
    particle_list particles("A", cells+nodes);

    auto global_mesh = Cabana::Grid::createUniformGlobalMesh(
                                 global_low_corner, global_high_corner, num_cell );
    // Create the global grid.
    auto global_grid = Cabana::Grid::createGlobalGrid(
        MPI_COMM_WORLD, global_mesh, periodic, partitioner );

    // Node Layouts
    auto node_scalar_layout = Cabana::Grid::createArrayLayout(
          global_grid, 1, 0, Cabana::Grid::Node() );

    auto node_vector_layout = Cabana::Grid::createArrayLayout(
          global_grid, 3, 0, Cabana::Grid::Node() );

    // Arrays
    auto vel_array = Cabana::Grid::createArray<double, MemorySpace>(
            "velocity", node_vector_layout );
    auto velcorr_array = Cabana::Grid::createArray<double, MemorySpace>(
            "velocorr", node_vector_layout );

    auto velocity_g = vel_array->view();
    auto velcorr_g  = velcorr_array->view();

    // Slices of Cabana
    auto positions = Cabana::slice<0>( particles );
    auto velocity  = Cabana::slice<1>( particles );
    auto v_nBody   = Cabana::slice<2>( particles );
    auto ids       = Cabana::slice<3>( particles );
    auto index     = Cabana::slice<4>( particles );

    double x[3], r;

    // Fake Grid Particles
    int particle_counter = 0;
    for ( int i = 0; i < global_num_cell[0]+1; ++i )
          for ( int j = 0; j < global_num_cell[1]+1; ++j )
                for ( int k = 0; k < global_num_cell[2]+1; ++k, ++particle_counter )
                {

                    x[0] = grid_min[0] + grid_delta[0] * (  i );
                    x[1] = grid_min[1] + grid_delta[1] * (  j );
                    x[2] = grid_min[2] + grid_delta[2] * (  k );
           
                    velocity_g( i,j,k,0 ) = 250.0;
		    velocity_g( i,j,k,1 ) = 250.0;
		    velocity_g( i,j,k,2 ) = 250.0;

		    velcorr_g( i,j,k,0 ) = 0.0;
                    velcorr_g( i,j,k,1 ) = 0.0;
                    velcorr_g( i,j,k,2 ) = 0.0;

                    positions( particle_counter, 0 ) = x[0];
                    positions( particle_counter, 1 ) = x[1];
                    positions( particle_counter, 2 ) = x[2];

		    index( particle_counter, 0 ) = i;
		    index( particle_counter, 1 ) = j;
		    index( particle_counter, 2 ) = k;

                    velocity(  particle_counter, 0)  = 250.0;
                    velocity(  particle_counter, 1)  = 250.0;
                    velocity(  particle_counter, 2)  = 250.0;
                    v_nBody(  particle_counter, 0)  = 0.0;
                    v_nBody(  particle_counter, 1)  = 0.0;
                    v_nBody(  particle_counter, 2)  = 0.0;
                    ids( particle_counter        )  = 1;
                }

      std::cout << "number grid particles " << particle_counter << std::endl;
      int num_grid = abs(particle_counter );

      // Real Particles
      for ( int p = 0; p < ppc; ++p )
        for ( int i = 0; i < global_num_cell[0]; ++i )
            for ( int j = 0; j < global_num_cell[1]; ++j )
                for ( int k = 0; k < global_num_cell[2]; ++k, ++particle_counter )
                {

                    x[0] = grid_min[0] + grid_delta[0] * ( 0.5 + i );
                    x[1] = grid_min[1] + grid_delta[1] * ( 0.5 + j );
                    x[2] = grid_min[2] + grid_delta[2] * ( 0.5 + k );

                       positions( particle_counter, 0 ) = x[0];
                       positions( particle_counter, 1 ) = x[1];
                       positions( particle_counter, 2 ) = x[2];

		       index( particle_counter, 0 ) = 0;
		       index( particle_counter, 1 ) = 0;
		       index( particle_counter, 2 ) = 0;

                       velocity(  particle_counter, 0)  = 1.0;
                       velocity(  particle_counter, 1)  = 1.0;
                       velocity(  particle_counter, 2)  = 1.0;
                       v_nBody(  particle_counter, 0)  = 0.0;
                       v_nBody(  particle_counter, 1)  = 0.0;
                       v_nBody(  particle_counter, 2)  = 0.0;
                       ids( particle_counter        )  = 100;
                }

      int num_p = abs(particle_counter - num_grid);

      std::cout << "number of particles = " << num_grid << std::endl;
      std::cout << "positions_size = " << positions.size() << std::endl;

      // Neighbor List
      using ListType = Cabana::LinkedCellList<MemorySpace,double>;
      ListType neigh_list(positions,0,num_p+num_grid,grid_delta,grid_min,grid_max,cell_size, 1);

      //Iterate to Find Neighbors 
      double xq[3], u[3], K[3];
      auto  Nbody_corr = KOKKOS_LAMBDA(const int p, const int  q){

        
	  if( ids(q) == 100 && ids(p) == 1){

            for(int d = 0; d < 3; d++){
              velcorr_g(index(p,0),index(p,1), index(p,2),d) += velocity(q,d); //K[d];
	      std::cout << "velocity q = " << velocity(q,d) << std::endl;
	      std::cout << " Velcorrection = " << velcorr_g(index(p,0), index(p,1), index(p,2),d) << std::endl;
	      std::cout << "i = " << index(p,0) << " j = " << index(p,1) << " k = " << index(p,2) << std::endl;
            }

	  }
     };

          Cabana::neighbor_parallel_for(Kokkos::RangePolicy<ExecutionSpace>( ExecutionSpace(), 0,num_p+num_grid), Nbody_corr, neigh_list, Cabana::FirstNeighborsTag(), Cabana::SerialOpTag(), "LocalCorrections" );

	  
     int iterate = 0;
     for ( int i = 0; i < global_num_cell[0]+1; ++i )
            for ( int j = 0; j < global_num_cell[1]+1; ++j )
                for ( int k = 0; k < global_num_cell[2]+1; ++k, ++iterate )
                {
                     int num_n =
                        Cabana::NeighborList<ListType>::numNeighbor( neigh_list, iterate );
                     std::cout << "num neighbors = " << num_n << std::endl;
                     std::cout << "velcorr_g1 = " << velcorr_g(i,j,k,0) << " velcorr_g2 = " << velcorr_g(i,j,k,1) << " velcorr_g3 = " << velcorr_g(i,j,k,2) << std::endl;
                     std::cout << "i = " << i << " j = " << j << " k = " << k << std::endl;
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
        std::cerr << "\nfor example: ./NBodyGrid 0.05 2 0 serial 1\n";
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
