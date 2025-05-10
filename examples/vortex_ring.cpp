#include <ExaMPM_BoundaryConditions.hpp>
#include <ExaMPM_Solver2.hpp>

#include <Cabana_Core.hpp>

#include <Cabana_Grid.hpp>

#include <Kokkos_Core.hpp>

#include <complex>
#include "fftx3.hpp"
// #include "interface.hpp"
// #include "/home/h82/Documents/Bluestone/SPIRAL/FFTX/fftx/examples/rconv/rconvObj.hpp"
// #include "rconvObj.hpp"
//#include "mddftObj.hpp"

#include <mpi.h>
#include <array>
#include <cmath>

//---------------------------------------------------------------------------//
// Create the problem setup. The initial geometry is a static water column
// from [0,0.4] in X, [0,0.6] in Z, with the entire Y domain filled.
struct ParticleInitFunc
{
    double _hp;
    double _h;
    ParticleInitFunc( const double cell_size, const double hp )
        : _hp( hp ), _h(cell_size)
    {
    }

    template <class ParticleType>
    KOKKOS_INLINE_FUNCTION bool operator()( const double x[3],
                                            ParticleType& p ) const
    {   
	double theta, q, R, U,a;
        double pi = 2*acos(0.0);
        double magn,vortx, vorty, vortz;
        q = pow( pow(x[0]-0.5, 2.0) + pow(x[1]-0.5,2.0) ,  0.5);

        R = 0.35;
        a = 0.075

        if( (pow(q-R,2.0) + pow(x[2]-0.5,2.0)) < pow(a,2.0)){

              theta = atan2((x[1]-0.5)/(x[0]-0.5));
	      vortz = -sin(theta);
              vortx = cos(theta); //15.0*U/(2.0*R*R)*x[1];
              vorty = 0.0; //-15.0*U/(2.0*R*R)*x[0];
	      Cabana::get<0>( p, 0 ) = vortx; //vortx;
              Cabana::get<0>( p, 1 ) = vorty; //vorty;
              Cabana::get<0>( p, 2 ) = vortz;
              
              // Velocity
              for ( int d = 0; d < 3; ++d ){
                Cabana::get<1>( p, d ) = 0.0;
		Cabana::get<3>( p, d ) = 0.0;
              }


              // Position
              for ( int d = 0; d < 3; ++d )
                 Cabana::get<2>( p, d ) = x[d]; ///+0.1*_h; // + 0.125; //0.5*_h;

	      return true;
      }

        return false;
    }
};

//---------------------------------------------------------------------------//
void initgrid(const double cell_size, const int ppc, const int halo_size,
               const std::string& exec_space, const double hp )
{
    // The dam break domain is in a box on [0,1] in each dimension.
    Kokkos::Array<double, 6> global_box = { 0,0,0,1,1,1};
    double center = 0;
    int c      = 4;
    // Compute the number of cells in each direction. The user input must
    // squarely divide the domain.
    std::array<int, 3> global_num_cell = {
        static_cast<int>( 1.0 / cell_size ),
        static_cast<int>( 1.0 / cell_size ),
        static_cast<int>( 1.0 / cell_size ) };

    std::array<int, 3> pgrid_num_cell = {
        static_cast<int>( 1.0 / hp ),
        static_cast<int>( 1.0 / hp ),
        static_cast<int>( 1.0 / hp ) };

    // This will look like a 2D problem so make the Y direction periodic.
    std::array<bool, 3> periodic = { false, false, false };

    // Due to the 2D nature of the problem we will only partition in Y. The
    // behavior of the fluid will be to largely just run out in X and Z with
    // little movement in Y.
    int comm_size;
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );
    std::array<int, 3> ranks_per_dim = { 1, 1, 1 };
    Cabana::Grid::ManualBlockPartitioner<3> partitioner( ranks_per_dim );

    // Free slip conditions (alternative: NO_SLIP)
    ExaMPM::BoundaryCondition bc;
    bc.boundary[0] = ExaMPM::BoundaryType::NO_SLIP;
    bc.boundary[1] = ExaMPM::BoundaryType::NO_SLIP;
    bc.boundary[2] = ExaMPM::BoundaryType::NO_SLIP;
    bc.boundary[3] = ExaMPM::BoundaryType::NO_SLIP;
    bc.boundary[4] = ExaMPM::BoundaryType::NO_SLIP;
    bc.boundary[5] = ExaMPM::BoundaryType::NO_SLIP;
    double t_final = 1.0;
    int write_freq = 1;
    // Solve the problem.
    auto solver = ExaMPM::createSolver(
        exec_space, MPI_COMM_WORLD, global_box, global_num_cell,pgrid_num_cell, periodic,
        partitioner, halo_size, ParticleInitFunc( cell_size, hp ),ppc,cell_size,hp,center,bc);
    solver->solve( t_final, write_freq,center,c,cell_size,hp );
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
	std::cerr << "\nwhere hp       edge length of a computational "
                     "cell for particle deposition\n";
        std::cerr << "\nfor example: ./init_grid 0.05 2 0 serial 0.025\n";
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

    //vorticity
    //
    double hp = std::atof( argv[5] );

    // // Convolution
    // double *input = new double[10*10*10];
    // double *output = new double[10*10*10];
    // std::complex<double> *symbol = new std::complex<double>[10*10*10];
    // //Vector of void pointers
    // std::vector<void*> args{output, input, symbol};
    // std::vector<int> sizes{10,10,10};

    // //rconv class
    // RCONVProblem conv{args, sizes, "rconv"};
    // // For Pruned change class name RCONV, "rconv" and add the correct obj file at the top

    // // Run the transform
    // conv.transform();


    // run the problem.
    initgrid( cell_size, ppc, halo_size,
              exec_space, hp );

    Kokkos::finalize();

    MPI_Finalize();

    return 0;
}

//---------------------------------------------------------------------------//
