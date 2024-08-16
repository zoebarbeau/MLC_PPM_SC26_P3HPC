

#ifndef EXAMPM_MLC_INTERP_HPP
#define EXAMPM_MLC_INTERP_HPP

#include <ExaMPM_Types.hpp>

#include <Cabana_Grid.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>

#include <cmath>
#include <type_traits>

namespace ExaMPM
{
//---------------------------------------------------------------------------//
// LOCAL INTERPOLATION
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
// Local grid-to-point.
//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//

namespace MLC_Interp
{
/*
  \brief Interpolate a scalar value to a point. 3D specialization.
  \param view A functor with view semantics of scalar grid data from which to
  interpolate.
  \param result The scalar value at the point.
*/
template <int Order>
struct GridData;

template<>	
struct GridData<1>
{

    double cell_size;
    double center;   
    static constexpr int num_space_dim = 1;
 
    GridData( const double h, const double c )
    {
       cell_size = h;
       center    = c;

    }

};

template<>
struct GridData<2>
{

    double cell_size;
    double center;
    static constexpr int num_space_dim = 2;

    GridData( const double h, const double c )
    {
       cell_size = h;
       center    = c;

    }

};

template<>
struct GridData<3>
{

    double cell_size;
    double center;
    static constexpr int num_space_dim = 3;

    GridData( const double h, const double c )
    {
       cell_size = h;
       center    = c;

    }

};

template <class ViewType, class GridDataType>
KOKKOS_INLINE_FUNCTION
    std::enable_if_t<1 == GridDataType::num_space_dim, void>
    value( const ViewType& view, const GridDataType& g, typename ViewType::value_type xp[3],typename ViewType::value_type result )
{

    result = 0.0;
    int ig = round( (xp[0]+g.center) / g.cell_size ); 
    int jg = round( (xp[1]+g.center) / g.cell_size );
    int kg = round( (xp[2]+g.center) / g.cell_size );
    result += view( ig, jg, kg, 0 ) ;
}

/*!
  \brief Interpolate a scalar value to a point. 2D specialization.
  \param view A functor with view semantics of scalar grid data from which to
  interpolate.
  \param sd The spline data to use for the interpolation.
  \param result The scalar value at the point.
*/
template <class ViewType, class GridDataType>
KOKKOS_INLINE_FUNCTION
    std::enable_if_t<2 == GridDataType::num_space_dim, void>
    value( const ViewType& view, const GridDataType& g,typename ViewType::value_type xp[3],typename ViewType::value_type result[2] )
{

    for( int d = 0; d < 2; d++)	
       result[d] = 0.0;

    int ig = round( (xp[0]+g.center) / g.cell_size );
    int jg = round( (xp[1]+g.center) / g.cell_size );
    int kg = round( (xp[2]+g.center) / g.cell_size );

     
    for ( int d = 0; d < 2; d++ )
            result[d] += view( ig, jg, kg, d );


}

//---------------------------------------------------------------------------//
/*!
  \brief Interpolate a vector value to a point. 3D specialization.
  \param view A functor with view semantics of vector grid data from which to
  `interpolate.
*/
  template <class ViewType, class GridDataType>
KOKKOS_INLINE_FUNCTION
 void   value( const ViewType& view, const GridDataType& g, typename ViewType::value_type xp[3],
           typename ViewType::value_type result[3])
{

    for( int d = 0; d < 3; d++)
       result[d] = 0.0;

    int ig = round( (xp[0]+g.center) / g.cell_size );
    int jg = round( (xp[1]+g.center) / g.cell_size );
    int kg = round( (xp[2]+g.center) / g.cell_size );

    for ( int d = 0; d < 3; d++ )
            result[d] += view( ig, jg, kg, d );
}
} //MLC_Interp
} //ExaMPM

#endif
