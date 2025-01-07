

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
    int ig = floor( (xp[0]+g.center) / g.cell_size ); 
    int jg = floor( (xp[1]+g.center) / g.cell_size );
    int kg = floor( (xp[2]+g.center) / g.cell_size );
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

    int ig = floor( (xp[0]+g.center) / g.cell_size );
    int jg = floor( (xp[1]+g.center) / g.cell_size );
    int kg = floor( (xp[2]+g.center) / g.cell_size );

     
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
  std::enable_if_t<3 == GridDataType::num_space_dim, void>
 value( const ViewType& view, const GridDataType& g, typename ViewType::value_type xp[3],
           typename ViewType::value_type result[3])
{

    for( int d = 0; d < 3; d++)
       result[d] = 0.0;

    int ig = floor( (xp[0]+g.center) / g.cell_size );
    int jg = floor( (xp[1]+g.center) / g.cell_size );
    int kg = floor( (xp[2]+g.center) / g.cell_size );


    for ( int d = 0; d < 3; d++ )
            result[d] += view( ig, jg, kg, d );
}

template<class ViewType, class GridDataType >
KOKKOS_INLINE_FUNCTION
 std::enable_if_t<3 == GridDataType::num_space_dim, void>
  f( const ViewType &view, int i, int j,const int k, const int d,
         const GridDataType& g,double& fx, double& fy, double& fz)
{

   double sx_p,sx_m,sy_p,sy_m,sz_p,sz_m;

   //Calculate FX       
   sx_p = view( i+1,j,k+1,d) + view(i+1,j,k-1,d)
          +view(i+1,j+1,k,d)  + view(i+1,j-1,k,d);

   sx_m = view( i-1,j,k+1,d) + view( i-1,j,k-1,d)
          +view( i-1,j+1,k,d) + view( i-1,j-1,k,d);

   fx = (sx_p - sx_m + 2*(view(i+1,j,k,d) - view(i-1,j,k,d)) ) / ( 12.0*g.cell_size );

   // Calculate FY
   sy_p = view( i,j+1,k+1,d) + view(i,j+1,k-1,d)
          +view(i+1,j+1,k,d)  + view(i-1,j+1,k,d);

   sy_m = view( i,j-1,k+1,d) + view( i,j-1,k-1,d)
          +view( i+1,j-1,k,d) + view( i-1,j-1,k,d);

   fy = (sy_p - sy_m + 2*(view(i,j+1,k,d) - view(i,j-1,k,d)) ) / ( 12*g.cell_size );

   // Calculate FZ
   sz_p = view( i,j+1,k+1,d) + view(i,j-1,k+1,d)
          +view(i+1,j,k+1,d)  + view(i-1,j,k+1,d);

   sz_m = view( i,j+1,k-1,d) + view( i,j-1,k-1,d)
          +view( i+1,j,k-1,d) + view( i-1,j,k-1,d);

   fz = (sz_p - sz_m + 2*(view(i,j,k+1,d) - view(i,j,k-1,d)) ) / ( 12*g.cell_size );



}

template<class ViewType, class GridDataType >
KOKKOS_INLINE_FUNCTION
 std::enable_if_t<3 == GridDataType::num_space_dim, void>
 f2( const ViewType &view, int i, int j, int k,int d,
	 const GridDataType& g,double& fxx, double& fyy, double& fzz, 
         double& fxy, double& fxz, double& fyz)
{

      //Calculate fxx
      fxx = ( view( i+1,j,k,d) + view(i-1,j,k,d) -2*view(i,j,k,d) ) / pow(g.cell_size, 2.0);

      //Calculate fyy
      fyy = ( view( i,j+1,k,d ) + view( i,j-1,k,d ) - 2*view(i,j,k,d) ) / pow( g.cell_size, 2.0);

      //Calculate fzz
      fzz = ( view( i,j,k+1,d ) + view( i,j,k-1,d ) - 2*view(i,j,k,d) ) / pow(g.cell_size, 2.0);

      //Calculate fxy
      fxy = ( ( view( i+1,j+1,k,d) - view( i-1,j+1,k,d) ) 
            - ( view( i+1,j-1,k,d) - view( i-1,j-1,k,d) ) ) / (4*pow(g.cell_size, 2.0) );

      fxz = ( ( view( i+1,j,k+1,d) - view( i-1,j,k+1,d ) )
	    - ( view( i+1,j,k-1,d) - view( i-1,j,k-1,d ) )) / (4*pow(g.cell_size, 2.0 ) );

      fyz = ( ( view( i,j+1,k+1,d) - view( i,j-1,k+1,d) )
            - ( view( i,j+1,k-1,d) - view( i,j-1,k-1,d) ) ) / (4*pow(g.cell_size, 2.0 ) );      


}

template<class ViewType, class GridDataType >
KOKKOS_INLINE_FUNCTION
 std::enable_if_t<3 == GridDataType::num_space_dim, void>
 f3( const ViewType &view, int i, int j, int k,int d,
         const GridDataType& g,double& fxxy, double& fxxz,
         double& fyyx, double& fyyz, double& fzzx, double& fzzy, double& fxyz)
{

     //Calculate fxyz

     fxyz = ( ( (view(i+1,j+1,k+1,d) - view(i-1,j+1,k+1,d))
	      - (view(i+1,j-1,k+1,d) - view(i-1,j-1,k+1,d))  )
              - ( (view(i+1,j+1,k-1,d) - view(i-1,j+1,k-1,d))
              -   (view(i+1,j-1,k-1,d) - view(i-1,j-1,k-1,d))) ) / (8*pow(g.cell_size, 3.0) );

     fxxy = ( (view(i+1,j+1,k,d) - 2*view(i,j+1,k,d) + view(i-1,j+1,k,d) )
             -(view(i+1,j-1,k,d) - 2*view(i,j-1,k,d) + view(i-1,j-1,k,d) ) ) / (2*pow(g.cell_size, 3.0));

     fxxz = ( (view(i+1,j,k+1,d) - 2*view(i,j,k+1,d) + view(i-1,j,k+1,d) )
             -(view(i+1,j,k-1,d) - 2*view(i,j,k-1,d) + view(i-1,j,k-1,d) ) ) / (2*pow(g.cell_size, 3.0)); 	     
     fyyx = ( (view(i+1,j+1,k,d) - 2*view(i+1,j,k,d) + view(i+1,j-1,k,d) )
             -(view(i-1,j+1,k,d) - 2*view(i-1,j,k,d) + view(i-1,j-1,k,d) ) ) / (2*pow(g.cell_size, 3.0));

     fyyz = ( (view(i,j+1,k+1,d) - 2*view(i,j,k+1,d) + view(i,j-1,k+1,d) )
             -(view(i,j+1,k-1,d) - 2*view(i,j,k-1,d) + view(i,j-1,k-1,d) ) ) / (2*pow(g.cell_size, 3.0));

     fzzx = ( (view(i+1,j,k+1,d) - 2*view(i+1,j,k,d) + view(i+1,j,k-1,d) )
             -(view(i-1,j,k+1,d) - 2*view(i-1,j,k,d) + view(i-1,j,k-1,d) ) ) / (2*pow(g.cell_size, 3.0));

     fzzy = ( (view(i,j+1,k+1,d) - 2*view(i,j+1,k,d) + view(i,j+1,k-1,d) )
             -(view(i,j-1,k+1,d) - 2*view(i,j-1,k,d) + view(i,j-1,k-1,d) ) ) / (2*pow(g.cell_size, 3.0));


}

template<class ViewType, class GridDataType>
KOKKOS_INLINE_FUNCTION
 void L7( const ViewType &view, int i, int j, int k, const GridDataType& g, double F[3] )
{


	for(int d = 0; d < 3; d++)
	    F[d] = 0.0;

        for(int d = 0; d < 3; d++)
	{

	     F[d] =  ( view(i+1,j,k,d) + view(i-1,j,k,d)
			+ view(i,j+1,k,d) + view(i,j-1,k,d)
			+ view(i,j,k+1,d) + view(i,j,k-1,d)
			- 6*view(i,j,k,d) ) / pow( g.cell_size, 2.0);

	}



}

template<class ViewType, class GridDataType>
KOKKOS_INLINE_FUNCTION
 void L27( const ViewType &view, int i, int j, int k, const GridDataType& g, double F[3] )
{


	double u_face[3] = {0.0,0.0,0.0};
        double u_corner[3] = {0.0,0.0,0.0};
	double u_edge[3] = {0.0,0.0,0.0};

        for(int d = 0; d < 3; d++)
            F[d] = 0.0;


	for(int si = i-1; si <= i+1; si++)
	   for(int sj = j-1; sj <= j+1; sj++)
              for(int sk = k-1; sk <= k+1; sk++)
              {
 
		 int s1 = si-i;
		 int s2 = sj-j;
		 int s3 = sk-k;

		 int  s = abs(s1) + abs(s2) + abs(s3);

	//	 std::cout << " s1 = " << s1 << " s2 = " << s2 << " s3 " << s3 << std::endl;
	//	 std::cout << " s = " << s << std::endl;
		  if( s == 1)
	          {
                     
		    for(int d = 0; d < 3; d++)
		       u_face[d] += view(si,sj,sk,d);
		    
 
		  }else if( s == 2)
		  {

		    for(int d = 0; d < 3; d++)
                       u_edge[d] += view(si,sj,sk,d);

		  }else if( s == 3)
                  {

		     for(int d = 0; d < 3; d++)
                       u_corner[d] += view(si,sj,sk,d);


		  }	  


	      }	      


        for(int d = 0; d < 3; d++)
        {

             F[d] = ( view(i,j,k,d)*-128.0/30.0 + u_corner[d]*1.0/30.0
			+ u_edge[d]*1.0/10.0 + 7.0/15.0*u_face[d]) / pow( g.cell_size, 2.0);    

        }


}

 template <class ViewType, class GridDataType>
KOKKOS_INLINE_FUNCTION
 std::enable_if_t<3 == GridDataType::num_space_dim, void>
 HarmonicValue( const ViewType& view, const GridDataType& g, typename ViewType::value_type xp[3],
           typename ViewType::value_type result[3])
{

    for( int d = 0; d < 3; d++)
       result[d] = 0.0;
    
    // grid index closest to the particle
    int i = floor( (xp[0]+g.center) / g.cell_size );
    int j = floor( (xp[1]+g.center) / g.cell_size );
    int k = floor( (xp[2]+g.center) / g.cell_size );
    //grid position
    double xg[3] = { i*g.cell_size - g.center, j*g.cell_size - g.center,
	             k*g.cell_size - g.center};

    Kokkos::printf(" x %f y %f z %f \n", xg[0],xg[1],xg[2]);
    //difference between particle and grid position, needed for interpolation
    double xdiff[3] = { xp[0]-xg[0], xp[1]-xg[1], xp[2]-xg[2]};
    double xdiff2[3] = { pow(xdiff[0], 2.0), pow(xdiff[1],2.0), pow(xdiff[2], 2.0) };
    double fx,fy,fz;
    double fxx,fyy,fzz,fxy,fyz,fxz;
    double fxxy,fxxz,fyyx,fyyz,fzzx,fzzy,fxyz;
    double sx_p, sx_m, sy_p, sy_m, sz_p, sz_m;

    for(int d = 0; d < 3; d++)
    {
        f(view,i,j,k,d,g,fx,fy,fz);
        f2(view,i,j,k,d,g,fxx,fyy,fzz,fxy,fxz,fyz);
	f3(view,i,j,k,d,g,fxxy,fxxz,fyyx,fyyz,fzzx,fzzy,fxyz);

	result[d] = view(i,j,k,d) + xdiff[0]*fx + xdiff[1]*fy + xdiff[2]*fz
		    + 0.5*( xdiff2[0]*fxx + xdiff2[1]*fyy + xdiff2[2]*fzz)
		    + xdiff[0]*xdiff[1]*fxy + xdiff[1]*xdiff[2]*fyz + xdiff[0]*xdiff[2]*fxz
		    +1.0/6.0*( (3*xdiff2[0] - xdiff2[1])*xdiff[1]*fxxy
		             + (3*xdiff2[0] - xdiff2[2])*xdiff[2]*fxxz		    
			     + (3*xdiff2[1] - xdiff2[0])*xdiff[0]*fyyx
		             + (3*xdiff2[1] - xdiff2[2])*xdiff[2]*fyyz 	     
                             + (3*xdiff2[2] - xdiff2[0])*xdiff[0]*fzzx
			     + (3*xdiff2[2] - xdiff2[1])*xdiff[1]*fzzy)
		    + xdiff[0]*xdiff[1]*xdiff[2]*fxyz;
//	std::cout << "velocity interp " << view(i,j,k,d) << std::endl;
/*	std::cout << " fx " << fx << " fy " << fy << " fz " << fz << std::endl;
	std::cout << " fxx " << fxx << " fyy " << fyy << " fzz " << std::endl;
	std::cout << " fxxy " << fxxy << " fxxz " << fxxz << " fyyx " << fyyx << " fyyz " << fyyz 
		  << " fzzx " << fzzx << " fzzy " << fzzy << std::endl; */
    }
}


} //MLC_Interp
} //ExaMPM

#endif
