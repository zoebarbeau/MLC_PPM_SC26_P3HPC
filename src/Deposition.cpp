template <class ProblemManagerType, class ExecutionSpace, class NeighborListType, class GridManager>
 void Deposition( const ExecutionSpace& exec_space, const ProblemManagerType& pm, const NeighborListType& Pi_list,
                  const NeighborListType& Ci_list, const GridManager& gridp, const int num_grid, 
		  const int extent, const double center, const double h,const double hp,const double corr_radius)
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
   
//    int sz = corr_radius*2 +1;
//   Kokkos::View<double*****> vel_loc("local_velocity",num_grid,sz,sz,sz,3);
// Kokkos::deep_copy( vel_loc, 0.0);
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
            Ci_list.getStencilCells( Ci_list.getParticleBin( i ), imin,imax, jmin,
                               jmax, kmin, kmax );
            bool particlefound = false;
            int counter = 0;

            // cor_radius*2+1 (Change it manually!) corr_radius = 4 default
            double vel_loc[9][9][9][3]={0};
            // corr_radius = 2
            // double vel_loc[5][5][5][3]={0};
            // // corr_radius = 3
            // double vel_loc[7][7][7][3]={0};
            // // corr_radius = 5
            // double vel_loc[11][11][11][3]={0};
            // corr_radius = 6
            // double vel_loc[13][13][13][3]={0};
            // // corr_radius = 7
            // double vel_loc[15][15][15][3]={0};
            // // corr_radius = 8
            // double vel_loc[17][17][17][3]={0};
      
   //Reset Ci to 0
/*            for( int ci = imin; ci <= imax; ci++)
                for( int cj = jmin; cj <= jmax; cj ++)
                   for( int ck = kmin; ck <= kmax; ck ++)
                   {
                          for(int d = 0; d < 3; d++)
                              vel_loc[ci-imin][cj-jmin][ck-kmin][d] = 0.0;

                   }
 */

//          assert(imax - imin <= 9 && jmax - jmin <= 9 && kmax - kmin <= 9);   
	    auto offset = Pi_list.binOffset(ii,jj,kk);
            auto size   = Pi_list.binSize(ii,jj,kk);


             //Loop over Ci
             for( std::size_t r = offset; r < offset+size; r++)
             {

		            auto j = Pi_list.getParticle( r );

                     counter++;
                     int p = j ;//- num_grid;
                     double vortp[3]  = { vorticity_p( p, 0 ), vorticity_p( p, 1 ), vorticity_p( p, 2 ) };
                     double xp[3]    = { positions(p,0), positions(p,1), positions(p,2) };



            	    // Iterate over Ci
                      for( int ci = imin; ci < imax; ci++)
                         for( int cj = jmin; cj < jmax; cj ++)
                              for( int ck = kmin; ck < kmax; ck ++){
        			            //Calculate jh
                                  double xg[3] = { ci*h - center, cj*h - center, ck*h - center};    
                                  //Get true particle ID in fake particle list       
                                  double K[3];
                                  //Calculate Green's Function
                                //   GreensFunction::Calculate_qK(xg, xp, vortp, K,hp,corr_radius);
                                  GreensFunction::Calculate_qK_MatVec_Fused(xg, xp, vortp, K,hp);
                                  for(int d = 0; d < 3; d++){
        
                                    // K[d] = 1.0;
                                     vel_loc[ci-imin][cj-jmin][ck-kmin][d] += K[d]; 
                                  }
        

                         }

                  }

                
             if( counter > 0 ){
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



//                            MLC_Interp::L27(vel_loc,c0i-imin, c0j-jmin, c0k-kmin,g,F_temp,i);

		
                         //Set F
   		         for(int d = 0; d < 3; d++){

                            double result = ( vel_loc[c0i-imin][c0j-jmin][c0k-kmin][d]*-128.0/30.0 + u_corner[d]*1.0/30.0 + 
                                              u_edge[d]*1.0/10.0 + 7.0/15.0*u_face[d]) / (g.cell_size*g.cell_size);

                            Kokkos::atomic_add(&F(c0i,c0j,c0k,d), result);
                         }

/*                            double result = ( vel_loc[c0i-imin][c0j-jmin][c0k-kmin][2]*-128.0/30.0 + u_corner[2]*1.0/30.0 +
                                              u_edge[2]*1.0/10.0 + 7.0/15.0*u_face[2]) / pow( g.cell_size, 2.0 );

                            Kokkos::atomic_add(&Fx(c0i,c0j,c0k,0), result);
*/                         

                          }

                      }

  //              }
/*              for( int ci = imin; ci < imax; ci++)
                  for( int cj = jmin; cj < jmax; cj ++)
                   for( int ck = kmin; ck < kmax; ck ++)
                   {
                          for(int d = 0; d < 3; d++)
                              vel_loc[ci][cj][ck][d] = 0.0;

                   }
*/             

	});

/*       auto res = counters.copy_to_host();
       printf("Deposition summary:\n");
       printf("  particles_interacted   = %llu\n", (unsigned long long)res[PerfCounters::PARTICLES_INTERACTED]);
       printf("  cells_withp          = %llu\n", (unsigned long long)res[PerfCounters::CELLS_WITH_PARTICLES]);
       printf(" num particles total = %d\n", pm.numParticle() );

*/


//          pm.save_F("Fy_", 1, 0);

//        int N = extent;
//        Kokkos::printf("velocity added");
/*        Kokkos::parallel_for("Copy 1D to 3D", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {N,N,N}),
        KOKKOS_LAMBDA(const int i, const int j, const int k) {

           if( abs(F(i,j,k,0)) > 1e-6)
           Kokkos::printf("i %d j %d z %d Fx %f Fy %f Fz %f \n", i,j,k,F(i,j,k,0),F(i,j,k,1),F(i,j,k,2));

       });
*/
}
