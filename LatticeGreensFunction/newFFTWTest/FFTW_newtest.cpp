# include <iostream>
# include <vector>
# include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <complex>
#include <algorithm>
#include <iomanip>
#include "fftw3.h"
#include "Proto.H"
#include "FFTWLGFConvolution.H"
#include "Proto_HDF5.H"

void curl(BoxData<double,DIM>& a_vel,
          BoxData<double,DIM>& a_psi,
          double a_h)
{
  a_vel.setVal(0.);
  for (int dir = 0; dir < DIM; dir++)
    {
      int dir1 = (dir + 1)%DIM;
      int dir2 = (dir + 2)%DIM;
     Stencil<double> d1 = Stencil<double>::Derivative(1,dir1,2);
      Stencil<double> d2 = Stencil<double>::Derivative(1,dir2,2);
      BoxData<double> psicomp1 = slice(a_psi,dir1);
      BoxData<double> psicomp2 = slice(a_psi,dir2);
      BoxData<double> velcomp = slice(a_vel,dir);
      velcomp += d2(psicomp1,-1.0/a_h);
      velcomp += d1(psicomp2,1.0/a_h);
    }
}

void Lap(BoxData<double,DIM>& a_vel,
          BoxData<double,DIM>& a_psi,
          double a_h)
{
  a_vel.setVal(0.);
 
   for (int dir = 0; dir < DIM; dir++)
    {
//    Stencil<double> d1 = Stencil<double>::Derivative(2,0,2);
//    Stencil<double> d2 = Stencil<double>::Derivative(2,1,2);
//    Stencil<double> d3 = Stencil<double>::Derivative(2,2,2);
      BoxData<double> psicompdir = slice(a_psi,dir);
      Stencil<double> lpl = Stencil<double>::Laplacian();
      BoxData<double> velcomp = slice(a_vel,dir);
      velcomp += lpl(psicompdir,1/(a_h*a_h));
//    velcomp += d2(psicompdir,1.0/(a_h*a_h));
//    velcomp += d3(psicompdir,1.0/(a_h*a_h));
    }
}

PROTO_KERNEL_START
void f_vorticity_F(
                   Point a_pt,
                   Var<double>& a_f,
                   double a_h,
                   double a_rad0,
                   double a_U,
                   int a_dir)
{
  Array<double,DIM> x;
  double rad0sq = a_rad0*a_rad0;
  double radius = 0.;
  for (int dir = 0; dir < DIM ; dir++)
    {
      x[dir] = a_pt[dir]*a_h - .5;
      radius += x[dir]*x[dir];
    }
  radius = sqrt(radius);
  double xx = 0;
  if (a_dir == 0) xx = x[1];
  if (a_dir == 1) xx = -x[0];
  if (radius < a_rad0)
    {
      a_f(0) = 15.0*a_U*xx/(2*rad0sq);
    }
  else
    {
      a_f(0) = 0.;
    } 
}
PROTO_KERNEL_END(f_vorticity_F, f_vorticity);

PROTO_KERNEL_START
void f_velocity_F(
                   Point a_pt,
                   Var<double>& a_f,
                   double a_h,
                   double a_rad0,
                   double a_U,
                   int a_dir)
{
  Array<double,DIM> x;
  double rad0sq = a_rad0*a_rad0;
  double radius = 0.;
  for (int dir = 0; dir < DIM ; dir++)
    {
      x[dir] = a_pt[dir]*a_h - .5;
      radius += x[dir]*x[dir];
    }
  radius = sqrt(radius);
  if (a_dir == 0)
    {
      double xz = x[0]*x[2];
      if (radius < a_rad0)
        {
          a_f(0) = -1.5*a_U*xz/(rad0sq);
        }
      else
        {
          a_f(0) = -1.5*a_U*xz*a_rad0*rad0sq/pow(radius,5);
        }
    }
  if (a_dir == 1)
    {
      double yz = x[1]*x[2];
      if (radius < a_rad0)
        {
          a_f(0) = -1.5*a_U*yz/(rad0sq);
        }
      else
        {
          a_f(0) = -1.5*a_U*yz*a_rad0*rad0sq/pow(radius,5);
        }
    }
  if (a_dir == 2)
    {
      double zsq = x[2]*x[2];
    if (radius < a_rad0)
      {
        a_f(0) = -1.5*a_U*(1.0 + zsq/rad0sq - 2*radius*radius/rad0sq);
      }
    else
      {
        a_f(0) = a_U*(1.0 + .5*pow(a_rad0/radius,3)
                      - 1.5*zsq*a_rad0*rad0sq/pow(radius,5));
      }
    }
}
PROTO_KERNEL_END(f_velocity_F, f_velocity);
using namespace Proto;
void transposeRowToCol(double* a_col,double* a_row,int N)
{
  for(int i = 0; i < N; i++){
   for(int j = 0; j < N; j++){
     for(int k = 0; k < N; k++){
       
       //Calculate the index in the domain doubled output vector,
       // corresponding to the low corner of the doubled domain.
       
       int index_row = i*N*N + j*N + k;
       int index_col = k*N*N + j*N + i;
       a_col[index_col] = a_row[index_row];
     }
   }
  }
}
void transposeColToRow(double* a_row,double* a_col,int N)
{
  for(int i = 0; i < N; i++){
   for(int j = 0; j < N; j++){
     for(int k = 0; k < N; k++){
       
       //Calculate the index in the domain doubled output vector,
       // corresponding to the low corner of the doubled domain.
       
       int index_row = i*N*N + j*N + k;
       int index_col = k*N*N + j*N + i;
       a_row[index_row] = a_col[index_col];
     }
   }
  }
}       
int main(){
    HDF5Handler h5;
    int N;
    std::string phiOutName;
    std::cout << "enter number of grid points" << std::endl;
    std::cin >> N ;
    double h = 1.0/N;
    Box bx(Point::Zeros(),Point::Ones(N-1));
    BoxData<double> f(bx),phi(bx);
    double U = 1.0;
    double rad0 = .25;
    BoxData<double,DIM> psi(bx);
    psi.setVal(0.);
    LGFConvolution lgfconv("../exec/phiTrimmed",h,N);

    // initialize omega.
    BoxData<double,DIM> omegaExact(bx);
    BoxData<double> omegacomp(bx);
    for (int dir = 0; dir < DIM-1; dir++)
      {
        omegacomp = slice(omegaExact,dir);
        forallInPlace_p(f_vorticity,omegacomp,h,rad0,U,dir);
      }
    
    // solve Laplacian^h(psi) = -omega <-> psi = - G^h * omega ,
    // one component at a time.
    for (int dir = 0; dir < DIM-1; dir++)
      {
        BoxData<double> psicomp = slice(psi,dir);
        // Proto is column-ordered, while the FFTW and FFTX assumes

        // data holders to store the inputs to and outputs from
        // LGFConvolution in row order.
        std::vector<double> dataIn(N*N*N,0.);
        std::vector<double> dataOut(N*N*N,0.);
        transposeColToRow(dataIn.data(),omegacomp.data(),N);
        //dataOut = G^h*dataIn.
        lgfconv.convolveWithG(dataIn,dataOut);
        transposeRowToCol(psicomp.data(),dataOut.data(),N);
        // don't forget to flip the sign.
        psicomp *= -1.0;
      }

    // Take a second-order finite-difference curl of psi to compute
    // the velocity.
    BoxData<double,DIM> velComputed(bx.grow(Point::Ones(-1)));
    curl(velComputed,psi,h);

    // Add back in the uniform background flow.
    BoxData<double> w = slice(velComputed,DIM-1);
    w += U;

    // compare computed versus exact velocities, vorticities. 
    BoxData<double,DIM> velExact(bx.grow(Point::Ones(-1)));
    for (int dir = 0; dir < DIM; dir++)
      {
        BoxData<double> velcomp = slice(velExact,dir);
        forallInPlace_p(f_velocity,velcomp,h,rad0,U,dir);
      }
  
    BoxData<double,DIM> FComputed(bx.grow(Point::Ones(-1)));
    Lap(FComputed,velExact,h);
 
    BoxData<double,DIM> omegaComputed(bx.grow(Point::Ones(-2)));
    BoxData<double,DIM> omegaCE(bx.grow(Point::Ones(-1)));
    curl(omegaComputed,velComputed,h);
    curl(omegaCE,velExact,h);
    h5.writePatch(1.0,velComputed,"velComputed");
    h5.writePatch(1.0,velExact,"velExact");
    h5.writePatch(1.0,omegaComputed,"omegaComputed");
    h5.writePatch(1.0,omegaExact,"omegaExact");
    h5.writePatch(1.0,omegaCE,"omegaCE");
    h5.writePatch(1.0,FComputed, "F");
    return 0;
}
