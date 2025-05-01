#include "Proto.H"
#include "Multigrid.H"
#include "Proto_HDF5.H"
#include "MakeLaplaceStencils.H"
#include <iostream>
using namespace Proto;

PROTO_KERNEL_START
void f_force_avg_0(const Point& a_pt, Var<double> a_data, double a_dx)
{
    double x0[DIM];
    double x1[DIM];
    
    double a = 0.125;
    for (int dir = 0; dir < DIM; dir++)
    {
        x0[dir] = a_pt[dir]*a_dx + a;
        x1[dir] = x0[dir] + a_dx;
    }
    
    double k = M_PI*2;
    a_data(0) = + cos(k*x1[0])*cos(k*x1[1])
                - cos(k*x0[0])*cos(k*x1[1])
                - cos(k*x1[0])*cos(k*x0[1])
                + cos(k*x0[0])*cos(k*x0[1]);
    a_data(0) *= 1.0/(k*k*a_dx*a_dx);
}
PROTO_KERNEL_END(f_force_avg_0, f_force_avg);

int MG_LEVEL = 0;
int SOLVE_ITER = 0;
void testStencil(Stencil<double> a_sten)
{
  int domainSize= 4;
  double errold;
  for (int lev = 0; lev < 4 ; lev++)
    {
      cout << "domainSize = " << domainSize << endl;
      double h = 1.0/domainSize;
      Box bx0(Point::Ones(1),Point::Ones(domainSize-1));
      Box bx = a_sten.domain(bx0);
      cout << bx << endl;
      BoxData<double> phi(bx);
      phi.setVal(1.);
      BoxData<double> LOfPhi = a_sten(phi,1./(h*h));
      cout << "LOfPhi(const.) = " << LOfPhi.absMax() << endl;
      forallInPlace_p(f_greensfcn,bx,phi,h,Point::Ones(domainSize + domainSize/2));                        LOfPhi = a_sten(phi,1./(h*h));
      double err = LOfPhi.absMax();
      cout << "LOfPhi(harmonic) = " << err << endl;
      if (lev > 0) cout << "rate = " << log(err/errold)/log(2.0) << endl;
      domainSize *= 2;
      errold = err;
      cout << endl;
    }
}
     
int main(int argc, char** argv)
{
    #ifdef PR_MPI
    MPI_Init(&argc, &argv);
    #endif
    
    // SETUP
    //#ifdef PR_HDF5
    HDF5Handler h5;
    //#endif
    int domainSize;
    int numIter = 20;
    int solverInd;
    cout << "input solver type = 0,1 ; domain size (power of two), numIter" << endl;
    cin >> solverInd >> domainSize >> numIter ;
    cout << "solver type =" << solverInd << ", domain size" << domainSize  << ", numIter = " << numIter << endl;  
    PR_TIMER_SETFILE(to_string(domainSize) 
                   + "_GreensFunction.time.table");
  PR_TIMERS("GreensFunction");


  // Construct stencil. 
  double diagCoef;
  Stencil<double> LStencil;
  int diamStencil;
  LaplaceStencils(LStencil,diagCoef,diamStencil,solverInd);
  double h = 1.0;
  // testStencil(LStencil);
  Box domainBoxValid(Point::Ones(-domainSize/2+1),Point::Ones(domainSize/2-1));
  Box domainBox = domainBoxValid.grow(Point::Ones(diamStencil));
  BoxData<double> phi(domainBox);
  BoxData<double> rhs(domainBoxValid);
  cout << "solver type =" << solverInd << ", domain size" << domainSize << endl; 
  if (solverInd < 2)
    {
      // using multigrid as a solver.
      // define solver.
      Multigrid<double> mg(LStencil,diagCoef,diamStencil);
      phi.setVal(0.);
      rhs.setVal(0.);
      rhs(Point::Zeros()) = 1.0;
      
      double tol = 1.e-12;
      double resnorm0 = rhs.absMax();

      // Begin mg iteration.
      for (int iter = 0; iter < numIter; iter++)
        {
          mg.mgRelax(phi,rhs,h);
          double resnorm = mg.resnorm(phi,rhs,h);
          cout << "residual at iter " << iter+1 << ": " << resnorm << endl;
          if (resnorm < tol*resnorm0) break;
        }
    }
  else
    {
      // Use multigrid with solverType = 1 as a smoother.
      double diagCoefSmooth;
      Stencil<double> LStencilSmooth;
      int diamStencilSmooth;
      LaplaceStencils(LStencilSmooth,diagCoefSmooth,diamStencilSmooth,1);
      Multigrid<double> mg(LStencilSmooth,diagCoefSmooth,diamStencilSmooth);
      Box domainBoxValid(Point::Ones(-domainSize/2+1),Point::Ones(domainSize/2-1));
      Box domainBox = domainBoxValid.grow(Point::Ones(diamStencil));
      cout << "domain = " << domainBox
           << ", valid domain =" << domainBoxValid << endl;
      cout << "diagCoef, diamStencil = " << diagCoef
           << ", " << diamStencil << endl;
      phi.setVal(0.);
      rhs.setVal(0.);
      rhs(Point::Zeros()) = 1.0;
      double resnorm0 = rhs.absMax();
      for (int iter = 0; iter < numIter; iter++)
        {
          mg.mgSmoother(phi,
                        rhs,
                        LStencil,
                        diagCoef,
                        h,
                        diamStencil);
          
          mg.setGhost(phi,h,diamStencil);
          BoxData<double> rhsSmoother(rhs.box());
          rhsSmoother.setVal(0.);
          rhsSmoother += rhs;
          rhsSmoother *= -1.0;
          rhsSmoother += LStencil(phi,1.0/(h*h));
          double resnorm = rhsSmoother.absMax();
          cout << "residual at iter " << iter+1 << ": " << resnorm << endl;
          if (resnorm < 1.e-14*resnorm0) break;
        }
    }
  //BoxData<double> phiTrimmed(Box(Point::Ones(-domainSize/4+1),Point::Ones(domainSize/4)));
  BoxData<double> phiTrimmed(Box(Point::Zeroes(),Point::Ones(domainSize/4)));
  phi.copyTo(phiTrimmed);
  Box bx = phiTrimmed.box();
  std::ofstream ofs ("G_"+to_string(domainSize/4)+"_Octant", std::ofstream::out);
  for (auto bit : bx)
    {
      ofs << bit << " , " << std::scientific << std::setprecision(12) << phiTrimmed(bit) << endl;
    }
  BoxData<double> phiExact = forall_p<double>(f_greensfcn,phi.box(),h,Point::Zeros());
  phi -= phiExact;
  BoxData<double> LOfPhiE = LStencil(phiExact,-1.0);
  //LOfPhiE(Point::Zeros()) += 1.0;
  BoxData<double> logLphi =
    forall<double>([] PROTO_LAMBDA
                   (Var<double>& a_loglphi,
                    Var<double>& a_lphi)
                   { 
                     a_loglphi(0) = log(abs(a_lphi(0)))/log(10.0);
                   },LOfPhiE);
  h5.writePatch(1.0,logLphi,"logLphi");
  
   ofstream filestream;
   ofstream filestreamL;
   filestreamL.open("LOfPhiE"+to_string(solverInd) + "_"+to_string(domainSize)+".curve");
   filestream.open("PhiErrScaled"+to_string(solverInd) + "_"+to_string(domainSize)+".curve");
   vector<double> phiErr1D(domainSize/4+1,0.);
   vector<double> LOfPhi1D(domainSize/4+1,0.);
   for (auto bit : phiTrimmed.box())
     {
       int index1D =
         max(abs(bit[0]*1.0),max(abs(bit[1]*1.0),abs(bit[2]*1.0)));
       double gfInv =
         4*M_PI*sqrt((bit[0]*bit[0]+bit[1]*bit[1] + bit[2]*bit[2])*1.0);       
       phiErr1D[index1D] = max(phiErr1D[index1D],
                             abs(abs(phiTrimmed(bit)*gfInv)-1.0));
       LOfPhi1D[index1D] = max(LOfPhi1D[index1D],abs(LOfPhiE(bit)));
     }
   for (int ll = 1; ll < domainSize/4+1; ll++)
     {
       filestream << ll << " " << phiErr1D[ll] << endl;
       filestreamL << ll << " " << LOfPhi1D[ll] << endl;
     }
   filestream.close();
   filestreamL.close();
  PR_TIMER_REPORT(); 
#ifdef PR_MPI
  MPI_Finalize();
#endif
  return 0;
}
