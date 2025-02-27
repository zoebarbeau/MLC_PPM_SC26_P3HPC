// Test to check FFTX rconv function
// Test: Take a field with point charge of 1/h^3 and zeros o.w. 
// Then convolve it with the Lattice Green's function data coming from PhiTrimmed file
// Expected result is the Lattice Green's function data on the reduced domain

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

void out_c(const char* outName,std::complex<double>* var,int Nx,int Ny,int Nz)
{
  std::ofstream ofs(outName,std::ofstream::out);
  for(int i = 0; i < 2*Nx; i++){
    for(int j = 0; j < 2*Ny; j++){
      for(int k = 0; k <= Nz; k++){
        int ind = k + (Nz + 1)*j + 2*Ny*(Nz + 1)*i;
        ofs << "index " << ind <<": " << i<< "," << j << "," << k << ": "<< var[ind] << std::endl;
      }
    }
  }
}
void out_r_doubled(const char* outName,double* var,int Nx,int Ny,int Nz)
{
  std::ofstream ofs(outName,std::ofstream::out);
  for(int i = 0; i < 2*Nx; i++){
    for(int j = 0; j < 2*Ny; j++){
      for(int k = 0; k < 2*Nz; k++){
        int ind = k+2*Nz*j + 4*Ny*Nz*i;
        ofs << "index " << ind <<": " << i<< "," << j << "," << k << ", "<< var[ind] << std::endl;
      }
    }
  }
}
void out_r(const char* outName,double* var,int Nx,int Ny,int Nz)
{
  std::ofstream ofs(outName,std::ofstream::out);
  for(int i = 0; i < Nx; i++){
    for(int j = 0; j < Ny; j++){
      for(int k = 0; k < Nz; k++){
        int ind = k + Nz*j + Ny*Nz*i;
        ofs << "ind " << ind <<": " << i<< "," << j << "," << k << ": "<< var[ind] << std::endl;
      }
    }
  }
}
void out_r_col(std::string outName,double* var,int Nx,int Ny,int Nz)
{
  std::ofstream ofs(outName,std::ofstream::out);
  std::ofstream ofs_debug(std::string(outName)+"_debug",std::ofstream::out);
  for(int k = 0; k < Nz; k++){
    for(int j = 0; j < Ny; j++){
      for(int i = 0; i < Nx; i++){
        int index = k + Nz*j + Ny*Nz*i;
        ofs_debug << "index " << index <<": " << i<< "," << j << "," << k << ": "<< var[index] << std::endl;
        ofs << std::scientific << std::setprecision(10) << var[index] << std::endl;
      }
    }
  }
}
int main(){
    //Dims for the physical domain
    const double Lx = 1.0, Ly = 1.0, Lz = 1.0;

    // Dims for computational domain (# of grid points)
    int Nx = 10, Ny = 10, Nz = 10;

    // Grid spacing h
    double hx = Lx/Nx;
    double hy = Ly/Ny;
    double hz = Lz/Nz;
    double h = hx;

    // Field with point charge 
    double Fpc_3D[Nx][Ny][Nz];
    double* Fpc_1D  = new double [Nx*Ny*Nz];

    // point charge coordinates
    
    int cx = 6;
    int cy = 6;
    int cz = 5;
    std::string phiOutName;
    std::cout << "enter cx,cy,cz, output file name" << std::endl;
    std::cin >> cx >> cy >> cz >> phiOutName;
    std::cout << "echo cx,cy,cz,output file name " << cx << "," << cy << "," << cz <<"," << phiOutName << std::endl;
    
    // 3D point charge field
    for(int i = 0; i < Nx; i++){
        for(int j = 0; j < Ny; j++){
            for(int k = 0; k < Nz; k++){
                // index for 1D F
                int index_f = i * Nz * Ny + j * Nz + k;

                if(i == cx && j == cy && k == cz){
                  Fpc_3D[i][j][k] = 1.0;///pow(h, 3);
                  std::cout << Fpc_3D[i][j][k] << std::endl;
                }
                else{
                    Fpc_3D[i][j][k] = 0.0;
                }
                Fpc_1D[index_f] = Fpc_3D[i][j][k];
            }
        }
    }

    // Domain doubling the point charge field. It will act as the second input for rconv
    
    int domaindouble_x = 2*Nx;
    int domaindouble_y = 2*Ny;
    int domaindouble_z = 2*Nz;
    // copy rhs into high side, pad the rest with zeros.
    double* Fpc_domaindouble = new double [domaindouble_x * domaindouble_y * domaindouble_z];
    for (int index = 0; index < 8*Nx*Ny*Nz;index++) Fpc_domaindouble[index] = 0.;
    for(int i = 0; i < Nx; i++){
      for(int j = 0; j < Ny; j++){
        for(int k = 0; k < Nz; k++){

          int index_dd = (i + Nx)*4*Nz*Ny + (j + Ny)*2*Nz + k + Nx;
          int index_1d = i * Nz * Ny + j * Nz + k;
          Fpc_domaindouble[index_dd] = Fpc_1D[index_1d];
        }
      } 
    }
    
  //printf("Hello from test!!\n");
    
    // Lattice Green's function.
    // Copy into the doubled domain the LGF, with the center
    // at the center of the doubled domain..
    int originx = 0;
    int originy = 0;
    int originz = 0;
  std::ifstream infileLGF("../exec/phiTrimmed");
  std::ofstream outlgf ("GOut", std::ofstream::out);
  std::vector<double> lgf_values(8*Nx*Ny*Nz,0.);
  if(infileLGF.is_open()){
    std::string line;
    int xdir;
    int ydir; 
    int zdir;
    double lgf;
    while(getline(infileLGF, line)){
     
      for(char& c : line){
        if (c == '(' || c == ')'){
          c = ' ';
        }
      }

      // Remove extra spaces around commas
      size_t pos = line.find(", ");
      while (pos != std::string::npos) {
        line.erase(pos, 1);  // Erase space after comma
        pos = line.find(", ", pos);
      }

      // Remove leading and trailing spaces
      size_t first = line.find_first_not_of(" \t");
      size_t last = line.find_last_not_of(" \t");

      if (first != std::string::npos && last != std::string::npos) {
        line = line.substr(first, last - first + 1);
      }
      // std::cout << line << std::endl;
      std::stringstream ss(line);
      
      ss >> xdir;
      ss.ignore(1, ',');
      ss >> ydir;
      ss.ignore(1, ',');
      ss >> zdir;
      ss.ignore(1, ',');
      ss >> lgf;

      if((xdir >= (-Nx+originx) && xdir < (Nx+originx)) && (ydir >= (-Ny+originy) && ydir < (Ny+originy)) && (zdir >= (-Nz+originz) && zdir < (Nz+originz))){
        int index = zdir + Nz - originz + 2*Nz*(ydir + Ny - originy) + 4*Ny*Nz*(xdir + Nx - originx);
        lgf_values[index] = lgf; 
      }
    }    
  }
  else{
    std::cout << "Unable to open file" << std::endl;
  }
    int k0 = Nz/2 + 2*Nz*Ny/2 + 4*Nz*Ny*Nx/2;
   infileLGF.close();

// Using FFTW to compute the convolution

//****************************//
//         FFTW test
// ***************************//
// Input 1 = lgf_values -> domain doubled symbol
// Input 2 = Fpc_domaindouble

std::complex<double> zerocx(0.,0.);

//FFTW call to compute r2c dft (Symbol)
 std::ofstream lgf0 ("lgf0", std::ofstream::out);
 std::ofstream lgf1 ("out1", std::ofstream::out);
 std::ofstream lgf2 ("out2", std::ofstream::out);
 int doubledRToCSize = 2*Nx*2*Ny*(Nz+1);
 std::cout << "doubledRToCSize = " << doubledRToCSize << std::endl;
 std::vector<std::complex<double>> out1(doubledRToCSize,zerocx);

 out_r_doubled("outlgf",lgf_values.data(),Nx,Ny,Nz);
 {
   fftw_plan p1 = fftw_plan_dft_r2c_3d(
                                       domaindouble_x, domaindouble_y,
                                       domaindouble_z, lgf_values.data(),
                                       (fftw_complex*)out1.data(), FFTW_ESTIMATE);
   fftw_execute(p1);
 }
 out_c("outlgf_t",out1.data(),Nx,Ny,Nz);
 
 std::vector<std::complex<double>> out2(doubledRToCSize,zerocx);

 out_r_doubled("outRhs",Fpc_domaindouble,Nx,Ny,Nz);
 {
   fftw_plan p2 = fftw_plan_dft_r2c_3d(domaindouble_x, domaindouble_y,
                                       domaindouble_z, Fpc_domaindouble,
                                       (fftw_complex*)out2.data(), FFTW_ESTIMATE);
   fftw_execute(p2);
 }
 out_c("outRhs_t",out2.data(),Nx,Ny,Nz);
 
 std::vector<std::complex<double>> temp(doubledRToCSize,zerocx);
 for(int i = 0; i < doubledRToCSize; i++){
   temp[i] = out1[i] * out2[i];
 }
 
 std::vector<double> out_final(8*Nx*Ny*Nz,0.);

 out_c("outprod_t",temp.data(),Nx,Ny,Nz);
 {
   fftw_plan p3 = fftw_plan_dft_c2r_3d(domaindouble_x, domaindouble_y,
                                      domaindouble_z, (fftw_complex*)temp.data(),
                                      out_final.data(), FFTW_ESTIMATE);
   fftw_execute(p3);
 }

 double normalization = 1.0/(Nx*Ny*Nz*8);
 for (int index = 0; index < 8*Nx*Ny*Nz; index++)
   out_final[index] *= normalization;
 out_r_doubled("prod_r",out_final.data(),Nx,Ny,Nz);
 out_r_col("outfull",out_final.data(),2*Nx,2*Ny,2*Nz);
 double *extract_output_fftw = new double[Nx * Ny * Nz];
  // Extracting the output for the orginal size from the above domain doubled output
 for(int i = 0; i < Nx; i++){
   for(int j = 0; j < Ny; j++){
     for(int k = 0; k < Nz; k++){
       
       //Calculate the index in the domain doubled output vector,
       // corresponding to the low corner of the doubled domain.
       
       int index_dd = i*4*Ny*Nz + j*2*Nz + k;
       // Calculate the index in the smaller output of the orginal domain size 
       int original_index_fftw = i * Nz * Ny + j * Nz + k;
       // Copying the values from larger to smaller output vector
       extract_output_fftw[original_index_fftw] = out_final[index_dd];       
     }
   }
 }
 out_r_col(phiOutName,extract_output_fftw,Nx,Ny,Nz);
    return 0;
}
