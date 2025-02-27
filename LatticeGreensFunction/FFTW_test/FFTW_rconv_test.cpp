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
    int cx = Nx/2;
    int cy = Ny/2;
    int cz = Nz/2;

    // 3D point charge field
    for(int i = 0; i < Nx; i++){
        for(int j = 0; j < Ny; j++){
            for(int k = 0; k < Nz; k++){
                // index for 1D F
               // int index_f = i * Nz * Ny + j * Nz + k;

                if(i == cx && j == cy && k == cz){
                    Fpc_3D[i][j][k] = 1.0/pow(h, 3);
                }
                else{
                    Fpc_3D[i][j][k] = 0.0;
                }
                //Fpc_1D[index_f] = Fpc_3D[i][j][k];
                //std::cout << "index_f=" << index_f << "\t" << "Fpc_1D = " <<Fpc_1D[index_f] << std::endl;
            }
        }
    }

    // Domain doubling the point charge field. It will act as the second input for rconv
    
    int domaindouble_x = 2*Nx;
    int domaindouble_y = 2*Ny;
    int domaindouble_z = 2*Nz;
    double Fpc_3D_dd[domaindouble_x][domaindouble_y][domaindouble_z];

    double* Fpc_domaindouble = new double [domaindouble_x * domaindouble_y * domaindouble_z];

    for(int i = 0; i < domaindouble_x; i++){
        for(int j = 0; j < domaindouble_y; j++){
            for(int k = 0; k < domaindouble_z; k++){

                //int index_dd = i*domaindouble_y*domaindouble_z + j*domaindouble_z + k;
                if((i < Nx) && (j >= Ny && j < domaindouble_y) && (k < Nz)){
                    //int index_1d = i * Nz * Ny + j * Nx + k;
                    //Fpc_domaindouble[index_dd] = Fpc_1D[index_1d]; 
                    Fpc_3D_dd[i][j][k] = Fpc_3D[i][j - Ny][k];    
                }
                else{
                    Fpc_3D_dd[i][j][k] = 0.0;
                    //Fpc_domaindouble[index_dd] = 0.0;     
                }
               // std::cout << "i = " << i << " ,j = " <<j <<" ,k = "<< k << "\t" << "Fpc_1D = " <<Fpc_3D_dd[i][j][k] << std::endl;
                int index_dd = i*domaindouble_y*domaindouble_z + j*domaindouble_z + k;
                Fpc_domaindouble[index_dd] = Fpc_3D_dd[i][j][k];
                //std::cout << "index_dd=" << index_dd << "\t" << "Fpc_1D_dd = " <<Fpc_domaindouble[index_dd] << std::endl;
            }
        }
    }


  //printf("Hello from test!!\n");
    
    // Lattice Green's function
  std::ifstream infileLGF("/home/h82/Documents/Bluestone/P3M/MLC_PPM/LatticeGreensFunction/exec/phiTrimmed");
  std::vector<double> lgf_values(8*Nx*Ny*Nz);
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

      if((xdir >= (-Nx) && xdir < (Nx)) && (ydir >= (-Ny) && ydir < (Ny)) && (zdir >= (-Nz) && zdir < (Nz))){
          int index_lgf = zdir + Nz + 2*Ny*(ydir + Ny) + 4*Ny*Nx*(xdir+Nx);
          lgf_values[index_lgf] = lgf;
          //lgf_values.push_back(lgf);
          // if(xdir == 0 && ydir == 0 && zdir == 0){
          //   lgf_values.push_back(1.0);
          // }
          // else{
          //   lgf_values.push_back(0.0);
          // }
          //std::cout<<"xdir = " << xdir << ",\t ydir = " << ydir << ",\t zdir = " << zdir << ",\t index_lgf = " << index_lgf<< ",\t"<< std::setprecision(10) << lgf_values[index_lgf] << std::endl; 
      }
    }
    // for(double lgfv : lgf_values){
    //   std::cout << std::setprecision(10) << lgfv << std::endl;
    // }
    infileLGF.close();
  }
  else{
    std::cout << "Unable to open file" << std::endl;
  }
  std::cout << "Size of LGF Vector=" << lgf_values.size()<<std::endl;

  for(int i = 0; i < domaindouble_x*domaindouble_y*domaindouble_z; i++){
    std::cout <<""<<lgf_values[i] << std::endl;
  }

// double greens_func; 
// // Computing Green's function in real space 1/4*pi*r
// for(int i = 0; i < domaindouble_x; i++){
//   for(int j = 0; j < domaindouble_y; j++){
//     for(int k = 0; k < domaindouble_z; k++){
//       int idx = i * domaindouble_z * domaindouble_y + j * domaindouble_z + k;
//       int x0 = i - cx;
//       int x1 = j - cy;
//       int x2 = k - cz;
//       // Compute the radius r = sqrt(x0^2 + x1^2 + x2^2)
//       double r = std::sqrt(x0 * x0 + x1 * x1 + x2 * x2) * h;

//       // Green's function 1/4*pi*r 
//       if(r > 0){
//         greens_func = 1.0/(4.0 * M_PI * r);
//       }
//       else{
//         greens_func = 0.0;
//       }
//       lgf_values.push_back(greens_func);

//       std::cout << "idx = " << idx << "\t"<<lgf_values[idx] << std::endl;
//     }
//   }
// }

// Using FFTW to compute the convolution

//****************************//
//         FFTW test
// ***************************//
// Input 1 = lgf_values -> domain doubled symbol
// Input 2 = Fpc_domaindouble

std::vector<std::complex<double>> temp(domaindouble_x * domaindouble_y * ((domaindouble_z/2)+1));
std::vector<std::complex<double>> out1(domaindouble_x * domaindouble_y * ((domaindouble_z/2)+1));
std::vector<std::complex<double>> out2(domaindouble_x * domaindouble_y * ((domaindouble_z/2)+1));
std::vector<double> out_final(domaindouble_x * domaindouble_y * domaindouble_z);
std::vector<double> out_normalize(domaindouble_x * domaindouble_y * domaindouble_z);

//FFTW call to compute r2c dft (Symbol)
fftw_plan p1 = fftw_plan_dft_r2c_3d(domaindouble_x, domaindouble_y, domaindouble_z, lgf_values.data(),
                  (fftw_complex*)out1.data(), FFTW_ESTIMATE);
fftw_execute(p1);                  
// for(int i0 = 0; i0 < domaindouble_x * domaindouble_y * ((domaindouble_z/2)+1); i0++){      
//          std::cout << " i0 =" << i0 << "\t" << out1[i0] << std::endl;
// }
fftw_plan p2 = fftw_plan_dft_r2c_3d(domaindouble_x, domaindouble_y, domaindouble_z, Fpc_domaindouble ,
                  (fftw_complex*)out2.data(), FFTW_ESTIMATE);
fftw_execute(p2);
printf("Input 2!!!\n");
// for(int i0 = 0; i0 < domaindouble_x * domaindouble_y * ((domaindouble_z/2)+1); i0++){      
//          std::cout << " i0 =" << i0 << "\t out1[i0] = " << out1[i0] << ",\t out2[i0] = " <<out2[i0] << ",\t out1 - out2 = " << out1[i0] - out2[i0]<< std::endl;
// }

// Pointwise Mulitply

auto complex_multiply = std::multiplies<
                            std::complex<double>>{}; 
    std::transform(out2.begin(), //start location 
                out2.end(), //end location
                out1.data(), //2nd input
                temp.begin(), //output 
                complex_multiply); //operator

// for(int i = 0; i < domaindouble_x * domaindouble_y * ((domaindouble_z/2)+1); i++){
//   //temp[i] = out2[i] * out1[i];
//   std::cout << "i = "<< i << " temp =  " << temp[i] << std::endl;
// }

// Inverse FFT using c2r
fftw_plan p3 = fftw_plan_dft_c2r_3d(domaindouble_x, domaindouble_y, domaindouble_z, (fftw_complex*)temp.data(),
                  out_final.data(), FFTW_ESTIMATE);

fftw_execute(p3);


// Normalizing the output : Mulitply with 1/(2N)^(9/2)
//double norm_sqrt_dd = sqrt(domaindouble_x);
//double norm_factor = 1.0/(pow(norm_sqrt_dd, 9));
double norm_factor = 1.0/(domaindouble_x * domaindouble_y * domaindouble_z);
for(int i = 0; i < domaindouble_x*domaindouble_y*domaindouble_z; i++){
  out_normalize[i] = norm_factor * out_final[i];
}


double *extract_output_fftw = new double[Nx * Ny * Nz];
double *lgf_extract_test = new double[Nx * Ny * Nz];
  // Extracting the output for the orginal size from the above domain doubled output
  for(int i = 0; i < Nx; i++){
    for(int j = 0; j < Ny; j++){
      for(int k = 0; k < Nz; k++){
        //Calculate the index in the domain doubled output vector
            int dd_index_fftw = i * domaindouble_y * domaindouble_z + j * domaindouble_z + Ny + k;
        // Calculate the index in the smaller output of the orginal domain size 
            int original_index_fftw = i * Nz * Ny + j * Nz + k;
        // Copying the values from larger to smaller output vector
          extract_output_fftw[original_index_fftw] = out_normalize[dd_index_fftw];

          //lgf vector orginal data without domain double
          lgf_extract_test[original_index_fftw] = lgf_values[dd_index_fftw];
      }
    }
  }
for(int i0 = 0; i0 < Nx * Ny * Nz; i0++){      
         std::cout << " i0 = " << i0 << "\t Conv Output ="<< extract_output_fftw[i0] << ",\t Original LGF = " << lgf_extract_test[i0] << std::endl;
}

    return 0;
}