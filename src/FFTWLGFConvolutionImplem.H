LGFConvolution::LGFConvolution(std::string a_lgffile,
                               double a_h,
                               int a_N)
{
  m_h = a_h;
  m_N = a_N;
  // Read in LGF values from file a_lgffile, and store in lgf_values
  // in row-major order.
  std::ifstream infileLGF(a_lgffile);
  std::vector<double> lgf_values(8*m_N*m_N*m_N,0.);
  int Nx = m_N;
  int Ny = m_N;
  int Nz = m_N;
  int domaindouble_x = 2*Nx;
  int domaindouble_y = 2*Ny;
  int domaindouble_z = 2*Nz;
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

      if(xdir >= -Nx && xdir < Nx && ydir >= -Ny && ydir < Ny && zdir >= -Nz && zdir < Nz){
        int index = zdir + Nz + 2*Nz*(ydir + Ny) + 4*Ny*Nz*(xdir + Nx);
        lgf_values[index] = lgf/a_h; // includes scaling of h=1 LGF to obtain G^h.
      }
    }    
  }
  else{
    std::cout << "Unable to open file" << std::endl;
  }
   infileLGF.close();
   // compute the Fourier transform of lgf_values and store it in m_Ghat.
   std::complex<double> zerocx(0.,0.);

   //FFTW call to compute r2c dft (Symbol)
 
   int doubledRToCSize = 2*Nx*2*Ny*(Nz+1);
   
   m_Ghat = std::vector<std::complex<double>>(doubledRToCSize,zerocx);
   fftw_plan p1 = fftw_plan_dft_r2c_3d(
                                       domaindouble_x, domaindouble_y,
                                       domaindouble_z, lgf_values.data(),
                                       (fftw_complex*)m_Ghat.data(), FFTW_ESTIMATE);
   fftw_execute(p1);
}

void LGFConvolution::convolveWithG(std::vector<double>& a_dataIn,
                                   std::vector<double>& a_dataOut)
{
  int Nx = m_N;
  int Ny = m_N;
  int Nz = m_N;
  int domaindouble_x = 2*Nx;
  int domaindouble_y = 2*Ny;
  int domaindouble_z = 2*Nz;
  int doubledRToCSize = 2*Nx*2*Ny*(Nz+1);
  std::complex<double> zerocx(0.,0.);
  std::vector<double> dataIndoubled(8*m_N*m_N*m_N,0.0);
  // Store a_dataIn in Upper corner of the domain-doubled box.
  
  for (int i = 0; i < Nx ; i++)
    {
      for (int j = 0; j < Ny; j++)
        {
          for (int k = 0;  k < Nz; k++)
            {
              int index_dd = (i + Nx)*4*Nz*Ny + (j + Ny)*2*Nz + k + Nz;
              int index_1d = i * Nz * Ny + j * Nz + k;
              dataIndoubled[index_dd] = a_dataIn[index_1d];              
            }
        }
    }
  // Apply rtoc FFT to domain-doubled input data.
  std::vector<std::complex<double>> out(doubledRToCSize,zerocx);
  fftw_plan p2 = fftw_plan_dft_r2c_3d(domaindouble_x, domaindouble_y,
                                      domaindouble_z, dataIndoubled.data(),
                                      (fftw_complex*)out.data(),
                                      FFTW_ESTIMATE);
  fftw_execute(p2);

  // compute product of m_Ghat and DFT of domain-doubled input.
  std::vector<std::complex<double>> product(doubledRToCSize,zerocx);
  for(int i = 0; i < doubledRToCSize; i++)
    {
      product[i] = m_Ghat[i] * out[i];
    }
  
  // apply ctor FFT to the product.
  std::vector<double> out_final(8*Nx*Ny*Nz,0.);
  fftw_plan p3 = fftw_plan_dft_c2r_3d(domaindouble_x, domaindouble_y,
                                      domaindouble_z,
                                      (fftw_complex*)product.data(),
                                      out_final.data(), FFTW_ESTIMATE);
  fftw_execute(p3);

  double normalization = pow(m_h,3)/(8*Nx*Ny*Nz);

  // Extracting the output for the orginal size from the above domain doubled output, and normalize.
  for (int i = 0; i < Nx; i++)
    {
      for (int j = 0; j < Ny; j++)
        {
          for (int k = 0; k < Nz; k++)
            {
              //Calculate the index in the domain doubled output vector,
              // corresponding to the low corner of the doubled domain.
       
              int index_dd = i*4*Ny*Nz + j*2*Nz + k;
              // Calculate the index in the smaller output of the
              // orginal domain  
              int original_index_fftw = i * Nz * Ny + j * Nz + k;
              // Copying the values from larger to smaller output vector
              a_dataOut[original_index_fftw] =
                out_final[index_dd]*normalization;     
            }
        }
    }
}
