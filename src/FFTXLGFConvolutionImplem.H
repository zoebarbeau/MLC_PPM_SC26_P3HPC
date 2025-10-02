void write_view_to_txt(const Kokkos::View<double*, Kokkos::HostSpace>& view,
                       const std::string& filename) {
  std::ofstream outfile(filename);
  if (!outfile.is_open()) {
    std::cerr << "Unable to open file " << filename << "\n";
    return;
  }
  for (size_t i = 0; i < view.extent(0); ++i) {
    outfile << i << " " << view(i) << "\n";
  }
  outfile.close();
}


template <typename ExecutionSpace>
void ConvolutionFFTX<ExecutionSpace>::read_LGF_file(const std::string& filename) {
  std::ifstream infileLGF(filename);
  if (!infileLGF.is_open()) {
    std::cerr << "Unable to open LGF file\n";
    return;
  }

  std::string line;
  int xdir, ydir, zdir;
  double lgf;

  while (getline(infileLGF, line)) {
    for (char& c : line) if (c == '(' || c == ')') c = ' ';
    size_t pos = line.find(", ");
    while (pos != std::string::npos) {
      line.erase(pos, 1);
      pos = line.find(", ", pos);
    }

    line.erase(0, line.find_first_not_of(" \t"));
    line.erase(line.find_last_not_of(" \t") + 1);

    std::stringstream ss(line);
    ss >> xdir; ss.ignore(1, ',');
    ss >> ydir; ss.ignore(1, ',');
    ss >> zdir; ss.ignore(1, ',');
    ss >> lgf;

    if (xdir >= 0 && xdir <= extent && ydir >= 0 && ydir <= extent && zdir >= 0 && zdir <= extent) {
      for (int isign : {-1, 1})
        for (int jsign : {-1, 1})
          for (int ksign : {-1, 1}) {
            int x = xdir * isign + extent;
            int y = ydir * jsign + extent;
            int z = zdir * ksign + extent;

            if (x < domaindouble_x && y < domaindouble_y && z < domaindouble_z) {
              int index = z + domaindouble_z * (y + domaindouble_y * x);
              lgf_values[index] = lgf / h;
            }
          }
    }
  }

  infileLGF.close();

  auto host_LGF = Kokkos::create_mirror_view(dev_LGF);
  for (size_t i = 0; i < lgf_values.size(); ++i) host_LGF(i) = lgf_values[i];
  Kokkos::deep_copy(dev_LGF, host_LGF);


  std::vector<void*> args1 = [&]() {
    static  auto symbol_data = symbol.data();
    static  auto lgf_data = dev_LGF.data();
    static  auto dummy1_data = dummy1.data();
    return std::vector<void*>{&symbol_data, &lgf_data, &dummy1_data};
  }();


  // Calculate the symbol i.e. forward DFT of the lattice Green's function
  std::vector<int> sizes{domaindouble_x, domaindouble_y, domaindouble_z};
  MDPRDFTProblem r2cdft{args1, sizes, "mdprdft"};
  r2cdft.transform();

}

template <typename ExecutionSpace>
template <typename ProblemManagerType>
void ConvolutionFFTX<ExecutionSpace>::compute_convolution(const ExecutionSpace& exec_space, ProblemManagerType& pm, double mdtime[45], double imdtime[45],int it) {
  auto F = pm.get(ExaMPM::Location::Node(), ExaMPM::Field::F());
  auto velx = pm.get(ExaMPM::Location::Node(), ExaMPM::Field::velx());
  Kokkos::deep_copy(velx, 0.0);
  // Assume MDPRDFTProblem and IMDPRDFTProblem are properly defined elsewhere
 // Kokkos::View<double*, memory_space> dummy1("dummy1", domaindouble_x * domaindouble_y * domaindouble_z);
   double *F1D_vec = F1D.data();

   for(int d = 0; d < 3; d++){
      Kokkos::parallel_for("Copy 4D to 1D", Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<3>>(exec_space,{0, 0, 0}, {extent, extent, extent}),
          KOKKOS_LAMBDA(const int i, const int j, const int k) {
              int index = i * extent * extent + j * extent + k;
              F1D(index) = F(i, j, k, d);
          });

      auto host_F1D = Kokkos::create_mirror_view(F1D);
      Kokkos::deep_copy(host_F1D, F1D);
      std::stringstream ss1, ss2;
      ss1 << "F1D_dir" << d << ".txt";
      write_view_to_txt(host_F1D, ss1.str());

      Kokkos::parallel_for("Place F1D in doubledomain", Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<3>>(exec_space,{0, 0, 0}, {extent, extent, extent}),
           KOKKOS_LAMBDA(const int k, const int j, const int i) {
            int index_dd = (k + extent)*4*extent*extent + (j + extent)*2*extent + i + extent;
            int index_1d = k * extent * extent + j * extent + i;
            F1D_domaindouble[index_dd] = F1D[index_1d];
        });

      std::vector<void*> args2 = [&]() {
      static auto Fdft_data = F_dft.data();
      static auto F1D_data = F1D_domaindouble.data();
      static auto dummy2_data = dummy2.data();
      return std::vector<void*>{&Fdft_data, &F1D_data, &dummy2_data};
      }();

      std::vector<int> sizes2{domaindouble_x, domaindouble_y, domaindouble_z};
      MDDFTProblem c2cdft2{args2, sizes2, "mdprdft"};
      c2cdft2.transform();
      mdtime[it+d] = c2cdft2.getTime();
      

      Kokkos::parallel_for("Pointwise_multiply", Kokkos::RangePolicy<ExecutionSpace>(exec_space,0,domaindouble_x * domaindouble_y * ((domaindouble_z/2)+1)),
        KOKKOS_LAMBDA(const int i) {
            double a = symbol(i).real();
            double b = symbol(i).imag();
            double c = F_dft(i).real();
            double d = F_dft(i).imag();

            //Pointwise multiply symbol and Fpc_dft
            double real_pw = a * c - b * d;
            double img_pw = a * d + b * c;
             pointwise_mul[i] = Complex(real_pw, img_pw);
       });

      std::vector<void*> args3 = [&]() {
      static auto out_data = out_idft.data();
      static auto pwise_data = pointwise_mul.data();
      static auto dummy3_data = dummy3.data();
      return std::vector<void*>{&out_data, &pwise_data, &dummy3_data};
      }();


      std::vector<int> sizes3{domaindouble_x, domaindouble_y, domaindouble_z};
      IMDDFTProblem c2cdft{args3, sizes3, "imdprdft"};
      c2cdft.transform();
      imdtime[it+d] = c2cdft.getTime();

      double norm_factor = pow(h,3)/(domaindouble_x * domaindouble_y * domaindouble_z);
      Kokkos::parallel_for("Normalize output", Kokkos::RangePolicy<ExecutionSpace>(exec_space,0,domaindouble_x * domaindouble_y * domaindouble_z),
        KOKKOS_LAMBDA(const int i) {

            out_normalize[i] = norm_factor * out_idft[i];
     });


     Kokkos::parallel_for("Normalize output", Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<3>>(exec_space, {0, 0, 0}, {extent, extent, extent}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
            int out_dd_index = k * domaindouble_y * domaindouble_x + j * domaindouble_x + i;
            int out_original_index = k * extent * extent + j * extent + i;
            conv_output[out_original_index] = out_normalize[out_dd_index];
     });


     auto velocity_g = pm.get(ExaMPM::Location::Node(), ExaMPM::Field::Velocity());
     int N = extent;
     double U = 1.0;

     if(d == 2){
     Kokkos::parallel_for("Copy 1D to 3D", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {N,N,N}),
        KOKKOS_LAMBDA(const int i, const int j, const int k) {

              int index_f = i * N * N + j * N + k;
              velocity_g(i,j,k,d) = 1.0;
              velx(i,j,k,0)       = 1.0;

     });


     }


//       Kokkos::fence();
       Kokkos::parallel_for("Copy 1D to 3D", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {N,N,N}),
        KOKKOS_LAMBDA(const int i, const int j, const int k) {
          int index_f = i * N * N + j * N + k;
          double x[3] = {i*h-0.5,j*h-0.5,k*h-0.5};
          double v_exact[3],v_error[3];
          double r =  sqrt(x[1]*x[1] + x[2]*x[2]+x[0]*x[0]);
          double R = 0.25;
          double R2 = 0.25*0.25;
          double U = 1;
          double R3 = 0.25*0.25*0.25;
          double xz = x[0]*x[2];
          double yz = x[1]*x[2];

          
          if(d == 2 ){
            velocity_g(i,j,k,d) += conv_output[index_f] ;
            velx(i,j,k,0)  += (conv_output[index_f]);

          }else if(d == 1){

            velocity_g(i,j,k,d) = conv_output[index_f];
            velx(i,j,k,0)  = conv_output[index_f];
          }else if(d == 0){

            velocity_g(i,j,k,d) = conv_output[index_f];
            velx(i,j,k,0) =  conv_output[index_f];

          }

          });
}



 

}
