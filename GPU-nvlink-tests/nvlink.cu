/*************************************************************************************
Erik Lundstrum 12/01/26
Based on Grid/tests/Test_meson_field.cc

*************************************************************************************/

#include <Grid/Grid.h>
#include <Grid/qcd/utils/A2Autils.h>
//#include "A2A_Mat_Vec_util.h"

using namespace Grid;
using namespace std;

const int TSRC = 0;  //timeslice where rho is nonzero
const int VDIM = 120; //length of each vector

typedef typename DomainWallFermionD::ComplexField ComplexField;
typedef typename DomainWallFermionD::FermionField FermionField;

int main(int argc, char *argv[])
{
  // initialization
  Grid_init(&argc, &argv);
  std::cout << GridLogMessage << "Grid initialized" << std::endl;

  // Lattice and rng setup 
  Coordinate latt_size   = GridDefaultLatt();
  Coordinate simd_layout = GridDefaultSimd(4, vComplex::Nsimd());
  Coordinate mpi_layout  = GridDefaultMpi();
  GridCartesian    grid(latt_size,simd_layout,mpi_layout);
  int Nt = GridDefaultLatt()[Tp];
  Lattice<iScalar<vInteger>> t(&grid);
  LatticeCoordinate(t, Tp);
  std::vector<int> seeds({1,2,3,4});
  GridParallelRNG          pRNG(&grid);
  pRNG.SeedFixedIntegers(seeds);


  int mom_size = 3;

  Eigen::Tensor<ComplexD,4, Eigen::RowMajor> Mpp(mom_size,Nt,VDIM,VDIM);

  Eigen::Map<Eigen::Matrix<ComplexD, Eigen::Dynamic, 1>>
    flat(Mpp.data(), Mpp.size());

  flat.setRandom();   // writes directly into Mpp

  cout << GridLogMessage << "Testing random matrix entry (1,1,100,100) = " << Mpp(1,1,100,100) << endl;

  // epilogue
  std::cout << GridLogMessage << "Grid is finalizing now" << std::endl;
  Grid_finalize();
  
  return EXIT_SUCCESS;
}

