#include <Grid/Grid.h>
#include <Grid/qcd/utils/A2Autils.h>
#include "A2A_Mat_Vec_util.h"

using namespace Grid;
using namespace std;

const int TSRC = 0;  //timeslice where rho is nonzero
const int VDIM = 2500; //length of each vector

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

  //Eigen::Tensor<ComplexD,5, Eigen::RowMajor> Mpp(momenta.size(),Gmu.size(),Nt,VDIM,VDIM);

  int moms = 4;
  int times = 24;

  Eigen::Tensor<ComplexD, 4, Eigen::RowMajor> M_test(moms ,times, VDIM, VDIM);
  Eigen::Tensor<ComplexD, 3, Eigen::RowMajor> Results(moms * times, VDIM, VDIM);
  //M_test = Mpp.chip(0, 0).chip(0, 0).chip(0, 0);

  Results.setZero();
  M_test.setConstant(ComplexD(0.0001, 0.0));
  //M_test.setRandom();
  //M_test.setZero();

  vector<int> contractions;
  for(int h=0; h<moms*times; h++) contractions.push_back(h);

  PipiA2Autils<WilsonImplR>::MesonField_MesonField_disconnected(M_test, Results, contractions, contractions, contractions);


  cout<< GridLogMessage << "=============== TEST RESULST MATRIX =================" <<endl;
  cout<< GridLogMessage << "COMPONENT (25, 2400, 2300) =  " << Results(25, 2400, 2300) << endl;
  cout<< GridLogMessage << "COMPONENT (50, 34, 2322) =  " << Results(50, 34, 2322) << endl;
  cout<< GridLogMessage << "COMPONENT (13, 13, 13) =  " << Results(13, 13, 13) << endl;
  cout<< GridLogMessage << "=====================================================" <<endl;

  // epilogue
  std::cout << GridLogMessage << "Grid is finalizing now" << std::endl;
  Grid_finalize();

  return EXIT_SUCCESS;
}
