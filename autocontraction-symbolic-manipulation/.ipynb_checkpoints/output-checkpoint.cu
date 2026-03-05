
#include <Grid/Grid.h>
#include <Grid/qcd/utils/A2Autils.h>
#include "A2A_Mat_Vec_util.h"

using namespace Grid;
using namespace std;

const int TSRC = 0;  //timeslice where rho is nonzero
const int VDIM = 100; //length of each vector

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

  // initial w and v vectors (need to import these correctly)
  std::vector<FermionField> w(VDIM,&grid);
  std::vector<FermionField> v(VDIM,&grid);


// ====================================================== //
// == DEFINE AND CONSTRUCT PION SOURCE & PROD MATRICES == //
// ====================================================== //


  int times = 4;
  Eigen::Tensor<ComplexD, 3, Eigen::RowMajor> Pion_source_fields(times, VDIM, VDIM);


  int num_products_level_1 = 8;
  Eigen::Tensor<ComplexD, 3, Eigen::RowMajor> Pion_product_fields_level_1(num_products, VDIM, VDIM);


  int num_products_level_2 = 16;
  Eigen::Tensor<ComplexD, 3, Eigen::RowMajor> Pion_product_fields_level_2(num_products, VDIM, VDIM);

  vector<int> A_vector_contractions = {3,3,1,1,2,2,0,0,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,};
  vector<int> B_vector_contractions = {1,0,3,2,1,0,3,2,2,5,2,4,0,7,0,6,3,1,3,0,1,3,1,2,};
  vector<int> C_vector_contractions = {0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,};

// ====================================================== //
// ====================================================== //


// ======================================= //
// == DEFINE ALL MATRIX-VECTOR PRODUCTS == //
// ======================================= //


  std::vector<FermionField> prod_vec1_mu(VDIM,&grid);
  prod_vec1_mu = Zero();


  std::vector<FermionField> prod_vec2_mu(VDIM,&grid);
  prod_vec2_mu = Zero();


  std::vector<FermionField> prod_vec3_mu(VDIM,&grid);
  prod_vec3_mu = Zero();


  std::vector<FermionField> prod_vec4_mu(VDIM,&grid);
  prod_vec4_mu = Zero();


  std::vector<FermionField> prod_vec5_mu(VDIM,&grid);
  prod_vec5_mu = Zero();


  std::vector<FermionField> prod_vec6_mu(VDIM,&grid);
  prod_vec6_mu = Zero();


  std::vector<FermionField> prod_vec7_mu(VDIM,&grid);
  prod_vec7_mu = Zero();


  std::vector<FermionField> prod_vec8_mu(VDIM,&grid);
  prod_vec8_mu = Zero();


  std::vector<FermionField> prod_vec9_mu(VDIM,&grid);
  prod_vec9_mu = Zero();


  std::vector<FermionField> prod_vec10_mu(VDIM,&grid);
  prod_vec10_mu = Zero();


  std::vector<FermionField> prod_vec11_mu(VDIM,&grid);
  prod_vec11_mu = Zero();


  std::vector<FermionField> prod_vec12_mu(VDIM,&grid);
  prod_vec12_mu = Zero();


  std::vector<FermionField> prod_vec13_mu(VDIM,&grid);
  prod_vec13_mu = Zero();


  std::vector<FermionField> prod_vec14_mu(VDIM,&grid);
  prod_vec14_mu = Zero();


  std::vector<FermionField> prod_vec15_mu(VDIM,&grid);
  prod_vec15_mu = Zero();


  std::vector<FermionField> prod_vec16_mu(VDIM,&grid);
  prod_vec16_mu = Zero();


  std::vector<FermionField> prod_vec17_mu(VDIM,&grid);
  prod_vec17_mu = Zero();


  std::vector<FermionField> prod_vec18_mu(VDIM,&grid);
  prod_vec18_mu = Zero();


  std::vector<FermionField> prod_vec19_mu(VDIM,&grid);
  prod_vec19_mu = Zero();


  std::vector<FermionField> prod_vec20_mu(VDIM,&grid);
  prod_vec20_mu = Zero();


  std::vector<FermionField> prod_vec21_mu(VDIM,&grid);
  prod_vec21_mu = Zero();


  std::vector<FermionField> prod_vec22_mu(VDIM,&grid);
  prod_vec22_mu = Zero();


  std::vector<FermionField> prod_vec23_mu(VDIM,&grid);
  prod_vec23_mu = Zero();


  std::vector<FermionField> prod_vec24_mu(VDIM,&grid);
  prod_vec24_mu = Zero();


  std::vector<FermionField> prod_vec25_mu(VDIM,&grid);
  prod_vec25_mu = Zero();


  std::vector<FermionField> prod_vec26_mu(VDIM,&grid);
  prod_vec26_mu = Zero();


  std::vector<FermionField> prod_vec27_mu(VDIM,&grid);
  prod_vec27_mu = Zero();


  std::vector<FermionField> prod_vec28_mu(VDIM,&grid);
  prod_vec28_mu = Zero();


// ======================================= //
// ======================================= //


// epilogue
  std::cout << GridLogMessage << "Grid is finalizing now" << std::endl;
  Grid_finalize();

  return EXIT_SUCCESS;
}
