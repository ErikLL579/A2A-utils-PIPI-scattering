
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
  int Nd = grid.Dimensions();
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

 
// CONTRACTION VECTORS FOR MATRIX-MATRIX MULTPLIES  //
  vector<int> A_vector_contractions = {3,3,1,1,2,2,0,0,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,};
  vector<int> B_vector_contractions = {1,0,3,2,1,0,3,2,2,5,2,4,0,7,0,6,3,1,3,0,1,3,1,2,};
  vector<int> C_vector_contractions = {0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,};
 
// BUFFER FLAGS FOR MATRIX-MATRIX MULTPLIES  //
  vector<int> buffer_flag_A = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,};
  vector<int> buffer_flag_B = {0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,};
 
// PERFORM MATRIX MULTIPLICATIONS //

  PipiA2Autils<WilsonImplR>::MesonField_MesonField_connected(Pion_source_fields, 
                                                             Pion_product_fields_level_1,
                                                             Pion_product_fields_level_2,
                                                             num_products_level_1, 
                                                             buffer_flag_A,
                                                             buffer_flag_B,
                                                             A_vector_contractions,
                                                             B_vector_contractions,
                                                             C_vector_contractions);


// ====================================================== //
// ========= COMPLETE MATRIX-MATRIX OPERATIONS ========== //
// ====================================================== //


// ======================================= //
// ===== GENERATE PHOTON PROPAGATORs ===== //
// ======================================= //


  RealD min = 1.0;
  vector<ComplexField> phtn_mu_nu(Nd*Nd, &grid);


  IVPhotonPropagator<WilsonImplR>::CoulombGaugeMomentumSpace( &phtn_mu_nu[0], min);


// ======================================= //
// == DEFINE ALL MATRIX-VECTOR PRODUCTS == //
// ======================================= //


  std::vector<FermionField> prod_vec1_mu(VDIM,&grid);
  for(int i=0; i<VDIM; i++) prod_vec1_mu[i] = Zero();


  std::vector<FermionField> prod_vec2_mu(VDIM,&grid);
  for(int i=0; i<VDIM; i++) prod_vec2_mu[i] = Zero();


  std::vector<FermionField> prod_vec3_mu(VDIM,&grid);
  for(int i=0; i<VDIM; i++) prod_vec3_mu[i] = Zero();


  std::vector<FermionField> prod_vec4_mu(VDIM,&grid);
  for(int i=0; i<VDIM; i++) prod_vec4_mu[i] = Zero();


  std::vector<FermionField> prod_vec5_mu(VDIM,&grid);
  for(int i=0; i<VDIM; i++) prod_vec5_mu[i] = Zero();


  std::vector<FermionField> prod_vec6_mu(VDIM,&grid);
  for(int i=0; i<VDIM; i++) prod_vec6_mu[i] = Zero();


  std::vector<FermionField> prod_vec7_mu(VDIM,&grid);
  for(int i=0; i<VDIM; i++) prod_vec7_mu[i] = Zero();


  std::vector<FermionField> prod_vec8_mu(VDIM,&grid);
  for(int i=0; i<VDIM; i++) prod_vec8_mu[i] = Zero();


  std::vector<FermionField> prod_vec9_mu(VDIM,&grid);
  for(int i=0; i<VDIM; i++) prod_vec9_mu[i] = Zero();


  std::vector<FermionField> prod_vec10_mu(VDIM,&grid);
  for(int i=0; i<VDIM; i++) prod_vec10_mu[i] = Zero();


  std::vector<FermionField> prod_vec11_mu(VDIM,&grid);
  for(int i=0; i<VDIM; i++) prod_vec11_mu[i] = Zero();


  std::vector<FermionField> prod_vec12_mu(VDIM,&grid);
  for(int i=0; i<VDIM; i++) prod_vec12_mu[i] = Zero();


  std::vector<FermionField> prod_vec13_mu(VDIM,&grid);
  for(int i=0; i<VDIM; i++) prod_vec13_mu[i] = Zero();


  std::vector<FermionField> prod_vec14_mu(VDIM,&grid);
  for(int i=0; i<VDIM; i++) prod_vec14_mu[i] = Zero();


  std::vector<FermionField> prod_vec15_mu(VDIM,&grid);
  for(int i=0; i<VDIM; i++) prod_vec15_mu[i] = Zero();


  std::vector<FermionField> prod_vec16_mu(VDIM,&grid);
  for(int i=0; i<VDIM; i++) prod_vec16_mu[i] = Zero();


  std::vector<FermionField> prod_vec17_mu(VDIM,&grid);
  for(int i=0; i<VDIM; i++) prod_vec17_mu[i] = Zero();


  std::vector<FermionField> prod_vec18_mu(VDIM,&grid);
  for(int i=0; i<VDIM; i++) prod_vec18_mu[i] = Zero();


  std::vector<FermionField> prod_vec19_mu(VDIM,&grid);
  for(int i=0; i<VDIM; i++) prod_vec19_mu[i] = Zero();


  std::vector<FermionField> prod_vec20_mu(VDIM,&grid);
  for(int i=0; i<VDIM; i++) prod_vec20_mu[i] = Zero();


  std::vector<FermionField> prod_vec21_mu(VDIM,&grid);
  for(int i=0; i<VDIM; i++) prod_vec21_mu[i] = Zero();


  std::vector<FermionField> prod_vec22_mu(VDIM,&grid);
  for(int i=0; i<VDIM; i++) prod_vec22_mu[i] = Zero();


  std::vector<FermionField> prod_vec23_mu(VDIM,&grid);
  for(int i=0; i<VDIM; i++) prod_vec23_mu[i] = Zero();


  std::vector<FermionField> prod_vec24_mu(VDIM,&grid);
  for(int i=0; i<VDIM; i++) prod_vec24_mu[i] = Zero();


  std::vector<FermionField> prod_vec25_mu(VDIM,&grid);
  for(int i=0; i<VDIM; i++) prod_vec25_mu[i] = Zero();


  std::vector<FermionField> prod_vec26_mu(VDIM,&grid);
  for(int i=0; i<VDIM; i++) prod_vec26_mu[i] = Zero();


  std::vector<FermionField> prod_vec27_mu(VDIM,&grid);
  for(int i=0; i<VDIM; i++) prod_vec27_mu[i] = Zero();


  std::vector<FermionField> prod_vec28_mu(VDIM,&grid);
  for(int i=0; i<VDIM; i++) prod_vec28_mu[i] = Zero();


// ======================================= //
// ===== BEGIN MAT-VECTOR OPERATIONS ===== //
// ======================================= //

                                      
  PipiA2Autils<WilsonImplR>::ContractMesonFieldAndVector(&prod_vec1_mu[0], Pion_source_fields[3], &v[0]);

                                      
  PipiA2Autils<WilsonImplR>::ContractMesonFieldAndVector(&prod_vec2_mu[0], Pion_source_fields[1], &v[0]);

                                      
  PipiA2Autils<WilsonImplR>::ContractMesonFieldAndVector(&prod_vec3_mu[0], Pion_source_fields[2], &v[0]);

                                      
  PipiA2Autils<WilsonImplR>::ContractMesonFieldAndVector(&prod_vec4_mu[0], Pion_source_fields[0], &v[0]);

                                      
  PipiA2Autils<WilsonImplR>::ContractMesonFieldAndVector(&prod_vec5_mu[0], Pion_product_fields_level_1[0], &v[0]);

                                      
  PipiA2Autils<WilsonImplR>::ContractMesonFieldAndVector(&prod_vec6_mu[0], Pion_product_fields_level_1[1], &v[0]);

                                      
  PipiA2Autils<WilsonImplR>::ContractMesonFieldAndVector(&prod_vec7_mu[0], Pion_product_fields_level_1[2], &v[0]);

                                      
  PipiA2Autils<WilsonImplR>::ContractMesonFieldAndVector(&prod_vec8_mu[0], Pion_product_fields_level_1[3], &v[0]);

                                      
  PipiA2Autils<WilsonImplR>::ContractMesonFieldAndVector(&prod_vec9_mu[0], Pion_product_fields_level_1[4], &v[0]);

                                      
  PipiA2Autils<WilsonImplR>::ContractMesonFieldAndVector(&prod_vec10_mu[0], Pion_product_fields_level_1[5], &v[0]);

                                      
  PipiA2Autils<WilsonImplR>::ContractMesonFieldAndVector(&prod_vec11_mu[0], Pion_product_fields_level_1[6], &v[0]);

                                      
  PipiA2Autils<WilsonImplR>::ContractMesonFieldAndVector(&prod_vec12_mu[0], Pion_product_fields_level_1[7], &v[0]);

                                      
  PipiA2Autils<WilsonImplR>::ContractMesonFieldAndVector(&prod_vec13_mu[0], Pion_product_fields_level_2[1], &v[0]);

                                      
  PipiA2Autils<WilsonImplR>::ContractMesonFieldAndVector(&prod_vec14_mu[0], Pion_product_fields_level_2[2], &v[0]);

                                      
  PipiA2Autils<WilsonImplR>::ContractMesonFieldAndVector(&prod_vec15_mu[0], Pion_product_fields_level_2[3], &v[0]);

                                      
  PipiA2Autils<WilsonImplR>::ContractMesonFieldAndVector(&prod_vec16_mu[0], Pion_product_fields_level_2[4], &v[0]);

                                      
  PipiA2Autils<WilsonImplR>::ContractMesonFieldAndVector(&prod_vec17_mu[0], Pion_product_fields_level_2[5], &v[0]);

                                      
  PipiA2Autils<WilsonImplR>::ContractMesonFieldAndVector(&prod_vec18_mu[0], Pion_product_fields_level_2[6], &v[0]);

                                      
  PipiA2Autils<WilsonImplR>::ContractMesonFieldAndVector(&prod_vec19_mu[0], Pion_product_fields_level_2[7], &v[0]);

                                      
  PipiA2Autils<WilsonImplR>::ContractMesonFieldAndVector(&prod_vec20_mu[0], Pion_product_fields_level_2[8], &v[0]);

                                      
  PipiA2Autils<WilsonImplR>::ContractMesonFieldAndVector(&prod_vec21_mu[0], Pion_product_fields_level_2[9], &v[0]);

                                      
  PipiA2Autils<WilsonImplR>::ContractMesonFieldAndVector(&prod_vec22_mu[0], Pion_product_fields_level_2[10], &v[0]);

                                      
  PipiA2Autils<WilsonImplR>::ContractMesonFieldAndVector(&prod_vec23_mu[0], Pion_product_fields_level_2[11], &v[0]);

                                      
  PipiA2Autils<WilsonImplR>::ContractMesonFieldAndVector(&prod_vec24_mu[0], Pion_product_fields_level_2[12], &v[0]);

                                      
  PipiA2Autils<WilsonImplR>::ContractMesonFieldAndVector(&prod_vec25_mu[0], Pion_product_fields_level_2[13], &v[0]);

                                      
  PipiA2Autils<WilsonImplR>::ContractMesonFieldAndVector(&prod_vec26_mu[0], Pion_product_fields_level_2[14], &v[0]);

                                      
  PipiA2Autils<WilsonImplR>::ContractMesonFieldAndVector(&prod_vec27_mu[0], Pion_product_fields_level_2[15], &v[0]);

                                      
  PipiA2Autils<WilsonImplR>::ContractMesonFieldAndVector(&prod_vec28_mu[0], Pion_product_fields_level_1[8], &v[0]);


// epilogue
  std::cout << GridLogMessage << "Grid is finalizing now" << std::endl;
  Grid_finalize();

  return EXIT_SUCCESS;
}
