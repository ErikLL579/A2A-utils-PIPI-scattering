/*************************************************************************************
Erik Lundstrum 12/01/26
Based on Grid/tests/Test_meson_field.cc

*************************************************************************************/

#include <Grid/Grid.h>
#include <Grid/qcd/utils/A2Autils.h>
#include "A2A_Mat_Vec_util.h"
#include "load_data.h"

using namespace Grid;
using namespace std;

const int TSRC = 0;  //timeslice where rho is nonzero
const int VDIM = 2768; //length of each vector

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
  int Nd = grid.Dimensions();

  Lattice<iScalar<vInteger>> t(&grid);
  LatticeCoordinate(t, Tp);
  std::vector<int> seeds({1,2,3,4});
  GridParallelRNG          pRNG(&grid);
  pRNG.SeedFixedIntegers(seeds);

  int Nt = 64;
  auto mf = loadMesonFields("/global/cfs/cdirs/mp13/lundstrum/Masaaki_data/traj_5070/pion_mom000.h5", "pion_mom000", Nt);

  cout << GridLogMessage << "meson field t= 0 test component (0,0) = " << mf[0](0, 0) << endl;


  int Nmodes = 2768;

  // light quark vectors, binSize=173 → 2768/173 = 16 bin files
  std::vector<FermionField> wl(Nmodes, &grid);
  std::vector<FermionField> vl(Nmodes, &grid);

  loadBinnedA2AVecs<173>(wl, "/global/cfs/cdirs/mp13/lundstrum/Masaaki_data/forErik/vw/vl_grid.173.5070", 5070, &grid);
  loadBinnedA2AVecs<173>(vl, "/global/cfs/cdirs/mp13/lundstrum/Masaaki_data/forErik/vw/vl_grid.173.5070", 5070, &grid);


  // epilogue
  std::cout << GridLogMessage << "Grid is finalizing now" << std::endl;
  Grid_finalize();

  return EXIT_SUCCESS;
}

