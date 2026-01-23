/*************************************************************************************
Erik vs Claude GPU Battle
Comparing original ContractMesonFieldAndVector implementation against optimized version

Tests correctness and performance of both implementations.
*************************************************************************************/

#include <Grid/Grid.h>
#include <Grid/qcd/utils/A2Autils.h>
#include "A2A_Mat_Vec_util.h"
#include "Claude-A2A-utils.h"

using namespace Grid;
using namespace std;

typedef typename DomainWallFermionD::ComplexField ComplexField;
typedef typename DomainWallFermionD::FermionField FermionField;

// Tolerance for floating point comparison
const double TOLERANCE = 1e-10;

int main(int argc, char *argv[])
{
  // initialization
  Grid_init(&argc, &argv);
  std::cout << GridLogMessage << "=== ERIK VS CLAUDE GPU BATTLE ===" << std::endl;
  std::cout << GridLogMessage << "Grid initialized" << std::endl;

  // Lattice and rng setup
  Coordinate latt_size   = GridDefaultLatt();
  Coordinate simd_layout = GridDefaultSimd(4, vComplex::Nsimd());
  Coordinate mpi_layout  = GridDefaultMpi();
  GridCartesian grid(latt_size, simd_layout, mpi_layout);

  std::cout << GridLogMessage << "Lattice size: "
            << latt_size[0] << "x"
            << latt_size[1] << "x"
            << latt_size[2] << "x"
            << latt_size[3] << std::endl;

  std::vector<int> seeds({1, 2, 3, 4});
  GridParallelRNG pRNG(&grid);
  pRNG.SeedFixedIntegers(seeds);

  // Test with different VDIM values to see scaling
  std::vector<int> vdim_tests = {32, 64, 128};

  for(int VDIM : vdim_tests) {
    std::cout << GridLogMessage << "\n========================================" << std::endl;
    std::cout << GridLogMessage << "Testing with VDIM = " << VDIM << std::endl;
    std::cout << GridLogMessage << "========================================" << std::endl;

    // Create input fermion fields
    std::vector<FermionField> phi(VDIM, &grid);
    std::cout << GridLogMessage << "Initialising " << VDIM << " random fermion fields..." << std::endl;
    for(int i = 0; i < VDIM; ++i) {
      random(pRNG, phi[i]);
    }

    // Create a non-trivial test matrix (not just identity)
    // Use random complex values to better test correctness
    Eigen::Tensor<ComplexD, 2, Eigen::RowMajor> M_test(VDIM, VDIM);
    std::cout << GridLogMessage << "Creating random test matrix..." << std::endl;
    srand(12345);  // Fixed seed for reproducibility
    for(int i = 0; i < VDIM; i++) {
      for(int j = 0; j < VDIM; j++) {
        double re = (double)rand() / RAND_MAX - 0.5;
        double im = (double)rand() / RAND_MAX - 0.5;
        M_test(i, j) = ComplexD(re, im);
      }
    }

    // Output arrays for both implementations
    std::vector<FermionField> y_erik(VDIM, &grid);
    std::vector<FermionField> y_claude(VDIM, &grid);
    for(int i = 0; i < VDIM; ++i) {
      y_erik[i] = Zero();
      y_claude[i] = Zero();
    }

    double start, stop;
    double erik_time, claude_time;

    // =========================================================================
    // ERIK'S IMPLEMENTATION
    // =========================================================================
    std::cout << GridLogMessage << "\n--- Running Erik's implementation ---" << std::endl;
    start = usecond();
    PipiA2Autils<WilsonImplR>::ContractMesonFieldAndVector(&y_erik[0], M_test, &phi[0]);
    stop = usecond();
    erik_time = stop - start;
    std::cout << GridLogMessage << "Erik's time: " << erik_time << " us ("
              << erik_time / 1e6 << " s)" << std::endl;

    // =========================================================================
    // CLAUDE'S IMPLEMENTATION
    // =========================================================================
    std::cout << GridLogMessage << "\n--- Running Claude's implementation ---" << std::endl;
    start = usecond();
    ClaudeA2Autils<WilsonImplR>::ContractMesonFieldAndVector(&y_claude[0], M_test, &phi[0]);
    stop = usecond();
    claude_time = stop - start;
    std::cout << GridLogMessage << "Claude's time: " << claude_time << " us ("
              << claude_time / 1e6 << " s)" << std::endl;

    // =========================================================================
    // CORRECTNESS CHECK
    // =========================================================================
    std::cout << GridLogMessage << "\n--- Correctness Check ---" << std::endl;
    bool all_correct = true;
    double max_rel_error = 0.0;

    for(int i = 0; i < VDIM; i++) {
      double norm_erik = norm2(y_erik[i]);
      double norm_claude = norm2(y_claude[i]);

      // Compute difference
      FermionField diff(&grid);
      diff = y_erik[i] - y_claude[i];
      double norm_diff = norm2(diff);

      double rel_error = (norm_erik > 0) ? sqrt(norm_diff / norm_erik) : sqrt(norm_diff);
      max_rel_error = std::max(max_rel_error, rel_error);

      if(rel_error > TOLERANCE) {
        std::cout << GridLogMessage << "MISMATCH at i=" << i
                  << ": Erik=" << norm_erik
                  << ", Claude=" << norm_claude
                  << ", rel_error=" << rel_error << std::endl;
        all_correct = false;
      }
    }

    if(all_correct) {
      std::cout << GridLogMessage << "CORRECTNESS: PASSED (max relative error: "
                << max_rel_error << ")" << std::endl;
    } else {
      std::cout << GridLogMessage << "CORRECTNESS: FAILED" << std::endl;
    }

    // =========================================================================
    // PERFORMANCE SUMMARY
    // =========================================================================
    std::cout << GridLogMessage << "\n--- Performance Summary (VDIM=" << VDIM << ") ---" << std::endl;
    std::cout << GridLogMessage << "Erik's implementation:   " << erik_time / 1e6 << " s" << std::endl;
    std::cout << GridLogMessage << "Claude's implementation: " << claude_time / 1e6 << " s" << std::endl;

    if(claude_time > 0) {
      double speedup = erik_time / claude_time;
      std::cout << GridLogMessage << "Speedup (Erik/Claude):   " << speedup << "x" << std::endl;

      if(speedup > 1.0) {
        std::cout << GridLogMessage << "WINNER: Claude" << std::endl;
      } else if(speedup < 1.0) {
        std::cout << GridLogMessage << "WINNER: Erik" << std::endl;
      } else {
        std::cout << GridLogMessage << "WINNER: Tie" << std::endl;
      }
    }

    std::cout << GridLogMessage << "Kernel launches - Erik: ~" << VDIM * VDIM
              << ", Claude: ~" << VDIM << std::endl;
  }

  // Final summary
  std::cout << GridLogMessage << "\n========================================" << std::endl;
  std::cout << GridLogMessage << "GPU BATTLE COMPLETE" << std::endl;
  std::cout << GridLogMessage << "========================================" << std::endl;

  Grid_finalize();
  return EXIT_SUCCESS;
}
