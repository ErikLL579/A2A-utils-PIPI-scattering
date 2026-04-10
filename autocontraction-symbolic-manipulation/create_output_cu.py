"""
Generate output.cu from optimized contraction files.

Reads I2_pipi_EM_cexpr_mom_optimized.txt and the original cexpr file,
then generates C++/CUDA code for Grid-based lattice QCD contractions.

Usage:
    python3 create_output_cu.py
"""

import re
from string import Template
from parse_contractions import (
    parse_phase1, parse_phase2, parse_phase3,
    phase2_to_buffer_indices,
    make_momentum_map, make_time_map,
    momentum_time_index_to_flattened_index,
    level1_to_contractions, level2_to_contractions,
    parse_expr_factors,
    choose_appropriate_matrix, prod_pi_to_matrix,
)


# =========================================================
# Configuration
# =========================================================

OPTIMIZED_FILE = 'I2_pipi_EM_cexpr_mom_optimized.txt'
ORIGINAL_FILE = 'luchang-qlat-AC-output/I2_pipi_EM_cexpr_original.txt'
OUTPUT_FILE = 'output.cu'

Nmodes = 100
tsrc = 0
min_photon_energy = 1.0

# Buffer index parameters
momenta_set = [0, 1, 2, 3]
time_set = [0, 1, 2, 3]
Nt = 1


# =========================================================
# Parse all inputs
# =========================================================

level1, level2 = parse_phase1(OPTIMIZED_FILE)
phase2 = parse_phase2(OPTIMIZED_FILE)
gamma_ket_indices, bra_indices = phase2_to_buffer_indices(OPTIMIZED_FILE)
terms, duplicates = parse_phase3(OPTIMIZED_FILE)
factors = parse_expr_factors(ORIGINAL_FILE, expr_index=4)


# =========================================================
# Index resolution
# =========================================================

# Override for the 4-pion-field buffer layout used in the notebook
def momentum_time_index_to_flattened_index(p, t, Nmodes, Nt):
    return t

def make_time_map(tsrc, tsnk, Delta):
    return {
        't_src': 0,
        't_snk': 2,
        't_src + Delta': 1,
        't_snk + Delta': 3,
    }

level1_contractions = level1_to_contractions(momenta_set, time_set, level1, Nmodes, Nt)
level2_contractions = level2_to_contractions(
    momenta_set, time_set, level1, level2, terms, Nmodes, Nt, level1_contractions[2]
)


# =========================================================
# C++ Templates
# =========================================================

template_Grid_preamble = Template("""
#include <Grid/Grid.h>
#include <Grid/qcd/utils/A2Autils.h>
#include "A2A_Mat_Vec_util.h"

using namespace Grid;
using namespace std;

const int TSRC = $temp_tsrc;  //timeslice where rho is nonzero
const int VDIM = $temp_Nmodes; //length of each vector

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
""")

template_Grid_epilogue = Template("""
// epilogue
  std::cout << GridLogMessage << "Grid is finalizing now" << std::endl;
  Grid_finalize();

  return EXIT_SUCCESS;
}
""")

template_pion_matrices_preamble = Template("""
// ====================================================== //
// == DEFINE AND CONSTRUCT PION SOURCE & PROD MATRICES == //
// ====================================================== //
""")

template_pion_matrices_epilogue = Template("""
// ====================================================== //
// ========= COMPLETE MATRIX-MATRIX OPERATIONS ========== //
// ====================================================== //
""")

template_pion_sorce_matrices = Template("""
  int times = 4;
  Eigen::Tensor<ComplexD, 3, Eigen::RowMajor> Pion_source_fields(times, VDIM, VDIM);
""")

template_pion_product_matrices = Template("""
  int num_products_level_$N = $num_prods;
  Eigen::Tensor<ComplexD, 3, Eigen::RowMajor> Pion_product_fields_level_$N(num_products, VDIM, VDIM);
""")

template_perform_matrix_multiplies = """
  PipiA2Autils<WilsonImplR>::MesonField_MesonField_connected(Pion_source_fields,
                                                             Pion_product_fields_level_1,
                                                             Pion_product_fields_level_2,
                                                             num_products_level_1,
                                                             buffer_flag_A,
                                                             buffer_flag_B,
                                                             A_vector_contractions,
                                                             B_vector_contractions,
                                                             C_vector_contractions);
"""

template_generate_photon_props = Template("""
// ======================================= //
// ===== GENERATE PHOTON PROPAGATORs ===== //
// ======================================= //
""")

template_define_photon_field = Template("""
  RealD min = $min_phtn_energy;
  vector<ComplexField> phtn_mu_nu(Nd*Nd, &grid);
""")

template_Coulomb_gauge = Template("""
  IVPhotonPropagator<WilsonImplR>::CoulombGaugeMomentumSpace( &phtn_mu_nu[0], min);
""")

template_prod_vec_preamble = Template("""
// ======================================= //
// == DEFINE ALL MATRIX-VECTOR PRODUCTS == //
// ======================================= //
""")

template_prod_vec_calculate_preamble = Template("""
// ======================================= //
// ===== BEGIN MAT-VECTOR OPERATIONS ===== //
// ======================================= //
""")

template_prod_vec_definition = Template("""
  std::vector<FermionField> $prod_vec(VDIM,&grid);
  for(int i=0; i<VDIM; i++) $prod_vec[i] = Zero();
""")

template_mat_vector = Template("""
  PipiA2Autils<WilsonImplR>::ContractMesonFieldAndVector(&$prod_vec[0], $Pi_Matrix, &v[0]);
""")

template_gamma_matrices = Template("""
  std::vector<Gamma::Algebra> Gmu = {
      Gamma::Algebra::GammaX,
      Gamma::Algebra::GammaT,
      Gamma::Algebra::GammaY,
      Gamma::Algebra::GammaZ
  };
""")

template_FFT_1_complex_field = Template("""
  std::vector<ComplexField> $TERM_FFT1_phi_mu_A(Nd, &grid);
  std::vector<ComplexField> $TERM_FFT1_phi_mu_B(Nd, &grid);
""")

template_FFT_1_prod = Template("""
  PipiA2Autils<WilsonImplR>::FFT_type1_prod(&$TERM_FFT1_phi_mu_A[0], &$ferm_field1[0], &$ferm_field2[0], Gmu);
""")

template_FFT_1_conv = Template("""
  ComplexD $Result_N;
  PipiA2Autils<WilsonImplR>::FFT_type1_convolve($Result_N, &$TERM_FFT1_phi_mu_A[0], &$TERM_FFT1_phi_mu_B[0], &phtn_mu_nu[0]);
""")

template_FFT_2 = Template("""
  ComplexD $Result_N;
  PipiA2Autils<WilsonImplR>::FFT_type2_contract_convolve($Result_N,
                                                         &$ferm_field1[0],
                                                         &$ferm_field2[0],
                                                         &$ferm_field3[0],
                                                         &$ferm_field4[0],
                                                         &phtn_mu_nu[0],
                                                         Gmu);
""")

template_pi_product = Template("""
  ComplexD $pi_prod_trace = Trace($pi_matrix_product_element);
""")

template_result_mult_pi_prod = Template("""
  // pi product factor
  $Result *= $pi_prod_trace;
""")

template_add_quark_charge = Template("""
  // quark charge factor
  $Result *= $quark_charge_factor;
""")


# =========================================================
# Build cu_code
# =========================================================

cu_code = [template_Grid_preamble.substitute(temp_Nmodes=Nmodes, temp_tsrc=tsrc)]


# --- Pion matrices and matrix-matrix contractions ---

template_A_mat_mat_contraction_vectors = "  vector<int> A_vector_contractions = {"
template_B_mat_mat_contraction_vectors = "  vector<int> B_vector_contractions = {"
template_C_mat_mat_contraction_vectors = "  vector<int> C_vector_contractions = {"

for val in level1_contractions[0].values():
    template_A_mat_mat_contraction_vectors += str(val) + ','
for val in level1_contractions[1].values():
    template_B_mat_mat_contraction_vectors += str(val) + ','
for val in level1_contractions[2].values():
    template_C_mat_mat_contraction_vectors += str(val) + ','

# [Aadd, Badd, flag_A, flag_B, Cadd]
for val in level2_contractions[0].values():
    template_A_mat_mat_contraction_vectors += str(val) + ','
for val in level2_contractions[1].values():
    template_B_mat_mat_contraction_vectors += str(val) + ','
for val in level2_contractions[4].values():
    template_C_mat_mat_contraction_vectors += str(val) + ','

template_A_mat_mat_contraction_vectors += '};'
template_B_mat_mat_contraction_vectors += '};'
template_C_mat_mat_contraction_vectors += '};'

# buffer flags
template_buffer_flag_A = '  vector<int> buffer_flag_A = {'
template_buffer_flag_B = '  vector<int> buffer_flag_B = {'
for i in level2_contractions[2]:
    template_buffer_flag_A += str(i) + ','
for i in level2_contractions[3]:
    template_buffer_flag_B += str(i) + ','
template_buffer_flag_A += '};'
template_buffer_flag_B += '};'

cu_code.append(template_pion_matrices_preamble.substitute())
cu_code.append(template_pion_sorce_matrices.substitute())
cu_code.append(template_pion_product_matrices.substitute(num_prods=len(level1), N=1))
cu_code.append(template_pion_product_matrices.substitute(num_prods=len(level2), N=2))

cu_code.append(" ")
cu_code.append("// CONTRACTION VECTORS FOR MATRIX-MATRIX MULTPLIES  //")
cu_code.append(template_A_mat_mat_contraction_vectors)
cu_code.append(template_B_mat_mat_contraction_vectors)
cu_code.append(template_C_mat_mat_contraction_vectors)

cu_code.append(" ")
cu_code.append("// BUFFER FLAGS FOR MATRIX-MATRIX MULTPLIES  //")
cu_code.append(template_buffer_flag_A)
cu_code.append(template_buffer_flag_B)

cu_code.append(" ")
cu_code.append("// PERFORM MATRIX MULTIPLICATIONS //")
cu_code.append(template_perform_matrix_multiplies)
cu_code.append(template_pion_matrices_epilogue.substitute())


# --- Photon propagators ---

cu_code.append(template_generate_photon_props.substitute())
cu_code.append(template_define_photon_field.substitute(min_phtn_energy=min_photon_energy))
cu_code.append(template_Coulomb_gauge.substitute())


# --- Matrix-vector product definitions and computations ---

cu_code.append(template_prod_vec_preamble.substitute())
for name, idx in gamma_ket_indices.items():
    cu_code.append(template_prod_vec_definition.substitute(prod_vec=name))
cu_code.append(template_prod_vec_calculate_preamble.substitute())

for name, idx in gamma_ket_indices.items():
    cu_code.append(template_mat_vector.substitute(
        prod_vec=name, Pi_Matrix=choose_appropriate_matrix(idx, len(level1))
    ))


# --- FFT operations and term assembly ---

cu_code.append(template_gamma_matrices.substitute())

pi_products = {}

for i in terms:
    traces = len(terms[i]['traces'])

    # FFT type 1: two disconnected traces, second is a vec_product (not a prod_Pi)
    if traces == 2 and terms[i]['traces'][1][0]['type'] != 'product':

        cu_code.append(template_FFT_1_complex_field.substitute(
            TERM_FFT1_phi_mu_A=i + '_FFT1_phi_mu_A',
            TERM_FFT1_phi_mu_B=i + '_FFT1_phi_mu_B'
        ))

        ferm_field_1_label = terms[i]['traces'][0][0]['ref'] + '_mu'
        ferm_field_2_label = terms[i]['traces'][1][0]['ref'] + '_mu'

        ferm_field_1A = bra_indices[ferm_field_1_label][0]
        ferm_field_1B = bra_indices[ferm_field_1_label][1]

        cu_code.append(template_FFT_1_prod.substitute(
            TERM_FFT1_phi_mu_A=i + '_FFT1_phi_mu_A',
            ferm_field1=ferm_field_1A,
            ferm_field2=ferm_field_1B
        ))

        ferm_field_2A = bra_indices[ferm_field_2_label][0]
        ferm_field_2B = bra_indices[ferm_field_2_label][1]

        cu_code.append(template_FFT_1_prod.substitute(
            TERM_FFT1_phi_mu_A=i + '_FFT1_phi_mu_B',
            ferm_field1=ferm_field_2A,
            ferm_field2=ferm_field_2B
        ))

        cu_code.append(template_FFT_1_conv.substitute(
            Result_N='Result_' + i,
            TERM_FFT1_phi_mu_A=i + '_FFT1_phi_mu_A',
            TERM_FFT1_phi_mu_B=i + '_FFT1_phi_mu_B'
        ))

        cu_code.append(template_add_quark_charge.substitute(
            Result='Result_' + i,
            quark_charge_factor='(' + factors[i] + ')'
        ))

    # FFT type 2 with extra pi-pi product term e.g. term_Type10_0
    if traces == 2 and terms[i]['traces'][1][0]['type'] == 'product':

        pi_prod = terms[i]['traces'][1][0]['ref']

        if pi_prod not in pi_products:
            pi_products[pi_prod] = pi_prod + '_trace'
            cu_code.append(template_pi_product.substitute(
                pi_prod_trace=pi_prod + '_trace',
                pi_matrix_product_element=prod_pi_to_matrix(pi_prod, len(level1))
            ))

        ferm_field_1_label = terms[i]['traces'][0][0]['ref'] + '_mu'
        ferm_field_2_label = terms[i]['traces'][0][1]['ref'] + '_mu'

        ferm_field_1A = bra_indices[ferm_field_1_label][0]
        ferm_field_1B = bra_indices[ferm_field_1_label][1]
        ferm_field_2A = bra_indices[ferm_field_2_label][0]
        ferm_field_2B = bra_indices[ferm_field_2_label][1]

        cu_code.append(template_FFT_2.substitute(
            Result_N='Result_' + i,
            ferm_field1=ferm_field_1A,
            ferm_field2=ferm_field_1B,
            ferm_field3=ferm_field_2A,
            ferm_field4=ferm_field_2B
        ))

        cu_code.append(template_result_mult_pi_prod.substitute(
            Result='Result_' + i,
            pi_prod_trace=pi_products[pi_prod]
        ))

        cu_code.append(template_add_quark_charge.substitute(
            Result='Result_' + i,
            quark_charge_factor='(' + factors[i] + ')'
        ))

    # FFT type 2 (single connected trace)
    if traces == 1:

        ferm_field_1_label = terms[i]['traces'][0][0]['ref'] + '_mu'
        ferm_field_2_label = terms[i]['traces'][0][1]['ref'] + '_mu'

        ferm_field_1A = bra_indices[ferm_field_1_label][0]
        ferm_field_1B = bra_indices[ferm_field_1_label][1]
        ferm_field_2A = bra_indices[ferm_field_2_label][0]
        ferm_field_2B = bra_indices[ferm_field_2_label][1]

        cu_code.append(template_FFT_2.substitute(
            Result_N='Result_' + i,
            ferm_field1=ferm_field_1A,
            ferm_field2=ferm_field_1B,
            ferm_field3=ferm_field_2A,
            ferm_field4=ferm_field_2B
        ))

        cu_code.append(template_add_quark_charge.substitute(
            Result='Result_' + i,
            quark_charge_factor='(' + factors[i] + ')'
        ))


# --- Save results to .txt file ---

cu_code.append('  // ====================================================== //')
cu_code.append('  // ============= SAVE RESULTS TO TXT FILE =============== //')
cu_code.append('  // ====================================================== //')
cu_code.append('  {')
cu_code.append('    std::ofstream outfile("EM_results.txt");')

for term_name in terms:
    label = term_name.replace('term_', '')
    cu_code.append(f'    outfile << "{label} = " << Result_{term_name} << std::endl;')

cu_code.append('    outfile << "// ============ DUPLICATE DIAGRAMS (x1 <-> x2) ============" << std::endl;')

for dup_name, target_name in duplicates.items():
    label = dup_name.replace('term_', '')
    target_label = target_name.replace('term_', '')
    cu_code.append(f'    outfile << "{label} = " << Result_{target_name} << std::endl;  // x1 <-> x2 of {target_label}')

cu_code.append('    outfile.close();')
cu_code.append('  }')


# --- Epilogue and write ---

cu_code.append(template_Grid_epilogue.substitute())

full_src = "\n".join(cu_code)
with open(OUTPUT_FILE, "w") as f:
    f.write(full_src)

print(f"Written {OUTPUT_FILE} ({len(cu_code)} lines)")
