#ifndef CLAUDE_A2A_UTILS_H
#define CLAUDE_A2A_UTILS_H

#include <Grid/Grid_Eigen_Tensor.h>

NAMESPACE_BEGIN(Grid);

///////////////////////////////////////////////////////////////////////////////
// Optimized A2A contraction utilities for pi-pi scattering
//
// Key optimization: Move the j-summation inside the GPU kernel to reduce
// kernel launch overhead from O(Nmodes^2) to O(Nmodes) and keep partial
// sums in registers instead of global memory.
//
// Original reference: A2A_Mat_Vec_util.h
///////////////////////////////////////////////////////////////////////////////

template <typename FImpl>
class ClaudeA2Autils
{
public:
  typedef typename FImpl::ComplexField ComplexField;
  typedef typename FImpl::FermionField FermionField;
  typedef typename FImpl::PropagatorField PropagatorField;

  typedef typename FImpl::SiteSpinor vobj;
  typedef typename vobj::scalar_object sobj;
  typedef typename vobj::scalar_type scalar_type;
  typedef typename vobj::vector_type vector_type;

  typedef iSpinMatrix<vector_type> SpinMatrix_v;
  typedef iSpinMatrix<scalar_type> SpinMatrix_s;
  typedef iSinglet<vector_type> Scalar_v;
  typedef iSinglet<scalar_type> Scalar_s;
  typedef iSpinColourMatrix<vector_type> SpinColourMatrix_v;

  //////////////////////////////////////////////////////////////////////////////
  // ContractMesonFieldAndVector
  //
  // Computes: y^{i}(x) = sum_j PI_{ij} * w_j(x)
  //
  // where PI is an Nmodes x Nmodes meson field matrix and w_j are fermion fields.
  //
  // Optimization strategy:
  // - Copy all w_j device pointers to a device-accessible array
  // - For each output mode i, launch ONE kernel that loops over all j internally
  // - Accumulate partial sums in registers, write to global memory once
  //
  // This reduces:
  // - Kernel launches: Nmodes^2 -> Nmodes
  // - Global memory writes of y: Nmodes per output -> 1 per output
  //////////////////////////////////////////////////////////////////////////////

  template <typename TensorType>
  static void ContractMesonFieldAndVector(FermionField *y_i1,
                                          const TensorType &meson_field_ij,
                                          const FermionField *wj);
};


template <class FImpl>
template <typename TensorType>
void ClaudeA2Autils<FImpl>::ContractMesonFieldAndVector(FermionField *y_i1,
                                                        const TensorType &meson_field_ij,
                                                        const FermionField *wj)
{
  typedef typename FImpl::SiteSpinor vobj;

  const int Nmodes = meson_field_ij.dimension(0);
  assert(Nmodes == meson_field_ij.dimension(1) &&
         "Meson field must be square (Nmodes x Nmodes)");

  GridBase *grid = wj[0].Grid();
  const int Nsimd = grid->Nsimd();
  const uint64_t Nsite = grid->oSites();

  // -------------------------------------------------------------------------
  // Step 1: Copy meson field matrix to device (once)
  // -------------------------------------------------------------------------
  deviceVector<ComplexD> PI_dev(Nmodes * Nmodes);
  acceleratorCopyToDevice(meson_field_ij.data(),
                          &PI_dev[0],
                          Nmodes * Nmodes * sizeof(ComplexD));
  ComplexD *PI_ptr = &PI_dev[0];

  // -------------------------------------------------------------------------
  // Step 2: Gather device pointers for all wj fields
  //
  // We create views for all wj and extract their device pointers into an
  // array that can be accessed from the GPU kernel.
  // -------------------------------------------------------------------------

  // Host-side storage for views (keeps them alive during computation)
  std::vector<typename FermionField::View_type> w_views;
  w_views.reserve(Nmodes);

  // Host-side array of device pointers
  std::vector<vobj*> w_ptrs_host(Nmodes);

  for(int j = 0; j < Nmodes; j++) {
    w_views.push_back(wj[j].View(AcceleratorRead));
    w_ptrs_host[j] = &w_views[j][0];
  }

  // Copy pointer array to device
  deviceVector<vobj*> w_ptrs_dev(Nmodes);
  acceleratorCopyToDevice(w_ptrs_host.data(),
                          &w_ptrs_dev[0],
                          Nmodes * sizeof(vobj*));
  vobj **wj_ptrs = &w_ptrs_dev[0];

  // -------------------------------------------------------------------------
  // Step 3: Main computation loop - O(Nmodes) kernel launches
  //
  // For each output mode i, we launch a single kernel that:
  // - Initializes y_local to zero in registers
  // - Loops over all j, accumulating PI_ij * w_j into registers
  // - Writes final result to global memory once
  // -------------------------------------------------------------------------

  for(int i = 0; i < Nmodes; i++) {
    autoView(y_i, y_i1[i], AcceleratorWrite);

    // Pointer to row i of the meson field matrix
    ComplexD *PI_row_i = PI_ptr + i * Nmodes;

    accelerator_for(ss, Nsite, Nsimd, {
      // Accumulator in registers - no global memory traffic during j-loop
      decltype(coalescedRead(y_i[ss])) y_local;
      zeroit(y_local);

      // Inner loop over j - all accumulation happens in registers
      for(int j = 0; j < Nmodes; j++) {
        auto w_local = coalescedRead(wj_ptrs[j][ss]);
        y_local = y_local + PI_row_i[j] * w_local;
      }

      // Single write to global memory per site
      coalescedWrite(y_i[ss], y_local);
    });
  }

  // Views are automatically released when w_views goes out of scope

  std::cout << GridLogMessage
            << "ContractMesonFieldAndVector completed (optimized): "
            << "Nmodes=" << Nmodes << ", Nsite=" << Nsite
            << std::endl;
}

NAMESPACE_END(Grid);

#endif // CLAUDE_A2A_UTILS_H
