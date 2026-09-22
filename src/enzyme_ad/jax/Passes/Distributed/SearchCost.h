#ifndef ENZYME_AD_JAX_PASSES_DISTRIBUTED_SEARCHCOST_H
#define ENZYME_AD_JAX_PASSES_DISTRIBUTED_SEARCHCOST_H

#include "src/enzyme_ad/jax/Dialect/Distributed/CollectiveScore.h"

#include "mlir/IR/BuiltinOps.h"

#include <cstddef>
#include <cstdint>

namespace mlir::enzyme::distributed {

// Cost of one lowered search candidate.
struct SearchCost {
  CollectiveCostSummary collective;
  // Whether `compute` was modelled at all, as opposed to zero.
  bool computeModelled = false;
  double compute = 0;
  size_t numKernels = 0;
  // Collective payload bytes, plain and weighted by participants. Metrics only.
  int64_t collectiveBytes = 0;
  int64_t collectiveBytesXParticipants = 0;

  bool feasible() const { return collective.feasible(); }
  double total() const { return collective.total + compute; }
};

// What the search needs from a cost model.
class SearchCostModel {
public:
  virtual ~SearchCostModel() = default;
  virtual SearchCost estimate(ModuleOp lowered) = 0;
};

// The collective cost model plus a FLOP-only compute stand-in, summed. The
// stand-in is off when both coefficients are zero.
class SummedCostModel : public SearchCostModel {
public:
  SummedCostModel(double kflopCoeff, double opIntercept)
      : kflopCoeff(kflopCoeff), opIntercept(opIntercept) {}

  SearchCost estimate(ModuleOp lowered) override;

  const CollectiveCostModel &collectiveModel() const { return collective; }
  bool computeEnabled() const { return kflopCoeff != 0 || opIntercept != 0; }

private:
  CollectiveCostModel collective;
  double kflopCoeff;
  double opIntercept;
};

} // namespace mlir::enzyme::distributed

#endif // ENZYME_AD_JAX_PASSES_DISTRIBUTED_SEARCHCOST_H
