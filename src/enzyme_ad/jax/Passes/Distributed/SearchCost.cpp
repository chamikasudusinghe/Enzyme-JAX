#include "src/enzyme_ad/jax/Passes/Distributed/SearchCost.h"

#include "mlir/IR/BuiltinTypes.h"
#include "src/enzyme_ad/jax/Dialect/Axis/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Distributed/Dialect.h"
#include "stablehlo/dialect/StablehloOps.h"

#include <algorithm>

namespace mlir::enzyme::distributed {

namespace {

constexpr double kKflopsPerFlop = 1.0 / 1000.0;

// 0 when unrecognized; this runs per op per candidate, so it must not abort.
int64_t bitWidth(Type ty) {
  if (auto intTy = dyn_cast<IntegerType>(ty))
    return intTy.getWidth();
  if (auto floatTy = dyn_cast<FloatType>(ty))
    return floatTy.getWidth();
  if (auto complexTy = dyn_cast<ComplexType>(ty)) {
    int64_t elem = bitWidth(complexTy.getElementType());
    return elem * 2;
  }
  return 0;
}

// Bytes of a statically shaped tensor or memref, else 0.
int64_t sizeBytes(Type ty) {
  auto shaped = dyn_cast<ShapedType>(ty);
  if (!shaped || !shaped.hasStaticShape())
    return 0;
  return shaped.getNumElements() *
         ((bitWidth(shaped.getElementType()) + 7) / 8);
}

int64_t groupExtent(Value v) {
  if (auto group = dyn_cast<axis::FactorGroupType>(v.getType()))
    return std::max<int64_t>(1, static_cast<int64_t>(group.getExtent()));
  return 1;
}

// 2 * (output elements) * (contracted extent), in KFLOP.
double dotKflops(ShapedType lhs, ShapedType out,
                 ArrayRef<int64_t> contracting) {
  if (!lhs || !out || !lhs.hasStaticShape() || !out.hasStaticShape())
    return 0;
  int64_t reduction = 1;
  for (int64_t d : contracting) {
    if (d < 0 || d >= lhs.getRank())
      return 0;
    reduction *= lhs.getDimSize(d);
  }
  return 2.0 * out.getNumElements() * reduction * kKflopsPerFlop;
}

double dotKflops(Operation *op) {
  if (auto dot = dyn_cast<stablehlo::DotOp>(op)) {
    auto lhs = dyn_cast<ShapedType>(dot.getLhs().getType());
    if (!lhs || lhs.getRank() == 0)
      return 0;
    return dotKflops(lhs, dyn_cast<ShapedType>(dot.getType()),
                     {lhs.getRank() - 1});
  }
  if (auto dot = dyn_cast<stablehlo::DotGeneralOp>(op))
    return dotKflops(
        dyn_cast<ShapedType>(dot.getLhs().getType()),
        dyn_cast<ShapedType>(dot.getType()),
        dot.getDotDimensionNumbers().getLhsContractingDimensions());
  return 0;
}

} // namespace

SearchCost SummedCostModel::estimate(ModuleOp lowered) {
  SearchCost cost;
  cost.collective = collective.summarize(lowered);
  cost.computeModelled = computeEnabled();

  lowered.walk([&](Operation *op) {
    if (auto coll = dyn_cast<DistributedCollectiveOp>(op)) {
      int64_t bytes = sizeBytes(coll.getInputObject().getType());
      cost.collectiveBytes += bytes;
      cost.collectiveBytesXParticipants +=
          bytes * groupExtent(coll.getOutputMesh());
      return WalkResult::advance();
    }
    auto kernel = dyn_cast<DistributedKernelOp>(op);
    if (!kernel)
      return WalkResult::advance();
    ++cost.numKernels;
    if (computeEnabled()) {
      kernel->walk([&](Operation *inner) {
        double kflops = dotKflops(inner);
        if (kflops > 0)
          cost.compute += opIntercept + kflopCoeff * kflops;
        else if (inner->getNumRegions() == 0 && inner != kernel)
          cost.compute += opIntercept;
      });
    }
    // Not skip(): this walk is post-order, so the body has already been
    // visited and skipping prunes nothing. Nothing is double counted by that
    // -- the loop above is the only thing that charges compute, and a
    // collective inside a kernel body is counted here exactly as the
    // collective model counts it.
    return WalkResult::advance();
  });
  return cost;
}

} // namespace mlir::enzyme::distributed
