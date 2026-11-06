#ifndef ENZYME_AD_JAX_PASSES_DISTRIBUTED_LOGICALAXISOVERLAP_H
#define ENZYME_AD_JAX_PASSES_DISTRIBUTED_LOGICALAXISOVERLAP_H

#include "src/enzyme_ad/jax/Dialect/Distributed/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace mlir::enzyme::distributed {

/**
 * For a module already in the Distributed dialect, builds a
 * map of logical axes that are "overlapping": they are assigned
 * to the same operation or collective, and should not be assigned
 * to the same axis of the physical device mesh.
 *
 * Overlapping is a symmetric but not necessarily transitive relation.
 *
 * The groups the relation is built from are kept as well as projected. Two axes
 * being adjacent does not say which group put them there, and a projection
 * cannot be inverted: a triangle is either one three-axis group or three
 * two-axis ones, and a group of one axis leaves no edge at all. Anything that
 * needs to attribute a cost term to the decisions in its scope therefore needs
 * the groups, not the edges.
 */
class LogicalAxisOverlap {
public:
  // One group of axes assigned to the same operation -- a factor scope of the
  // cost, since the cost is a sum of one term per kernel and one per
  // collective, each reading only the axes on that op.
  struct Scope {
    enum class Kind { Kernel, CollectiveIn, CollectiveOut };
    Kind kind;
    Operation *op;
    llvm::SmallVector<TypedValue<LogicalMeshAxisType>> axes;
  };

  static llvm::StringRef kindName(Scope::Kind kind) {
    switch (kind) {
    case Scope::Kind::Kernel:
      return "kernel";
    case Scope::Kind::CollectiveIn:
      return "collective_in";
    case Scope::Kind::CollectiveOut:
      return "collective_out";
    }
    return "unknown";
  }

private:
  using LogicalAxisSet =
      llvm::SmallSetVector<TypedValue<LogicalMeshAxisType>, 4>;

  // Map from a logical axis to the set of logical axes it overlaps with
  llvm::DenseMap<TypedValue<LogicalMeshAxisType>, LogicalAxisSet> overlaps;

  void addOverlap(TypedValue<LogicalMeshAxisType> axis1,
                  TypedValue<LogicalMeshAxisType> axis2) {
    if (axis1 == axis2)
      return;
    overlaps[axis1].insert(axis2);
    overlaps[axis2].insert(axis1);
  }

  // Groups in module walk order, which is what makes the ids stable.
  llvm::SmallVector<Scope> scopes;

  // Records the group, then marks every pair within it as mutually
  // overlapping. A group of one axis is still recorded: it contributes a cost
  // term even though it contributes no edge. A group of none is not, since a
  // term that reads no logical axis is constant under every decision.
  void addMutualOverlaps(Scope::Kind kind, Operation *op,
                         llvm::ArrayRef<TypedValue<LogicalMeshAxisType>> axes);

public:
  LogicalAxisOverlap(ModuleOp moduleOp);
  std::optional<const llvm::ArrayRef<TypedValue<LogicalMeshAxisType>>>
  getOverlaps(TypedValue<LogicalMeshAxisType> axis) const;

  llvm::ArrayRef<Scope> getScopes() const { return scopes; }
};

} // namespace mlir::enzyme::distributed
#endif // ENZYME_AD_JAX_PASSES_DISTRIBUTED_LOGICALAXISOVERLAP_H