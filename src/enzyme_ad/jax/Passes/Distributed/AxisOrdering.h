#ifndef ENZYME_AD_JAX_PASSES_DISTRIBUTED_AXISORDERING_H
#define ENZYME_AD_JAX_PASSES_DISTRIBUTED_AXISORDERING_H

#include "src/enzyme_ad/jax/Passes/Distributed/LogicalAxisOverlap.h"

#include "llvm/ADT/StringRef.h"

#include <optional>
#include <vector>

namespace mlir::enzyme::distributed {

/**
 * The order the search decides logical axes in.
 *
 * The order is not a tie-break. `setupNextAxis` rebuilds the available physical
 * space from the whole mesh and subtracts the bindings of *overlapping* axes
 * only, so an axis decided earlier claims factors an overlapping axis decided
 * later cannot have. Whoever goes first chooses from a larger feasible set, and
 * the axes that never share an operation do not compete at all.
 *
 * Which makes the order a real lever, and leaves the question of what it should
 * be. Three answers, one per principle:
 *
 *  - `Ir`: declaration order, which is what the search did before this existed
 *    and is the control arm. It is not a principle, it is the absence of one.
 *
 *  - `Extent`: largest extent first. The objective's dominant term scales with
 *    payload, so the decision with the largest effect on the cost should be
 *    taken while the feasible set is least constrained. Exp 0016 supports this
 *    independently of any graph argument: the one axis interaction that
 *    dominates the transformer's cost is between its two largest extents, and
 *    one of those axes sits at degree 4 of 14 -- extent predicted the coupling
 *    where graph position did not.
 *
 *  - `Separator`: a minimum vertex separator first, then each block it leaves,
 *    recursively. Conditioning on a separator renders the remaining blocks
 *    conditionally independent, so this is the one order in which no later
 *    decision faces cross-block contention: once the separator is bound, each
 *    block sees a mesh reduced by the separator's claim and by nothing else.
 *    Exp 0016 measured the premise rather than assuming it -- a pair of axes
 *    reaching across a minimum separator buys -0.0011 to -0.0015 held-out R2,
 *    i.e. nothing, on four of five programs.
 *
 * How well `Separator` can work is bounded by the graph. It is near-exact on a
 * pinched chain, whose separator is a single axis splitting it 4+3, and
 * degenerates on a dense one: the transformer's minimum separator is 4 of its
 * 15 axes and attention's is 5 of 8, so there "separator first" approaches
 * "most axes first" and the order stops saying much. That is the same
 * dense-blob limit that defeated the other two mechanisms in exp 0017, and it
 * is a property of the programs, not of this code.
 */
enum class AxisOrder { Ir, Extent, Separator };

// Parses the `axis-order` option. std::nullopt for an unrecognised name, so the
// caller can report it against the option rather than silently picking one.
std::optional<AxisOrder> parseAxisOrder(llvm::StringRef name);

// Permutes `axes` in place. Every ordering is a permutation of the input and
// keeps declaration order as its tie-break, so a run is reproducible from its
// command line and `Ir` leaves the vector untouched.
void reorderAxes(std::vector<TypedValue<LogicalMeshAxisType>> &axes,
                 const LogicalAxisOverlap &overlap, AxisOrder order);

} // namespace mlir::enzyme::distributed

#endif
