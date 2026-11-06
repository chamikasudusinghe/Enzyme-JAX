#ifndef ENZYME_AD_JAX_PASSES_DISTRIBUTED_SEARCHMETRICS_H
#define ENZYME_AD_JAX_PASSES_DISTRIBUTED_SEARCHMETRICS_H

#include "src/enzyme_ad/jax/Passes/Distributed/BeamSearchDriver.h"
#include "src/enzyme_ad/jax/Passes/Distributed/SearchCost.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace mlir::enzyme::distributed {

// Per-candidate search data, written as CSV when the search completes. Never
// changes what the search does. Records hold a shared_ptr to their node: the
// queue drops candidates that miss the beam and the allocator reuses the
// address.
template <typename Node> class SearchMetrics : public BeamSearchObserver<Node> {
public:
  using NodePtr = std::shared_ptr<Node>;

  explicit SearchMetrics(llvm::StringRef path) : path(path) {}

  // Called by the scorer, the only place a cost breakdown exists.
  // `assignment` describes the completed candidate that was actually lowered
  // and priced, which is not `node` itself: the scorer completes a scratch
  // clone first. Empty when the caller does not supply one.
  void recordCost(const NodePtr &node, double score, const SearchCost &cost,
                  llvm::StringRef assignment = {}) {
    evaluations++;
    Record &r = recordFor(node);
    if (r.firstEval == std::numeric_limits<size_t>::max())
      r.firstEval = evaluations - 1;
    r.evalCount++;
    r.score = score;
    r.cost = cost;
    r.hasCost = true;
    if (!assignment.empty())
      r.assignment = assignment.str();
  }

  void recordSurrogate(const NodePtr &node, double predicted) {
    Record &r = recordFor(node);
    r.surrogate = predicted;
    r.hasSurrogate = true;
  }

  // `index` is the child's position in its parent's generated list. A rollout
  // regenerates a node's children on every visit, so a running counter would
  // number the same option differently each time and the root-to-leaf sequence
  // would stop being a decision vector.
  void noteGenerated(const NodePtr &child, const NodePtr &parent,
                     long long index) {
    // The parent first: recordFor can reallocate `records`, so the child's
    // reference has to be taken after any other record is created. Interning
    // it here is also what gives the root a row under the rollout strategies,
    // which never form a generation.
    long long parentId =
        parent == child ? -1 : static_cast<long long>(recordFor(parent).id);
    Record &r = recordFor(child);
    if (r.childIndex < 0)
      r.childIndex = index;
    r.depth = child->getAxisIndex();
    if (r.parentId < 0)
      r.parentId = parentId;
  }

  void onScored(const NodePtr &child, const NodePtr &parent) override {
    Record &r = recordFor(child);
    r.depth = child->getAxisIndex();
    r.finalized = child->finalized();
    r.score = child->score;
    // Rollouts report a scored leaf with parent == child. Setting that as
    // lineage makes a node its own parent.
    if (r.parentId < 0 && parent != child) {
      auto it = ids.find(parent.get());
      if (it != ids.end())
        r.parentId = static_cast<long long>(it->second);
    }
    if (r.childIndex < 0 && parent != child)
      r.childIndex = childCounter[parent.get()]++;
  }

  void onFinalized(const NodePtr &node) override {
    Record &r = recordFor(node);
    r.finalized = true;
    r.depth = node->getAxisIndex();
  }

  void onSearchComplete(const NodePtr &best) override {
    if (best)
      recordFor(best).isBest = true;
    write();
  }

  // From the beam queue's per-generation callback. A generation arrives in
  // ascending score order, so the rank from the best end is size - index.
  void onGenerationFormed(llvm::ArrayRef<NodePtr> generation) {
    for (size_t i = 0; i < generation.size(); ++i) {
      Record &r = recordFor(generation[i]);
      size_t rank = generation.size() - i;
      r.bestRank = std::min(r.bestRank, rank);
      r.worstRank = std::max(r.worstRank, rank);
      r.generations++;
    }
  }

private:
  struct Record {
    size_t id = 0;
    long long parentId = -1;
    size_t depth = 0;
    bool finalized = false;
    bool isBest = false;
    double score = 0;
    SearchCost cost;
    bool hasCost = false;
    // Rank within a generation, 1 = best.
    size_t bestRank = std::numeric_limits<size_t>::max();
    size_t worstRank = 0;
    size_t generations = 0;
    // The explorer enumerates deterministically, so the root-to-leaf sequence
    // of these is the decision vector.
    long long childIndex = -1;
    double surrogate = 0;
    bool hasSurrogate = false;
    // Canonical decisions of the priced candidate: per logical axis, which
    // factors it took. Unlike the root-to-leaf childIndex sequence this does
    // not depend on the path taken to reach the assignment, which is what an
    // analysis over axes rather than over decision steps needs.
    std::string assignment;
    // Scorer-call index of this node's first pricing, and how many times it was
    // priced. Rows are written in generation order, so without these the order
    // the search actually evaluated candidates in is not recoverable -- and an
    // incumbent-relative question ("was this candidate already beaten when it
    // was priced?") can only be asked in evaluation order. evalCount is
    // separate because MCTS re-scores a node on every visit through it.
    size_t firstEval = std::numeric_limits<size_t>::max();
    size_t evalCount = 0;
  };

  std::string path;
  // Every scorer call, including repeats on the same node.
  size_t evaluations = 0;
  llvm::DenseMap<const Node *, size_t> ids;
  llvm::DenseMap<const Node *, long long> childCounter;
  std::vector<Record> records;
  // Keeps every node alive so its address stays a unique key.
  std::vector<NodePtr> pinned;

  Record &recordFor(const NodePtr &node) {
    auto it = ids.find(node.get());
    if (it != ids.end())
      return records[it->second];
    size_t id = records.size();
    ids[node.get()] = id;
    pinned.push_back(node);
    records.emplace_back();
    records.back().id = id;
    return records.back();
  }

  void write() {
    std::error_code ec;
    llvm::raw_fd_ostream os(path, ec, llvm::sys::fs::OF_Text);
    if (ec) {
      llvm::errs() << "distributed-search-strategies: cannot write metrics to '"
                   << path << "': " << ec.message() << "\n";
      return;
    }
    os << "id,parent_id,child_index,depth,finalized,is_best,score,collective,"
          "compute,total,num_collectives,num_kernels,plan_failed,best_rank,"
          "worst_rank,generations,surrogate,collective_bytes,"
          "collective_bytes_x_participants,assignment,first_eval,"
          "eval_count\n";
    for (const Record &r : records) {
      os << r.id << ',' << r.parentId << ',' << r.childIndex << ',' << r.depth
         << ',' << (r.finalized ? 1 : 0) << ',' << (r.isBest ? 1 : 0) << ',';
      // -inf is written as is: clamping it would hide "infeasible". %.17g so
      // the components still sum to the total.
      os << llvm::format("%.17g", r.score) << ',';
      if (r.hasCost)
        os << llvm::format("%.17g", r.cost.collective.total) << ','
           << llvm::format("%.17g", r.cost.compute) << ','
           << llvm::format("%.17g", r.cost.total()) << ','
           << r.cost.collective.numCollectives << ',' << r.cost.numKernels
           << ',' << (r.cost.feasible() ? 0 : 1) << ',';
      else
        os << ",,,,,,";
      if (r.bestRank != std::numeric_limits<size_t>::max())
        os << r.bestRank;
      os << ',' << r.worstRank << ',' << r.generations << ',';
      if (r.hasSurrogate)
        os << llvm::format("%.17g", r.surrogate);
      os << ',';
      if (r.hasCost)
        os << r.cost.collectiveBytes << ','
           << r.cost.collectiveBytesXParticipants;
      else
        os << ',';
      os << ',' << r.assignment << ',';
      if (r.evalCount)
        os << r.firstEval;
      os << ',' << r.evalCount << '\n';
    }
    // Not a data row. MCTS re-scores nodes, so rows understate evaluations.
    os << "# evaluations=" << evaluations
       << " nodes_registered=" << records.size() << "\n";
  }
};

} // namespace mlir::enzyme::distributed

#endif // ENZYME_AD_JAX_PASSES_DISTRIBUTED_SEARCHMETRICS_H
