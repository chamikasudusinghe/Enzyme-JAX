#include "src/enzyme_ad/jax/Passes/Distributed/AxisOrdering.h"

#include "src/enzyme_ad/jax/Dialect/Axis/Utilities.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <limits>
#include <queue>

namespace mlir::enzyme::distributed {

std::optional<AxisOrder> parseAxisOrder(llvm::StringRef name) {
  if (name == "ir")
    return AxisOrder::Ir;
  if (name == "extent")
    return AxisOrder::Extent;
  if (name == "separator")
    return AxisOrder::Separator;
  return std::nullopt;
}

namespace {

// Adjacency over positions in the axis vector, which is what every routine
// below works in: the overlap relation is keyed by Value, and resolving it once
// up front keeps the graph code free of MLIR types.
using Graph = std::vector<llvm::SmallVector<unsigned>>;

Graph buildGraph(const std::vector<TypedValue<LogicalMeshAxisType>> &axes,
                 const LogicalAxisOverlap &overlap) {
  llvm::DenseMap<TypedValue<LogicalMeshAxisType>, unsigned> index;
  for (auto [i, axis] : llvm::enumerate(axes))
    index[axis] = i;

  Graph graph(axes.size());
  for (auto [i, axis] : llvm::enumerate(axes)) {
    auto neighbours = overlap.getOverlaps(axis);
    if (!neighbours)
      continue;
    for (TypedValue<LogicalMeshAxisType> neighbour : *neighbours) {
      auto it = index.find(neighbour);
      // Axes this search does not decide are not part of the problem.
      if (it != index.end() && it->second != i)
        graph[i].push_back(it->second);
    }
    llvm::sort(graph[i]);
  }
  return graph;
}

// Max flow by BFS augmentation. The networks here hold two nodes per axis and
// their flow value is bounded by the axis count, so the simplest augmenting
// algorithm is ample -- the graphs this runs on have 6 to 87 axes and it runs
// once per pass, not once per candidate.
struct FlowNetwork {
  struct Edge {
    unsigned to;
    int64_t capacity;
    unsigned reverse;
  };

  std::vector<std::vector<Edge>> edges;

  explicit FlowNetwork(unsigned nodes) : edges(nodes) {}

  void addEdge(unsigned from, unsigned to, int64_t capacity) {
    unsigned forwardIndex = edges[from].size();
    unsigned reverseIndex = edges[to].size();
    // A self-loop would make the two indices collide, and the split-graph
    // construction never needs one.
    assert(from != to && "self-loop in flow network");
    edges[from].push_back({to, capacity, reverseIndex});
    edges[to].push_back({from, 0, forwardIndex});
  }

  // Returns the max flow, leaving the residual capacities in place so the cut
  // can be read off afterwards.
  int64_t run(unsigned source, unsigned sink) {
    int64_t total = 0;
    while (true) {
      // BFS for an augmenting path, recording how each node was reached.
      std::vector<std::pair<unsigned, unsigned>> from(
          edges.size(), {std::numeric_limits<unsigned>::max(), 0});
      std::queue<unsigned> queue;
      queue.push(source);
      from[source] = {source, 0};
      while (!queue.empty() &&
             from[sink].first == std::numeric_limits<unsigned>::max()) {
        unsigned node = queue.front();
        queue.pop();
        for (auto [i, edge] : llvm::enumerate(edges[node])) {
          if (edge.capacity <= 0 ||
              from[edge.to].first != std::numeric_limits<unsigned>::max())
            continue;
          from[edge.to] = {node, static_cast<unsigned>(i)};
          queue.push(edge.to);
        }
      }
      if (from[sink].first == std::numeric_limits<unsigned>::max())
        return total;

      // Push the path's bottleneck.
      int64_t bottleneck = std::numeric_limits<int64_t>::max();
      for (unsigned node = sink; node != source; node = from[node].first) {
        auto [prev, edgeIndex] = from[node];
        bottleneck = std::min(bottleneck, edges[prev][edgeIndex].capacity);
      }
      for (unsigned node = sink; node != source; node = from[node].first) {
        auto [prev, edgeIndex] = from[node];
        Edge &edge = edges[prev][edgeIndex];
        edge.capacity -= bottleneck;
        edges[edge.to][edge.reverse].capacity += bottleneck;
      }
      total += bottleneck;
    }
  }

  // Nodes reachable from the source in the residual network.
  std::vector<bool> reachable(unsigned source) const {
    std::vector<bool> seen(edges.size(), false);
    std::queue<unsigned> queue;
    queue.push(source);
    seen[source] = true;
    while (!queue.empty()) {
      unsigned node = queue.front();
      queue.pop();
      for (const Edge &edge : edges[node])
        if (edge.capacity > 0 && !seen[edge.to]) {
          seen[edge.to] = true;
          queue.push(edge.to);
        }
    }
    return seen;
  }
};

constexpr int64_t kInfinite = std::numeric_limits<int64_t>::max() / 4;

// The minimum set of vertices whose removal disconnects `source` from `sink`,
// which must not be adjacent. Standard reduction: split every other vertex into
// in and out halves joined by a unit edge, so a unit of flow through the vertex
// costs one, and give the graph's own edges infinite capacity. The max flow is
// then the size of the minimum vertex cut, and the cut is read off the residual
// network as the vertices whose in half is reachable from the source while
// their out half is not.
llvm::SmallVector<unsigned> minVertexCut(const Graph &graph,
                                         llvm::ArrayRef<unsigned> members,
                                         unsigned source, unsigned sink) {
  llvm::DenseMap<unsigned, unsigned> local;
  for (auto [i, member] : llvm::enumerate(members))
    local[member] = i;

  unsigned n = members.size();
  FlowNetwork network(2 * n);
  auto in = [](unsigned i) { return 2 * i; };
  auto out = [](unsigned i) { return 2 * i + 1; };

  for (unsigned i = 0; i < n; ++i) {
    // The endpoints are not candidates for removal -- a cut containing one of
    // them would separate the pair trivially and says nothing about the graph.
    bool endpoint = members[i] == source || members[i] == sink;
    network.addEdge(in(i), out(i), endpoint ? kInfinite : 1);
  }
  for (unsigned i = 0; i < n; ++i)
    for (unsigned neighbour : graph[members[i]]) {
      auto it = local.find(neighbour);
      if (it != local.end())
        network.addEdge(out(i), in(it->second), kInfinite);
    }

  network.run(out(local[source]), in(local[sink]));
  std::vector<bool> seen = network.reachable(out(local[source]));

  llvm::SmallVector<unsigned> cut;
  for (unsigned i = 0; i < n; ++i)
    if (seen[in(i)] && !seen[out(i)])
      cut.push_back(members[i]);
  llvm::sort(cut);
  return cut;
}

// Connected components of the subgraph induced on `members`, each sorted, and
// the components themselves ordered by their smallest member so the result is
// deterministic.
std::vector<llvm::SmallVector<unsigned>>
components(const Graph &graph, llvm::ArrayRef<unsigned> members) {
  llvm::DenseMap<unsigned, bool> present;
  for (unsigned member : members)
    present[member] = true;

  llvm::DenseMap<unsigned, bool> seen;
  std::vector<llvm::SmallVector<unsigned>> result;
  for (unsigned start : members) {
    if (seen.count(start))
      continue;
    llvm::SmallVector<unsigned> component;
    llvm::SmallVector<unsigned> stack{start};
    seen[start] = true;
    while (!stack.empty()) {
      unsigned node = stack.pop_back_val();
      component.push_back(node);
      for (unsigned neighbour : graph[node])
        if (present.count(neighbour) && !seen.count(neighbour)) {
          seen[neighbour] = true;
          stack.push_back(neighbour);
        }
    }
    llvm::sort(component);
    result.push_back(std::move(component));
  }
  // `members` arrives sorted, so the outer loop already visits components by
  // their smallest member; sorting again would be a no-op.
  return result;
}

// The graph's own minimum vertex separator, by the Esfahanian-Hakimi reduction:
// it is enough to try one minimum-degree vertex against each of its
// non-neighbours, plus each non-adjacent pair of its neighbours. That is O(n)
// max-flow calls instead of the O(n^2) an all-pairs search would need, which
// matters at 87 axes and is free at 6. Empty when the subgraph is complete, in
// which case no separator exists and nothing can be decomposed.
llvm::SmallVector<unsigned> minimumSeparator(const Graph &graph,
                                             llvm::ArrayRef<unsigned> members) {
  llvm::DenseMap<unsigned, bool> present;
  for (unsigned member : members)
    present[member] = true;

  auto degreeWithin = [&](unsigned node) {
    unsigned degree = 0;
    for (unsigned neighbour : graph[node])
      if (present.count(neighbour))
        ++degree;
    return degree;
  };
  auto adjacent = [&](unsigned a, unsigned b) {
    return llvm::is_contained(graph[a], b);
  };

  unsigned pivot = members.front();
  for (unsigned member : members)
    if (degreeWithin(member) < degreeWithin(pivot))
      pivot = member;

  llvm::SmallVector<unsigned> pivotNeighbours;
  for (unsigned member : members)
    if (member != pivot && adjacent(pivot, member))
      pivotNeighbours.push_back(member);

  llvm::SmallVector<std::pair<unsigned, unsigned>> candidates;
  for (unsigned member : members)
    if (member != pivot && !adjacent(pivot, member))
      candidates.push_back({pivot, member});
  for (unsigned i = 0; i < pivotNeighbours.size(); ++i)
    for (unsigned j = i + 1; j < pivotNeighbours.size(); ++j)
      if (!adjacent(pivotNeighbours[i], pivotNeighbours[j]))
        candidates.push_back({pivotNeighbours[i], pivotNeighbours[j]});

  llvm::SmallVector<unsigned> best;
  for (auto [source, sink] : candidates) {
    llvm::SmallVector<unsigned> cut =
        minVertexCut(graph, members, source, sink);
    if (cut.empty())
      continue;
    if (best.empty() || cut.size() < best.size())
      best = std::move(cut);
  }
  return best;
}

// Nested dissection: the separator, then each block it leaves, recursively.
// Every block is itself a graph, so applying the same rule inside it is the
// same principle rather than an extension of it -- stopping after one level
// would leave the blocks in declaration order, which is the thing being
// replaced.
void appendSeparatorOrder(const Graph &graph, llvm::ArrayRef<unsigned> members,
                          llvm::SmallVectorImpl<unsigned> &out) {
  if (members.size() <= 2) {
    out.append(members.begin(), members.end());
    return;
  }

  std::vector<llvm::SmallVector<unsigned>> blocks = components(graph, members);
  if (blocks.size() > 1) {
    // Already decomposed: axes in different components cannot contend, so
    // there is nothing to separate and each block is handled on its own.
    for (const llvm::SmallVector<unsigned> &block : blocks)
      appendSeparatorOrder(graph, block, out);
    return;
  }

  llvm::SmallVector<unsigned> separator = minimumSeparator(graph, members);
  if (separator.empty() || separator.size() == members.size()) {
    // A complete subgraph, or one whose only separator is everything: no
    // decomposition exists, so declaration order is the honest answer. Using
    // extent order here instead would make this arm a blend of two principles
    // and the comparison between them unreadable.
    out.append(members.begin(), members.end());
    return;
  }

  out.append(separator.begin(), separator.end());

  llvm::DenseMap<unsigned, bool> removed;
  for (unsigned member : separator)
    removed[member] = true;
  llvm::SmallVector<unsigned> rest;
  for (unsigned member : members)
    if (!removed.count(member))
      rest.push_back(member);

  for (const llvm::SmallVector<unsigned> &block : components(graph, rest))
    appendSeparatorOrder(graph, block, out);
}

} // namespace

void reorderAxes(std::vector<TypedValue<LogicalMeshAxisType>> &axes,
                 const LogicalAxisOverlap &overlap, AxisOrder order) {
  if (order == AxisOrder::Ir || axes.size() < 2)
    return;

  if (order == AxisOrder::Extent) {
    // Stable, so equal extents keep declaration order.
    std::stable_sort(axes.begin(), axes.end(),
                     [](TypedValue<LogicalMeshAxisType> a,
                        TypedValue<LogicalMeshAxisType> b) {
                       return axis::getAxisExtent(a) > axis::getAxisExtent(b);
                     });
    return;
  }

  Graph graph = buildGraph(axes, overlap);
  llvm::SmallVector<unsigned> all;
  for (unsigned i = 0; i < axes.size(); ++i)
    all.push_back(i);

  llvm::SmallVector<unsigned> permutation;
  appendSeparatorOrder(graph, all, permutation);
  assert(permutation.size() == axes.size() && "ordering dropped an axis");

  std::vector<TypedValue<LogicalMeshAxisType>> reordered;
  reordered.reserve(axes.size());
  for (unsigned i : permutation)
    reordered.push_back(axes[i]);
  axes = std::move(reordered);
}

} // namespace mlir::enzyme::distributed
