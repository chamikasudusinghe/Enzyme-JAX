// RUN: enzymexlamlir-opt --mlir-print-op-on-diagnostic=false --sdy-propagation-pipeline --shardy-to-distributed-pipeline --cse --canonicalize --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh448 device_target \"cpu\" axes [!distributed.physical_comm_axis<4, 32>,!distributed.physical_comm_axis<4, 8>, !distributed.physical_comm_axis<8, 1>]"' --distributed-search-strategies="search-strategy=mcts search-samples=120 search-seed=1 kflop-coeff=0.5 metrics-out=%t.a.csv" %s -o /dev/null
// RUN: enzymexlamlir-opt --mlir-print-op-on-diagnostic=false --sdy-propagation-pipeline --shardy-to-distributed-pipeline --cse --canonicalize --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh448 device_target \"cpu\" axes [!distributed.physical_comm_axis<4, 32>,!distributed.physical_comm_axis<4, 8>, !distributed.physical_comm_axis<8, 1>]"' --distributed-search-strategies="search-strategy=mcts search-samples=120 search-seed=1 kflop-coeff=0.5 metrics-out=%t.b.csv" %s -o /dev/null
// RUN: diff %t.a.csv %t.b.csv
// RUN: FileCheck --input-file=%t.a.csv %s

// MCTS keeps its per-node statistics in a DenseMap, which moves every value
// when it grows, so nothing may hold a reference to one across a lookup. This
// input is large enough for the map to pass that point: a budget of 120 on a
// tree this wide registers far more than the 48 entries that fit in the
// initial table.
//
// What this test does NOT do is catch that defect. It was run against the
// code before the fix and passed: the stale read landed on memory that had not
// been reused, so the search returned the same plan. The proof is the ASAN
// reproducer in experiments/0012-audit-search-side/uaf_repro.cpp. What this
// pins is a tree deep enough to exercise the select path, and determinism at a
// fixed seed.

// CHECK: id,parent_id,child_index,depth,finalized,is_best,score,collective,compute,total,num_collectives,num_kernels,plan_failed,
// Lineage is complete: the root is registered, so its children point at it
// rather than at nothing.
// CHECK: 0,-1,-1,0,
// CHECK: 1,0,0,
// CHECK: # evaluations=120

module @attention {
  sdy.mesh @mesh = <["data"=4, "tile"=4, "model"=2]>
  func.func @main(
      %arg0: tensor<128x512xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"data"}, {"tile"}]>},
      %arg1: tensor<512x512xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"tile"}, {"model"}]>},
      %arg2: tensor<512x512xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"tile"}, {"model"}]>})
      -> (tensor<8x8x16x16xf32>) {
    %q = stablehlo.dot_general %arg0, %arg1, batching_dims = [] x [], contracting_dims = [1] x [0] :
      (tensor<128x512xf32>, tensor<512x512xf32>) -> tensor<128x512xf32>
    %k = stablehlo.dot_general %arg0, %arg2, batching_dims = [] x [], contracting_dims = [1] x [0] :
      (tensor<128x512xf32>, tensor<512x512xf32>) -> tensor<128x512xf32>
    %q_heads = stablehlo.reshape %q : (tensor<128x512xf32>) -> tensor<8x8x16x64xf32>
    %k_heads = stablehlo.reshape %k : (tensor<128x512xf32>) -> tensor<8x8x16x64xf32>
    %k_t = stablehlo.transpose %k_heads, dims = [0, 1, 3, 2] :
      (tensor<8x8x16x64xf32>) -> tensor<8x8x64x16xf32>
    %scores = stablehlo.dot_general %q_heads, %k_t, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2] :
      (tensor<8x8x16x64xf32>, tensor<8x8x64x16xf32>) -> tensor<8x8x16x16xf32>
    return %scores : tensor<8x8x16x16xf32>
  }
}
