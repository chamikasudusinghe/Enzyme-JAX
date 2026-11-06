// RUN: enzymexlamlir-opt --mlir-print-op-on-diagnostic=false --sdy-propagation-pipeline --shardy-to-distributed-pipeline --cse --canonicalize --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh448 device_target \"cpu\" axes [!distributed.physical_comm_axis<4, 32>,!distributed.physical_comm_axis<4, 8>, !distributed.physical_comm_axis<8, 1>]"' --distributed-search-strategies="search-strategy=random search-samples=200 search-seed=1 kflop-coeff=0.5 metrics-out=%t.random.csv" %s -o /dev/null
// RUN: enzymexlamlir-opt --mlir-print-op-on-diagnostic=false --sdy-propagation-pipeline --shardy-to-distributed-pipeline --cse --canonicalize --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh448 device_target \"cpu\" axes [!distributed.physical_comm_axis<4, 32>,!distributed.physical_comm_axis<4, 8>, !distributed.physical_comm_axis<8, 1>]"' --distributed-search-strategies="search-strategy=twostage search-samples=200 surrogate-train=50 surrogate-pool=150 search-seed=1 kflop-coeff=0.5 metrics-out=%t.nopool.csv" %s -o /dev/null
// RUN: awk -F, '$10 != "" { print $7 }' %t.random.csv | sort > %t.random.scores
// RUN: awk -F, '$10 != "" { print $7 }' %t.nopool.csv | sort > %t.nopool.scores
// RUN: diff %t.random.scores %t.nopool.scores
// RUN: enzymexlamlir-opt --mlir-print-op-on-diagnostic=false --sdy-propagation-pipeline --shardy-to-distributed-pipeline --cse --canonicalize --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh448 device_target \"cpu\" axes [!distributed.physical_comm_axis<4, 32>,!distributed.physical_comm_axis<4, 8>, !distributed.physical_comm_axis<8, 1>]"' --distributed-search-strategies="search-strategy=twostage search-samples=200 surrogate-train=50 surrogate-pool=3000 search-seed=1 kflop-coeff=0.5 metrics-out=%t.csv" %s -o /dev/null
// RUN: awk -F, 'NR == 1 { for (i = 1; i <= NF; i++) if ($i == "surrogate") c = i; next } $c != "" { print $c }' %t.csv > %t.surrogate
// RUN: FileCheck --input-file=%t.surrogate --check-prefix=POSITIVE --implicit-check-not='{{^-}}' %s

// With the pool equal to K there is nothing to select from, so two-stage
// scores exactly the rollouts random does. Compare every score, not just the
// best: on this input both find their best among the shared training
// rollouts, so the best alone would not notice a broken pool.

// The log target keeps every surrogate prediction positive: at least one row
// carries a prediction, and no prediction is negative.
//
// The column is located by header name rather than by position. Pinning it as
// "the fourth-last field" broke twice as columns were appended -- once for
// `assignment`, once for `first_eval`/`eval_count` -- and where the column sits
// is not the property under test.
// POSITIVE: {{^[0-9]}}

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
