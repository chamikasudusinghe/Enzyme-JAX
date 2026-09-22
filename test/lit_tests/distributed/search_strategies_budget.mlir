// RUN: enzymexlamlir-opt --mlir-print-op-on-diagnostic=false --sdy-propagation-pipeline --shardy-to-distributed-pipeline --cse --canonicalize --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh2 device_target \"cpu\" axes [!distributed.physical_comm_axis<2, 1>]"' --distributed-search-strategies="search-strategy=random search-samples=20 search-seed=1 metrics-out=%t.rnd.csv" --distributed-search-lowering-pipeline %s -o /dev/null
// RUN: FileCheck --input-file=%t.rnd.csv --check-prefixes=HEADER,BUDGET %s
// RUN: enzymexlamlir-opt --mlir-print-op-on-diagnostic=false --sdy-propagation-pipeline --shardy-to-distributed-pipeline --cse --canonicalize --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh2 device_target \"cpu\" axes [!distributed.physical_comm_axis<2, 1>]"' --distributed-search-strategies="search-strategy=mcts search-samples=20 search-seed=1 metrics-out=%t.mcts.csv" --distributed-search-lowering-pipeline %s -o /dev/null
// RUN: FileCheck --input-file=%t.mcts.csv --check-prefix=BUDGET %s
// RUN: enzymexlamlir-opt --mlir-print-op-on-diagnostic=false --sdy-propagation-pipeline --shardy-to-distributed-pipeline --cse --canonicalize --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh2 device_target \"cpu\" axes [!distributed.physical_comm_axis<2, 1>]"' --distributed-search-strategies="search-strategy=twostage search-samples=20 surrogate-train=5 search-seed=1 metrics-out=%t.ts.csv" --distributed-search-lowering-pipeline %s -o /dev/null
// RUN: FileCheck --input-file=%t.ts.csv --check-prefix=BUDGET %s
// RUN: enzymexlamlir-opt --mlir-print-op-on-diagnostic=false --sdy-propagation-pipeline --shardy-to-distributed-pipeline --cse --canonicalize --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh2 device_target \"cpu\" axes [!distributed.physical_comm_axis<2, 1>]"' --distributed-search-strategies="search-strategy=twostage search-samples=20 surrogate-train=0 search-seed=1 metrics-out=%t.fb.csv" %s -o /dev/null 2>&1 | FileCheck --check-prefix=FALLBACK %s
// RUN: FileCheck --input-file=%t.fb.csv --check-prefix=BUDGET %s
// RUN: not enzymexlamlir-opt --mlir-print-op-on-diagnostic=false --sdy-propagation-pipeline --shardy-to-distributed-pipeline --cse --canonicalize --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh2 device_target \"cpu\" axes [!distributed.physical_comm_axis<2, 1>]"' --distributed-search-strategies="search-strategy=bogus" %s -o /dev/null 2>&1 | FileCheck --check-prefix=UNKNOWN %s
// RUN: enzymexlamlir-opt --mlir-print-op-on-diagnostic=false --sdy-propagation-pipeline --shardy-to-distributed-pipeline --cse --canonicalize --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh2 device_target \"cpu\" axes [!distributed.physical_comm_axis<2, 1>]"' --distributed-search-strategies="search-strategy=twostage search-samples=20 surrogate-train=5 surrogate-pool=5 search-seed=1 metrics-out=%t.pool.csv" %s -o /dev/null 2>&1 | FileCheck --check-prefix=POOL %s
// RUN: FileCheck --input-file=%t.pool.csv --check-prefix=BUDGET %s
// RUN: enzymexlamlir-opt --mlir-print-op-on-diagnostic=false --sdy-propagation-pipeline --shardy-to-distributed-pipeline --cse --canonicalize --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh2 device_target \"cpu\" axes [!distributed.physical_comm_axis<2, 1>]"' --distributed-search-strategies="search-strategy=mcts search-samples=20 uct-backup=max search-seed=1 metrics-out=%t.max.csv" %s -o /dev/null
// RUN: FileCheck --input-file=%t.max.csv --check-prefix=BUDGET %s
// RUN: enzymexlamlir-opt --mlir-print-op-on-diagnostic=false --sdy-propagation-pipeline --shardy-to-distributed-pipeline --cse --canonicalize --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh2 device_target \"cpu\" axes [!distributed.physical_comm_axis<2, 1>]"' --distributed-search-strategies="search-strategy=mcts search-samples=20 surrogate-prior=1.0 surrogate-train=5 search-seed=1 metrics-out=%t.prior.csv" %s -o /dev/null
// RUN: FileCheck --input-file=%t.prior.csv --check-prefix=BUDGET %s
// RUN: enzymexlamlir-opt --mlir-print-op-on-diagnostic=false --sdy-propagation-pipeline --shardy-to-distributed-pipeline --cse --canonicalize --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh2 device_target \"cpu\" axes [!distributed.physical_comm_axis<2, 1>]"' --distributed-search-strategies="search-strategy=beam beam-size=2 metrics-out=%t.beam.csv" %s -o /dev/null
// RUN: FileCheck --input-file=%t.beam.csv --check-prefixes=HEADER,RANKS %s
// RUN: not enzymexlamlir-opt --mlir-print-op-on-diagnostic=false --sdy-propagation-pipeline --shardy-to-distributed-pipeline --cse --canonicalize --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh2 device_target \"cpu\" axes [!distributed.physical_comm_axis<2, 1>]"' --distributed-search-strategies="search-strategy=mcts uct-backup=bogus" %s -o /dev/null 2>&1 | FileCheck --check-prefix=BACKUP %s
// RUN: not enzymexlamlir-opt --mlir-print-op-on-diagnostic=false --sdy-propagation-pipeline --shardy-to-distributed-pipeline --cse --canonicalize --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh2 device_target \"cpu\" axes [!distributed.physical_comm_axis<2, 1>]"' --distributed-search-strategies="search-strategy=twostage surrogate-target=bogus" %s -o /dev/null 2>&1 | FileCheck --check-prefix=TARGET %s

// Every strategy spends exactly its budget, counted in scorer calls: MCTS
// re-scores nodes, so its row count can be lower. Each run is followed by the
// lowering pipeline, which only succeeds once the winner has been applied.

// HEADER: id,parent_id,child_index,depth,finalized,is_best,score,collective,compute,total,num_collectives,num_kernels,plan_failed,
// BUDGET: # evaluations=20

// With no training samples the surrogate cannot be fitted, and two-stage
// falls back to random rollouts for the whole budget.
// FALLBACK: surrogate fit failed on 0 samples; remaining 20 evaluations fall back to random rollouts

// UNKNOWN: unknown search-strategy 'bogus' (known: beam, random, mcts, twostage)

// A pool smaller than the candidates to evaluate would leave budget unspent,
// so it is raised and the run still spends all 20.
// POOL: surrogate-pool 5 is below the 15 candidates to evaluate; raising it to 15

// Both are compared against one spelling, so an unknown value has to fail
// rather than silently select the default.
// BACKUP: unknown uct-backup 'bogus' (known: mean, max)
// TARGET: unknown surrogate-target 'bogus' (known: log, linear)

// Generation ranks come from the beam queue, so only a beam run fills the
// last three columns. The root, then the winner: finalized, flagged is_best,
// and ranked within its generation.
// RANKS: 0,-1,-1,0,0,0,0,,,,,,,1,1,1,,,
// RANKS: 8,6,0,3,1,1,0,0,0,0,0,1,0,2,2,1,,0,0

module @tiny {
  sdy.mesh @mesh = <["x"=2]>
  func.func @main(
      %arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>},
      %arg1: tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>})
      -> (tensor<8x8xf32>) {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [] x [], contracting_dims = [1] x [0] :
      (tensor<8x16xf32>, tensor<16x8xf32>) -> tensor<8x8xf32>
    return %0 : tensor<8x8xf32>
  }
}
