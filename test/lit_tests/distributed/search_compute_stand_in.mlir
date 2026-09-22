// RUN: enzymexlamlir-opt --mlir-print-op-on-diagnostic=false --sdy-propagation-pipeline --shardy-to-distributed-pipeline --cse --canonicalize --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh4 device_target \"cpu\" axes [!distributed.physical_comm_axis<4, 1>]"' --distributed-search-strategies="beam-size=100 dump-candidates dump-best kflop-coeff=0.5" %s -o /dev/null 2>&1 | FileCheck %s
// RUN: enzymexlamlir-opt --mlir-print-op-on-diagnostic=false --sdy-propagation-pipeline --shardy-to-distributed-pipeline --cse --canonicalize --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh4 device_target \"cpu\" axes [!distributed.physical_comm_axis<4, 1>]"' --distributed-search-strategies="beam-size=100 dump-candidates" %s -o /dev/null 2>&1 | FileCheck --check-prefix=OFF %s
// RUN: enzymexlamlir-opt --mlir-print-op-on-diagnostic=false --sdy-propagation-pipeline --shardy-to-distributed-pipeline --cse --canonicalize --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh4 device_target \"cpu\" axes [!distributed.physical_comm_axis<4, 1>]"' --distributed-search-strategies="beam-size=100 dump-candidates dump-best kflop-coeff=1.0" %s -o /dev/null 2>&1 | FileCheck --check-prefix=DOUBLE %s
// RUN: enzymexlamlir-opt --mlir-print-op-on-diagnostic=false --sdy-propagation-pipeline --shardy-to-distributed-pipeline --cse --canonicalize --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh4 device_target \"cpu\" axes [!distributed.physical_comm_axis<4, 1>]"' --distributed-search-strategies="beam-size=100 dump-candidates dump-best kflop-coeff=0.5 op-intercept=1.0" %s -o /dev/null 2>&1 | FileCheck --check-prefix=INTERCEPT %s

// With kflop-coeff=0.5 a kernel costs half its dot KFLOPs, on top of the
// collective cost. The dot is 64x128 by 128x64, 2 * 64 * 64 * 128 flops:
//
//   - fully local: 1048.576 KFLOP, so 524.288
//   - output split 2 or 4 ways: 262.144 or 131.072
//   - contraction split 4 ways: 131.072 plus the 24576.33 all-reduce
//
// Splitting the output 4 ways needs no collective, so it wins.

// CHECK-DAG: // compute cost: total=5.242880e+02 over 1 kernels
// CHECK-DAG: // compute cost: total=2.621440e+02 over 1 kernels
// CHECK-DAG: // compute cost: total=1.310720e+02 over 1 kernels
// CHECK-DAG: // Search candidate (ok, score=-2.470740e+04):
// CHECK-DAG: // Best search candidate (ok, score=-1.310720e+02):

// Each coefficient's effect in isolation, so the test pins the formula rather
// than one setting of it. Doubling the rate doubles the flop term; a per-op
// charge adds exactly one per op, and the winning kernel holds two.
// DOUBLE: // Best search candidate (ok, score=-2.621440e+02):
// INTERCEPT: // Best search candidate (ok, score=-1.330720e+02):

// Off by default: scores are collective cost alone.
// OFF-NOT: compute cost

module {
  sdy.mesh @mesh = <["x"=4]>
  func.func @main(
      %arg0: tensor<64x128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>},
      %arg1: tensor<128x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
      -> (tensor<64x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] :
      (tensor<64x128xf32>, tensor<128x64xf32>) -> tensor<64x64xf32>
    return %0 : tensor<64x64xf32>
  }
}
