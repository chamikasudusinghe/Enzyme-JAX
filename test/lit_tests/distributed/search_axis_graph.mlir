// RUN: enzymexlamlir-opt --mlir-print-op-on-diagnostic=false --sdy-propagation-pipeline --shardy-to-distributed-pipeline --cse --canonicalize --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh2 device_target \"cpu\" axes [!distributed.physical_comm_axis<2, 1>]"' --distributed-search-strategies="beam-size=2 axis-graph-out=%t.graph.csv" %s -o /dev/null
// RUN: FileCheck --input-file=%t.graph.csv %s

// The two matmuls share no tensor, so no logical axis of one can land on the
// same kernel or collective as one of the other. The overlap graph must come
// out as two disjoint triangles, {0,3,4} and {1,2,5} -- one per matmul, each
// over its lhs, rhs and contracted axes.
//
// A cross-component edge here would mean the relation reports coupling the
// program does not have, and every decomposition built on it would be
// unsound. Pinning the whole file rather than a few lines is deliberate: the
// absence of an edge is the property under test, and only an exact match
// shows an absence.

// CHECK:      kind,a,b
// CHECK-NEXT: axis,0,8
// CHECK-NEXT: axis,1,8
// CHECK-NEXT: axis,2,8
// CHECK-NEXT: axis,3,8
// CHECK-NEXT: axis,4,16
// CHECK-NEXT: axis,5,16
// CHECK-NEXT: edge,0,3
// CHECK-NEXT: edge,0,4
// CHECK-NEXT: edge,1,2
// CHECK-NEXT: edge,1,5
// CHECK-NEXT: edge,2,5
// CHECK-NEXT: edge,3,4
// CHECK-NOT:  edge

module @two_blocks {
  sdy.mesh @mesh = <["x"=2]>
  func.func @main(
      %arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>},
      %arg1: tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>},
      %arg2: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>},
      %arg3: tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>})
      -> (tensor<8x8xf32>, tensor<8x8xf32>) {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [] x [], contracting_dims = [1] x [0] :
      (tensor<8x16xf32>, tensor<16x8xf32>) -> tensor<8x8xf32>
    %1 = stablehlo.dot_general %arg2, %arg3, batching_dims = [] x [], contracting_dims = [1] x [0] :
      (tensor<8x16xf32>, tensor<16x8xf32>) -> tensor<8x8xf32>
    return %0, %1 : tensor<8x8xf32>, tensor<8x8xf32>
  }
}
