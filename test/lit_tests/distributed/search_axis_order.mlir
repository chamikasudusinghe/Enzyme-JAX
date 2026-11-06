// RUN: enzymexlamlir-opt --mlir-print-op-on-diagnostic=false --sdy-propagation-pipeline --shardy-to-distributed-pipeline --cse --canonicalize --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh2 device_target \"cpu\" axes [!distributed.physical_comm_axis<2, 1>]"' --distributed-search-strategies="beam-size=2 axis-order=ir axis-graph-out=%t.ir.csv" %s -o /dev/null
// RUN: FileCheck --input-file=%t.ir.csv --check-prefix=IR %s

// RUN: enzymexlamlir-opt --mlir-print-op-on-diagnostic=false --sdy-propagation-pipeline --shardy-to-distributed-pipeline --cse --canonicalize --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh2 device_target \"cpu\" axes [!distributed.physical_comm_axis<2, 1>]"' --distributed-search-strategies="beam-size=2 axis-order=extent axis-graph-out=%t.extent.csv" %s -o /dev/null
// RUN: FileCheck --input-file=%t.extent.csv --check-prefix=EXTENT %s

// RUN: enzymexlamlir-opt --mlir-print-op-on-diagnostic=false --sdy-propagation-pipeline --shardy-to-distributed-pipeline --cse --canonicalize --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh2 device_target \"cpu\" axes [!distributed.physical_comm_axis<2, 1>]"' --distributed-search-strategies="beam-size=2 axis-order=separator axis-graph-out=%t.separator.csv" %s -o /dev/null
// RUN: FileCheck --input-file=%t.separator.csv --check-prefix=SEP %s

// RUN: not enzymexlamlir-opt --mlir-print-op-on-diagnostic=false --sdy-propagation-pipeline --shardy-to-distributed-pipeline --cse --canonicalize --insert-physical-mesh=$'configuration-string="distributed.PhysicalMesh @mesh2 device_target \"cpu\" axes [!distributed.physical_comm_axis<2, 1>]"' --distributed-search-strategies="beam-size=2 axis-order=nonsense" %s -o /dev/null 2>&1 | FileCheck --check-prefix=BADNAME %s

// The order the search decides axes in is not a tie-break: setupNextAxis
// subtracts only *overlapping* axes' bindings from the available mesh, so an
// axis decided earlier claims physical factors an overlapping axis decided
// later cannot have. This pins the three orders apart on one module, using the
// axis-graph CSV -- which is written after the reordering, so its axis indices
// are the decision order and its edges are expressed in it.
//
// The module is the two-disjoint-matmuls one from search_axis_graph.mlir: two
// triangles, {0,3,4} and {1,2,5} in declaration order, with extents 8 and 16.
// It discriminates all three because declaration order happens to interleave
// the two components and to put both 16s last.

// Declaration order. The two components are interleaved -- 0 is adjacent to 3
// and 4, while 1, 2 and 5 sit between and after them -- which is the arbitrary
// order the other two arms are measured against.
// IR:      kind,a,b
// IR-NEXT: axis,0,8
// IR-NEXT: axis,1,8
// IR-NEXT: axis,2,8
// IR-NEXT: axis,3,8
// IR-NEXT: axis,4,16
// IR-NEXT: axis,5,16
// IR-NEXT: edge,0,3
// IR-NEXT: edge,0,4
// IR-NEXT: edge,1,2
// IR-NEXT: edge,1,5
// IR-NEXT: edge,2,5
// IR-NEXT: edge,3,4

// Largest extent first, ties keeping declaration order. The two contracted
// axes (16) are decided before any of the four tensor axes (8), so the
// decisions with the largest payload choose from the unreduced mesh. The
// components stay interleaved: this arm ignores the graph by construction.
// EXTENT:      kind,a,b
// EXTENT-NEXT: axis,0,16
// EXTENT-NEXT: axis,1,16
// EXTENT-NEXT: axis,2,8
// EXTENT-NEXT: axis,3,8
// EXTENT-NEXT: axis,4,8
// EXTENT-NEXT: axis,5,8

// Separator first, then each block, recursively. Here the graph is already
// disconnected, so there is no separator to take and the rule reduces to its
// base case: each component is decided as one uninterrupted stretch. That is
// the property under test -- axes that cannot contend are no longer interleaved
// with axes that can, so every decision in the second triangle faces a mesh
// reduced only by its own triangle.
// SEP:      kind,a,b
// SEP-NEXT: axis,0,8
// SEP-NEXT: axis,1,8
// SEP-NEXT: axis,2,16
// SEP-NEXT: axis,3,8
// SEP-NEXT: axis,4,8
// SEP-NEXT: axis,5,16
// SEP-NEXT: edge,0,1
// SEP-NEXT: edge,0,2
// SEP-NEXT: edge,1,2
// SEP-NEXT: edge,3,4
// SEP-NEXT: edge,3,5
// SEP-NEXT: edge,4,5
// SEP-NOT:  edge

// An unknown name is reported against the option rather than silently falling
// back to one of the three, since a typo would otherwise be measured as a
// result.
// BADNAME: axis-order must be 'ir', 'extent' or 'separator', got 'nonsense'

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
