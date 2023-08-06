// NOTE: Assertions have been autogenerated by utils/generate-test-checks.py
// RUN: mlir-opt %s -sparsification | FileCheck %s

// Test to demonstrate the difference between non-annotated dense tensors
// and all-dense-annotated "sparse" tensors. The former class remains as
// two-dimensional tensors that are bufferized by subsequent passes. The
// latter class is linearized into one-dimensional buffers that are backed
// by the runtime support library.

#DenseMatrix = #sparse_tensor.encoding<{ lvlTypes = [ "dense", "dense"  ] }>

#trait_2d = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>,  // A
    affine_map<(i,j) -> (i,j)>   // X (out)
  ],
  iterator_types = ["parallel", "parallel"],
  doc = "X(i,j) = A(i,j) + 1"
}

#trait_3d = {
  indexing_maps = [
    affine_map<(i,j,k) -> (i,j,k)>,  // A
    affine_map<(i,j,k) -> (i,j)>     // X (out)
  ],
  iterator_types = ["parallel", "parallel", "reduction"],
  doc = "X(i,j) += A(i,j,k)"
}

//
// Test with an all-dense-annotated "sparse" matrix as input and
// a non-annotated dense matrix as output.
//
// CHECK-LABEL:   func @dense1(
// CHECK-SAME:                 %[[VAL_0:.*]]: tensor<32x16xf32, #sparse_tensor.encoding<{{.*}}>>,
// CHECK-SAME:                 %[[VAL_1:.*]]: tensor<32x16xf32>) -> tensor<32x16xf32> {
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 32 : index
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 16 : index
// CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_6:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_7:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<32x16xf32, #sparse_tensor.encoding<{{.*}}>> to memref<?xf32>
// CHECK:           %[[VAL_8:.*]] = bufferization.to_memref %[[VAL_1]] : memref<32x16xf32>
// CHECK:           scf.for %[[VAL_9:.*]] = %[[VAL_5]] to %[[VAL_3]] step %[[VAL_6]] {
// CHECK:             scf.for %[[VAL_10:.*]] = %[[VAL_5]] to %[[VAL_4]] step %[[VAL_6]] {
// CHECK:               %[[VAL_11:.*]] = arith.muli %[[VAL_9]], %[[VAL_4]] : index
// CHECK:               %[[VAL_12:.*]] = arith.addi %[[VAL_11]], %[[VAL_10]] : index
// CHECK:               %[[VAL_13:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_12]]] : memref<?xf32>
// CHECK:               %[[VAL_14:.*]] = arith.addf %[[VAL_13]], %[[VAL_2]] : f32
// CHECK:               memref.store %[[VAL_14]], %[[VAL_8]]{{\[}}%[[VAL_9]], %[[VAL_10]]] : memref<32x16xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           %[[VAL_15:.*]] = bufferization.to_tensor %[[VAL_8]] : memref<32x16xf32>
// CHECK:           return %[[VAL_15]] : tensor<32x16xf32>
// CHECK:         }
func.func @dense1(%arga: tensor<32x16xf32, #DenseMatrix>,
                  %argx: tensor<32x16xf32>)
	     -> tensor<32x16xf32> {
  %c = arith.constant 1.0 : f32
  %0 = linalg.generic #trait_2d
     ins(%arga: tensor<32x16xf32, #DenseMatrix>)
    outs(%argx: tensor<32x16xf32>) {
      ^bb(%a: f32, %x: f32):
        %1 = arith.addf %a, %c : f32
        linalg.yield %1 : f32
  } -> tensor<32x16xf32>
  return %0 : tensor<32x16xf32>
}

//
// Test with a non-annotated dense matrix as input and
// an all-dense annotated "sparse" matrix as output.
//
// CHECK-LABEL:   func @dense2(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<32x16xf32>,
// CHECK-SAME:      %[[VAL_1:.*]]: tensor<32x16xf32, #sparse_tensor.encoding<{{.*}}>>) -> tensor<32x16xf32, #sparse_tensor.encoding<{{.*}}>> {
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 32 : index
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 16 : index
// CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_6:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_7:.*]] = bufferization.to_memref %[[VAL_0]] : memref<32x16xf32>
// CHECK:           %[[VAL_8:.*]] = sparse_tensor.values %[[VAL_1]] : tensor<32x16xf32, #sparse_tensor.encoding<{{.*}}>> to memref<?xf32>
// CHECK:           scf.for %[[VAL_9:.*]] = %[[VAL_5]] to %[[VAL_3]] step %[[VAL_6]] {
// CHECK:             scf.for %[[VAL_10:.*]] = %[[VAL_5]] to %[[VAL_4]] step %[[VAL_6]] {
// CHECK:               %[[VAL_11:.*]] = arith.muli %[[VAL_9]], %[[VAL_4]] : index
// CHECK:               %[[VAL_12:.*]] = arith.addi %[[VAL_11]], %[[VAL_10]] : index
// CHECK:               %[[VAL_13:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_9]], %[[VAL_10]]] : memref<32x16xf32>
// CHECK:               %[[VAL_14:.*]] = arith.addf %[[VAL_13]], %[[VAL_2]] : f32
// CHECK:               memref.store %[[VAL_14]], %[[VAL_8]]{{\[}}%[[VAL_12]]] : memref<?xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           %[[VAL_15:.*]] = sparse_tensor.load %[[VAL_1]] : tensor<32x16xf32, #sparse_tensor.encoding<{{.*}}>>
// CHECK:           return %[[VAL_15]] : tensor<32x16xf32, #sparse_tensor.encoding<{{.*}}>>
// CHECK:         }
func.func @dense2(%arga: tensor<32x16xf32>,
                  %argx: tensor<32x16xf32, #DenseMatrix>)
	     -> tensor<32x16xf32, #DenseMatrix> {
  %c = arith.constant 1.0 : f32
  %0 = linalg.generic #trait_2d
     ins(%arga: tensor<32x16xf32>)
    outs(%argx: tensor<32x16xf32, #DenseMatrix>) {
      ^bb(%a: f32, %x: f32):
        %1 = arith.addf %a, %c : f32
        linalg.yield %1 : f32
  } -> tensor<32x16xf32, #DenseMatrix>
  return %0 : tensor<32x16xf32, #DenseMatrix>
}


//
// Test with a non-annotated dense matrix as input and
// an all-dense annotated "sparse" matrix as output.
// The missing innermost "k" index (due to a reduction) is accounted
// for by scalarizing the reduction operation for the output tensor.
//
// CHECK-LABEL:   func @dense3(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<32x16x8xf32>,
// CHECK-SAME:      %[[VAL_1:.*]]: tensor<32x16xf32, #sparse_tensor.encoding<{{.*}}>>) -> tensor<32x16xf32, #sparse_tensor.encoding<{{.*}}>> {
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 8 : index
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 32 : index
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 16 : index
// CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_6:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_7:.*]] = bufferization.to_memref %[[VAL_0]] : memref<32x16x8xf32>
// CHECK:           %[[VAL_8:.*]] = sparse_tensor.values %[[VAL_1]] : tensor<32x16xf32, #sparse_tensor.encoding<{{.*}}}>> to memref<?xf32>
// CHECK:           scf.for %[[VAL_9:.*]] = %[[VAL_5]] to %[[VAL_3]] step %[[VAL_6]] {
// CHECK:             scf.for %[[VAL_10:.*]] = %[[VAL_5]] to %[[VAL_4]] step %[[VAL_6]] {
// CHECK:               %[[VAL_11:.*]] = arith.muli %[[VAL_9]], %[[VAL_4]] : index
// CHECK:               %[[VAL_12:.*]] = arith.addi %[[VAL_11]], %[[VAL_10]] : index
// CHECK:               %[[VAL_13:.*]] = memref.load %[[VAL_8]]{{\[}}%[[VAL_12]]] : memref<?xf32>
// CHECK:               %[[VAL_14:.*]] = scf.for %[[VAL_15:.*]] = %[[VAL_5]] to %[[VAL_2]] step %[[VAL_6]] iter_args(%[[VAL_16:.*]] = %[[VAL_13]]) -> (f32) {
// CHECK:                 %[[VAL_17:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_9]], %[[VAL_10]], %[[VAL_15]]] : memref<32x16x8xf32>
// CHECK:                 %[[VAL_18:.*]] = arith.addf %[[VAL_16]], %[[VAL_17]] : f32
// CHECK:                 scf.yield %[[VAL_18]] : f32
// CHECK:               }
// CHECK:               memref.store %[[VAL_19:.*]], %[[VAL_8]]{{\[}}%[[VAL_12]]] : memref<?xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           %[[VAL_20:.*]] = sparse_tensor.load %[[VAL_1]] : tensor<32x16xf32, #sparse_tensor.encoding<{{.*}}>>
// CHECK:           return %[[VAL_20]] : tensor<32x16xf32, #sparse_tensor.encoding<{{.*}}>>
// CHECK:         }
func.func @dense3(%arga: tensor<32x16x8xf32>,
                  %argx: tensor<32x16xf32, #DenseMatrix>)
	     -> tensor<32x16xf32, #DenseMatrix> {
  %0 = linalg.generic #trait_3d
     ins(%arga: tensor<32x16x8xf32>)
    outs(%argx: tensor<32x16xf32, #DenseMatrix>) {
      ^bb(%a: f32, %x: f32):
        %1 = arith.addf %x, %a : f32
        linalg.yield %1 : f32
  } -> tensor<32x16xf32, #DenseMatrix>
  return %0 : tensor<32x16xf32, #DenseMatrix>
}
