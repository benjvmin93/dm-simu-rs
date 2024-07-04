#[cfg(test)]
mod tests_tensor {
    use dm_simu_rs::{tensor::Tensor, tools::bitwise_int_to_bin_vec};
    use num_complex::Complex;

    #[test]
    fn test_tensor_creation() {
        let shape = vec![2, 3];
        let tensor: Tensor<Complex<f64>> = Tensor::new(&shape);
        assert_eq!(tensor.shape, shape);
        assert_eq!(tensor.data.len(), 6); // 2 * 3 = 6 elements
        for &value in &tensor.data {
            assert_eq!(value, Complex::new(0.0, 0.0)); // All elements initialized to zero
        }
    }

    #[test]
    fn test_tensor_get_set() {
        let shape = vec![2, 2];
        let mut tensor: Tensor<Complex<f64>> = Tensor::new(&shape);

        // Set elements
        tensor.set(&[0, 0], Complex::new(1.0, 0.0));
        tensor.set(&[0, 1], Complex::new(2.0, 0.0));
        tensor.set(&[1, 0], Complex::new(3.0, 0.0));
        tensor.set(&[1, 1], Complex::new(4.0, 0.0));

        // Get elements
        assert_eq!(tensor.get(&[0, 0]), Complex::new(1.0, 0.0));
        assert_eq!(tensor.get(&[0, 1]), Complex::new(2.0, 0.0));
        assert_eq!(tensor.get(&[1, 0]), Complex::new(3.0, 0.));
        assert_eq!(tensor.get(&[1, 1]), Complex::new(4.0, 0.0));
    }

    #[test]
    fn test_tensor_addition() {
        let shape = vec![2, 2];
        let tensor1 = Tensor {
            data: vec![
                Complex::new(1.0, 0.0),
                Complex::new(2.0, 0.),
                Complex::new(3., 0.),
                Complex::new(4., 0.),
            ],
            shape: shape.clone(),
        };
        let tensor2 = Tensor {
            data: vec![
                Complex::new(5.0, 0.),
                Complex::new(6., 0.),
                Complex::new(7., 0.),
                Complex::new(8., 0.),
            ],
            shape: shape.clone(),
        };

        let result = tensor1.add(&tensor2);
        assert_eq!(
            result.data,
            vec![
                Complex::new(6., 0.),
                Complex::new(8., 0.),
                Complex::new(10., 0.),
                Complex::new(12., 0.)
            ]
        );
    }

    #[test]
    fn test_tensor_multiplication() {
        let shape = vec![2, 2];
        let tensor1 = Tensor {
            data: vec![
                Complex::new(1., 0.),
                Complex::new(2., 0.),
                Complex::new(3., 0.),
                Complex::new(4., 0.),
            ],
            shape: shape.clone(),
        };
        let tensor2 = Tensor {
            data: vec![
                Complex::new(5., 0.),
                Complex::new(6., 0.),
                Complex::new(7., 0.),
                Complex::new(8., 0.),
            ],
            shape: shape.clone(),
        };

        let result = tensor1.multiply(&tensor2);
        assert_eq!(
            result.data,
            vec![
                Complex::new(5., 0.),
                Complex::new(12., 0.),
                Complex::new(21., 0.),
                Complex::new(32., 0.)
            ]
        );
    }

    #[test]
    fn test_tensor_get_large_tensor() {
        let nqubits = 10;
        let size = 2_i32.pow(nqubits);
        let data = (0..(size * size)).collect::<Vec<i32>>();
        let t = Tensor::from_vec(data, vec![2; 2 * nqubits as usize]);
        for (i, x) in t.data.iter().enumerate() {
            let indices = bitwise_int_to_bin_vec(i, 2 * nqubits as usize);
            let elt = t.get(&indices);
            assert_eq!(elt, *x);
        }
    }

    #[test]
    fn test_tensor_get_specific_data() {
        let nqubits = 7;
        let size = 2_i32.pow(nqubits) as usize;
        let mut data = vec![Complex::ZERO; size * size];
        let val = Complex::new(0.25, 0.);
        for i in 0..4 {
            data[i] = val;
        }
        let t = Tensor::from_vec(data, vec![2; 2 * nqubits as usize]);
        for (i, x) in t.data.iter().enumerate() {
            let indices = bitwise_int_to_bin_vec(i, 2 * nqubits as usize);
            let elt = t.get(&indices);
            if i < 4 {
                assert_eq!(elt, val);
            } else {
                assert_eq!(elt, *x);
            }
        }
    }

    #[test]
    fn test_tensor_product_simple_case() {
        // Create the first tensor: [1, 2, 3]
        let tensor1 = Tensor::from_vec(
            vec![
                Complex::new(1., 0.),
                Complex::new(2., 0.),
                Complex::new(3., 0.),
            ],
            vec![3],
        );

        // Create the second tensor: [4, 5]
        let tensor2 = Tensor::from_vec(vec![Complex::new(4., 0.), Complex::new(5., 0.)], vec![2]);

        // Calculate the tensor product

        let result_tensor = tensor1.product(&tensor2);

        // Expected result:
        // Shape: [3, 2]
        // Data: [1*4, 1*5, 2*4, 2*5, 3*4, 3*5] => [4, 5, 8, 10, 12, 15]
        assert_eq!(result_tensor.shape, vec![3, 2]);
        assert_eq!(
            result_tensor.data,
            vec![
                Complex::new(4., 0.),
                Complex::new(5., 0.),
                Complex::new(8., 0.),
                Complex::new(10., 0.),
                Complex::new(12., 0.),
                Complex::new(15., 0.)
            ]
        );
    }

    #[test]
    fn test_tensor_product_with_zeros() {
        let tensor1 = Tensor::from_vec(vec![0, 0], vec![2]);
        let tensor2 = Tensor::from_vec(vec![1, 2], vec![2]);

        let result_tensor = tensor1.product(&tensor2);

        assert_eq!(result_tensor.shape, vec![2, 2]);
        assert_eq!(result_tensor.data, vec![0, 0, 0, 0]);
    }

    #[test]
    fn test_tensor_product_different_shapes() {
        let tensor1 = Tensor::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8], vec![4, 2]);

        let tensor2 = Tensor::from_vec(vec![1, 2, 3, 4, 5], vec![5]);

        let result_tensor = tensor1.product(&tensor2);

        // Expected result:
        // Shape: [3, 2]
        // Data: [1*4, 1*5, 2*4, 2*5, 3*4, 3*5] => [4, 5, 8, 10, 12, 15]
        assert_eq!(result_tensor.shape, vec![4, 2, 5]);
        assert_eq!(
            result_tensor.data,
            vec![
                1, 2, 3, 4, 5, 2, 4, 6, 8, 10, 3, 6, 9, 12, 15, 4, 8, 12, 16, 20, 5, 10, 15, 20,
                25, 6, 12, 18, 24, 30, 7, 14, 21, 28, 35, 8, 16, 24, 32, 40
            ]
        );
    }

    #[test]
    fn test_tensordot_2D_1() {
        // Tensor A (shape: [2, 2])
        let a_data = vec![
            Complex::new(1., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(1., 0.),
        ];

        // Tensor B (shape: [2, 2])
        let b_data = vec![
            Complex::new(1., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
        ];

        // Expected tensor (shape: [2, 2])
        let expected_data = vec![
            Complex::new(1., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
            Complex::new(0., 0.),
        ];

        let a_tensor = Tensor::from_vec(a_data, vec![2, 2]);
        let b_tensor = Tensor::from_vec(b_data, vec![2, 2]);
        let expected_tensor = Tensor::from_vec(expected_data, vec![2, 2]);

        // Tensordot on axes 1 of A and 0 of B
        let result_tensor = a_tensor.tensordot(&b_tensor, (&[1], &[0])).unwrap();

        assert_eq!(result_tensor.data, expected_tensor.data);
        assert_eq!(result_tensor.shape, expected_tensor.shape);
    }

    #[test]
    fn test_tensordot_2D_2() {
        // Tensor A (shape: [2, 2])
        let a_data = vec![
            Complex::new(1., 0.),
            Complex::new(2., 0.),
            Complex::new(3., 0.),
            Complex::new(4., 0.),
        ];

        // Tensor B (shape: [2, 2])
        let b_data = vec![
            Complex::new(5., 0.),
            Complex::new(6., 0.),
            Complex::new(7., 0.),
            Complex::new(8., 0.),
        ];

        // Expected tensor (shape: [2, 2])
        let expected_data = vec![
            Complex::new(19., 0.),
            Complex::new(22., 0.),
            Complex::new(43., 0.),
            Complex::new(50., 0.),
        ];

        let a_tensor = Tensor::from_vec(a_data, vec![2, 2]);
        let b_tensor = Tensor::from_vec(b_data, vec![2, 2]);
        let expected_tensor = Tensor::from_vec(expected_data, vec![2, 2]);

        // Tensordot on axes 1 of A and 0 of B
        let result_tensor = a_tensor.tensordot(&b_tensor, (&[1], &[0])).unwrap();

        assert_eq!(result_tensor.data, expected_tensor.data);
        assert_eq!(result_tensor.shape, expected_tensor.shape);
    }

    #[test]
    fn test_tensordot_different_shapes() {
        // Tensor A (shape: [2, 2])
        let a_data = vec![
            Complex::new(1., 0.),
            Complex::new(2., 0.),
            Complex::new(3., 0.),
            Complex::new(4., 0.),
        ];

        // Tensor B (shape: [2, 2, 2, 2])
        let b_data = vec![
            Complex::new(1.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(3.0, 0.0),
            Complex::new(4.0, 0.0),
            Complex::new(5.0, 0.0),
            Complex::new(6.0, 0.0),
            Complex::new(7.0, 0.0),
            Complex::new(8.0, 0.0),
            Complex::new(9.0, 0.0),
            Complex::new(10.0, 0.0),
            Complex::new(11.0, 0.0),
            Complex::new(12.0, 0.0),
            Complex::new(13.0, 0.0),
            Complex::new(14.0, 0.0),
            Complex::new(15.0, 0.0),
            Complex::new(16.0, 0.0),
        ];

        // Expected tensor (shape: [2, 2, 2, 2])
        let expected_data = vec![
            Complex::new(19., 0.0),
            Complex::new(22.0, 0.0),
            Complex::new(25.0, 0.0),
            Complex::new(28.0, 0.0),
            Complex::new(31.0, 0.0),
            Complex::new(34.0, 0.0),
            Complex::new(37.0, 0.0),
            Complex::new(40.0, 0.0),
            Complex::new(39.0, 0.0),
            Complex::new(46.0, 0.0),
            Complex::new(53.0, 0.0),
            Complex::new(60.0, 0.0),
            Complex::new(67.0, 0.0),
            Complex::new(74.0, 0.0),
            Complex::new(81.0, 0.0),
            Complex::new(88.0, 0.0),
        ];

        let a_tensor = Tensor::from_vec(a_data, vec![2, 2]);
        let b_tensor = Tensor::from_vec(b_data, vec![2, 2, 2, 2]);
        let expected_tensor = Tensor::from_vec(expected_data, vec![2, 2, 2, 2]);
        let result_tensor = a_tensor.tensordot(&b_tensor, (&[1], &[0])).unwrap();

        assert_eq!(result_tensor.data, expected_tensor.data);
        assert_eq!(result_tensor.shape, expected_tensor.shape);
    }

    #[test]
    fn test_tensordot_3Dx2D_1() {
        let t_a = Tensor::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], vec![2, 3, 2]);
        let t_b = Tensor::from_vec(vec![13, 14, 15, 16, 17, 18], vec![3, 2]);
        let res = t_a.tensordot(&t_b, (&[1, 2], &[0, 1])).unwrap();
        assert_eq!(res.shape, vec![2]);
        assert_eq!(res.data, vec![343, 901]);
    }

    #[test]
    fn test_tensordot_3Dx2D_2() {
        let t_a = Tensor::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], vec![2, 3, 2]);
        let t_b = Tensor::from_vec(vec![13, 14, 15, 16, 17, 18], vec![3, 2]);
        let res = t_a.tensordot(&t_b, (&[2, 1], &[1, 0])).unwrap();
        assert_eq!(res.shape, vec![2]);
        assert_eq!(res.data, vec![343, 901]);
    }

    #[test]
    fn test_tensordot_2Dx3D_1() {
        let t_a = Tensor::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], vec![2, 3, 2]);
        let t_b = Tensor::from_vec(vec![13, 14, 15, 16, 17, 18], vec![3, 2]);
        let res = t_b.tensordot(&t_a, (&[0, 1], &[1, 2])).unwrap();
        assert_eq!(res.shape, vec![2]);
        assert_eq!(res.data, vec![343, 901]);
    }

    #[test]
    fn test_tensordot_2Dx3D_2() {
        let t_a = Tensor::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], vec![2, 3, 2]);
        let t_b = Tensor::from_vec(vec![13, 14, 15, 16, 17, 18], vec![3, 2]);
        let res = t_b.tensordot(&t_a, (&[1, 0], &[2, 1])).unwrap();
        assert_eq!(res.shape, vec![2]);
        assert_eq!(res.data, vec![343, 901]);
    }

    #[test]
    fn test_transpose_single_axis() {
        // Create a 1D tensor
        let tensor = Tensor::from_vec(
            vec![
                Complex::new(1.0, 0.0),
                Complex::new(2.0, 0.0),
                Complex::new(3.0, 0.0),
            ],
            vec![3],
        );

        // Transpose axes [0] (identity transpose for 1D tensor)
        let transposed = tensor.transpose(&[0]).unwrap();

        // Check the new shape (should be the same as the original)
        assert_eq!(transposed.shape, vec![3]);

        // Check the new data (should be the same as the original)
        assert_eq!(transposed.data, tensor.data);
    }

    #[test]
    fn test_transpose_identity() {
        // Create a 2x3 tensor
        let tensor = Tensor::from_vec(
            vec![
                Complex::new(1.0, 0.0),
                Complex::new(2.0, 0.0),
                Complex::new(3.0, 0.0),
                Complex::new(4.0, 0.0),
                Complex::new(5.0, 0.0),
                Complex::new(6.0, 0.0),
            ],
            vec![2, 3],
        );

        // Transpose axes [0, 1] (identity transpose)
        let transposed = tensor.transpose(&[0, 1]).unwrap();

        // Check the new shape (should be the same as the original)
        assert_eq!(transposed.shape, vec![2, 3]);

        // Check the new data (should be the same as the original)
        assert_eq!(transposed.data, tensor.data);
    }

    #[test]
    fn test_transpose_2d() {
        // Create a 2x3 tensor
        let tensor = Tensor::from_vec(
            vec![
                Complex::new(1.0, 0.0),
                Complex::new(2.0, 0.0),
                Complex::new(3.0, 0.0),
                Complex::new(4.0, 0.0),
                Complex::new(5.0, 0.0),
                Complex::new(6.0, 0.0),
            ],
            vec![2, 3],
        );

        // Transpose axes [1, 0]
        let transposed = tensor.transpose(&[1, 0]).unwrap();

        // Check the new shape
        assert_eq!(transposed.shape, vec![3, 2]);

        // Check the new data
        assert_eq!(
            transposed.data,
            vec![
                Complex::new(1.0, 0.0),
                Complex::new(4.0, 0.0),
                Complex::new(2.0, 0.0),
                Complex::new(5.0, 0.0),
                Complex::new(3.0, 0.0),
                Complex::new(6.0, 0.0)
            ]
        );
    }

    #[test]
    fn test_transpose_3d() {
        // Create a 2x3x4 tensor
        let tensor: Tensor<Complex<f64>> = Tensor::from_vec(
            (0..24)
                .map(|x| Complex::new(x as f64, 0.0))
                .collect::<Vec<Complex<f64>>>(),
            vec![2, 3, 4],
        );

        // Transpose axes [2, 0, 1]
        let transposed = tensor.transpose(&[2, 0, 1]).unwrap();

        // Check the new shape
        assert_eq!(transposed.shape, vec![4, 2, 3]);

        // Check the new data
        let expected_data = vec![
            Complex::new(0.0, 0.0),
            Complex::new(4.0, 0.0),
            Complex::new(8.0, 0.0),
            Complex::new(12.0, 0.0),
            Complex::new(16.0, 0.0),
            Complex::new(20.0, 0.0),
            Complex::new(1.0, 0.0),
            Complex::new(5.0, 0.0),
            Complex::new(9.0, 0.0),
            Complex::new(13.0, 0.0),
            Complex::new(17.0, 0.0),
            Complex::new(21.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(6.0, 0.0),
            Complex::new(10.0, 0.0),
            Complex::new(14.0, 0.0),
            Complex::new(18.0, 0.0),
            Complex::new(22.0, 0.0),
            Complex::new(3.0, 0.0),
            Complex::new(7.0, 0.0),
            Complex::new(11.0, 0.0),
            Complex::new(15.0, 0.0),
            Complex::new(19.0, 0.0),
            Complex::new(23.0, 0.0),
        ];
        assert_eq!(transposed.data, expected_data);
    }

    #[test]
    fn test_transpose_4d() {
        // Create a 2x2x2x2 tensor
        let vec = (0..16)
            .map(|x| Complex::new(x as f64, 0.0))
            .collect::<Vec<Complex<f64>>>();
        let tensor = Tensor::from_vec(vec, vec![2, 2, 2, 2]);

        // Transpose
        let transposed = tensor.transpose(&[]).unwrap();

        // Check the new shape
        assert_eq!(transposed.shape, vec![2, 2, 2, 2]);

        // Check the new data
        let expected_data = vec![
            Complex::new(0.0, 0.0),
            Complex::new(8.0, 0.0),
            Complex::new(4.0, 0.0),
            Complex::new(12.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(10.0, 0.0),
            Complex::new(6.0, 0.0),
            Complex::new(14.0, 0.0),
            Complex::new(1.0, 0.0),
            Complex::new(9.0, 0.0),
            Complex::new(5.0, 0.0),
            Complex::new(13.0, 0.0),
            Complex::new(3.0, 0.0),
            Complex::new(11.0, 0.0),
            Complex::new(7.0, 0.0),
            Complex::new(15.0, 0.0),
        ];
        assert_eq!(transposed.data, expected_data);
    }

    #[test]
    fn test_transpose_empty_axes() {
        let tensor = Tensor::from_vec((0..6).collect(), vec![2, 3]);

        let transposed_tensor = tensor.transpose(&[]).unwrap();
        assert_eq!(transposed_tensor.shape, vec![3, 2]);

        let expected_data = vec![0, 3, 1, 4, 2, 5];
        assert_eq!(transposed_tensor.data, expected_data);
    }

    #[test]
    fn test_transpose_3d_empty_axes() {
        let tensor = Tensor::from_vec((0..24).collect(), vec![2, 3, 4]);

        let transposed_tensor = tensor.transpose(&[]).unwrap();
        assert_eq!(transposed_tensor.shape, vec![4, 3, 2]);

        let expected_data = vec![
            0, 12, 4, 16, 8, 20, 1, 13, 5, 17, 9, 21, 2, 14, 6, 18, 10, 22, 3, 15, 7, 19, 11, 23,
        ];
        assert_eq!(transposed_tensor.data, expected_data);
    }

    #[test]
    fn test_moveaxis_unchanged() {
        // Create a 2x2 tensor
        let vec = (0..4)
            .map(|x| Complex::new(x as f64, 0.))
            .collect::<Vec<Complex<f64>>>();
        let tensor = Tensor::from_vec(vec, vec![2, 2]);

        // Move axis (0, 1) to (0, 1) => Remains tensor unchanged
        let new_tensor = tensor.moveaxis(&[0, 1], &[0, 1]).unwrap();

        assert_eq!(new_tensor.shape, vec![2, 2]);
        let expected_data = vec![
            Complex::new(0., 0.),
            Complex::new(1., 0.),
            Complex::new(2., 0.),
            Complex::new(3., 0.),
        ];
        assert_eq!(new_tensor.data, expected_data);
    }

    #[test]
    fn test_moveaxis_transpose() {
        // Create a 2x2 tensor
        let tensor = Tensor::from_vec((0..4).collect(), vec![2, 2]);

        // Move axis (0, 1) to (1, 0) => Transpose tensor
        let new_tensor = tensor.moveaxis(&[0, 1], &[1, 0]).unwrap();
        assert_eq!(new_tensor.shape, vec![2, 2]);

        let expected_data = vec![0, 2, 1, 3];
        assert_eq!(new_tensor.data, expected_data);
    }

    #[test]
    fn test_moveaxis_4d_tensor() {
        // Create a 2x2x2x2 tensor
        let tensor = Tensor::from_vec(
            (0..16)
                .map(|x| Complex::new(x as f64, 0.))
                .collect::<Vec<Complex<f64>>>(),
            vec![2, 2, 2, 2],
        );

        // Move axis (0, tensor.shape.len() - 1) to (0, 2)
        let new_tensor = tensor
            .moveaxis(&[0, (tensor.shape.len() - 1) as i32], &[0, 2])
            .unwrap();
        assert_eq!(new_tensor.shape, vec![2, 2, 2, 2]);

        let expected_data = vec![
            Complex::new(0., 0.),
            Complex::new(2., 0.),
            Complex::new(1., 0.),
            Complex::new(3., 0.),
            Complex::new(4., 0.),
            Complex::new(6., 0.),
            Complex::new(5., 0.),
            Complex::new(7., 0.),
            Complex::new(8., 0.),
            Complex::new(10., 0.),
            Complex::new(9., 0.),
            Complex::new(11., 0.),
            Complex::new(12., 0.),
            Complex::new(14., 0.),
            Complex::new(13., 0.),
            Complex::new(15., 0.),
        ];
        assert_eq!(new_tensor.data, expected_data);
    }
    #[test]
    fn test_moveaxis_single_to_single() {
        let tensor = Tensor {
            data: vec![
                Complex::new(1., 0.),
                Complex::new(2., 0.),
                Complex::new(3., 0.),
                Complex::new(4., 0.),
                Complex::new(5., 0.),
                Complex::new(6., 0.),
            ],
            shape: vec![2, 3], // 2 rows, 3 columns
        };

        // Move axis 0 to position 1 (2x3 to 3x2)
        let moved_matrix = tensor.moveaxis(&[0], &[1]).unwrap();
        assert_eq!(
            moved_matrix.data,
            vec![
                Complex::new(1., 0.),
                Complex::new(4., 0.),
                Complex::new(2., 0.),
                Complex::new(5., 0.),
                Complex::new(3., 0.),
                Complex::new(6., 0.)
            ]
        );
        assert_eq!(moved_matrix.shape, vec![3, 2]);
    }
    #[test]
    fn test_moveaxis_with_negative_indices_1() {
        let tensor = Tensor {
            data: vec![
                Complex::new(1., 0.),
                Complex::new(2., 0.),
                Complex::new(3., 0.),
                Complex::new(4., 0.),
                Complex::new(5., 0.),
                Complex::new(6., 0.),
                Complex::new(7., 0.),
                Complex::new(8., 0.),
                Complex::new(9., 0.),
                Complex::new(10., 0.),
                Complex::new(11., 0.),
                Complex::new(12., 0.),
            ],
            shape: vec![2, 2, 3],
        };

        let result = tensor.moveaxis(&[0, -1], &[-1, 0]).unwrap();
        assert_eq!(result.shape, vec![3, 2, 2]);
        assert_eq!(
            result.data,
            vec![
                Complex::new(1.0, 0.0),
                Complex::new(7.0, 0.0),
                Complex::new(4.0, 0.0),
                Complex::new(10.0, 0.0),
                Complex::new(2.0, 0.0),
                Complex::new(8.0, 0.0),
                Complex::new(5.0, 0.0),
                Complex::new(11.0, 0.0),
                Complex::new(3.0, 0.0),
                Complex::new(9.0, 0.0),
                Complex::new(6.0, 0.0),
                Complex::new(12.0, 0.0),
            ]
        );
    }
    #[test]
    fn test_moveaxis_with_negative_indices_2() {
        let tensor = Tensor::from_vec(
            vec![
                Complex::new(1.0, 0.0),
                Complex::new(2.0, 0.0),
                Complex::new(3.0, 0.0),
                Complex::new(4.0, 0.0),
                Complex::new(5.0, 0.0),
                Complex::new(6.0, 0.0),
                Complex::new(7.0, 0.0),
                Complex::new(8.0, 0.0),
            ],
            vec![2, 2, 2],
        );

        let result = tensor.moveaxis(&[-1, -2], &[0, 1]).unwrap();
        let expected = Tensor::from_vec(
            vec![
                Complex::new(1.0, 0.0),
                Complex::new(5.0, 0.0),
                Complex::new(3.0, 0.0),
                Complex::new(7.0, 0.0),
                Complex::new(2.0, 0.0),
                Complex::new(6.0, 0.0),
                Complex::new(4.0, 0.0),
                Complex::new(8.0, 0.0),
            ],
            vec![2, 2, 2],
        );
        assert_eq!(result.data, expected.data);
        assert_eq!(result.shape, expected.shape);
    }
    #[test]
    fn test_moveaxis_3d() {
        let data = (0..24)
            .into_iter()
            .map(|e| Complex::new(e as f64, 0.))
            .collect();
        let tensor_3d = Tensor {
            data,
            shape: vec![2, 3, 4], // Shape (2, 3, 4)
        };

        // Move axes 0 to 2 and 2 to 0
        let moved_tensor_3d = tensor_3d.moveaxis(&[0, 2], &[2, 0]).unwrap();
        assert_eq!(
            moved_tensor_3d.data,
            vec![
                Complex::new(0., 0.),
                Complex::new(12., 0.),
                Complex::new(4., 0.),
                Complex::new(16., 0.),
                Complex::new(8., 0.),
                Complex::new(20., 0.),
                Complex::new(1., 0.),
                Complex::new(13., 0.),
                Complex::new(5., 0.),
                Complex::new(17., 0.),
                Complex::new(9., 0.),
                Complex::new(21., 0.),
                Complex::new(2., 0.),
                Complex::new(14., 0.),
                Complex::new(6., 0.),
                Complex::new(18., 0.),
                Complex::new(10., 0.),
                Complex::new(22., 0.),
                Complex::new(3., 0.),
                Complex::new(15., 0.),
                Complex::new(7., 0.),
                Complex::new(19., 0.),
                Complex::new(11., 0.),
                Complex::new(23., 0.)
            ]
        );
        assert_eq!(moved_tensor_3d.shape, vec![4, 3, 2]);
    }
}
