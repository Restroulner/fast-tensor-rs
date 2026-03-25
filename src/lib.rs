//! FastTensor-rs: A lightweight, high-speed tensor manipulation library for edge AI.

pub mod tensor;
pub mod ops;

pub use tensor::Tensor;
pub use ops::{add, mul};

#[cfg(test)]
mod tests {
    use super::*{
        tensor::Tensor,
        ops::{
            add,
            mul
        }
    };

    #[test]
    fn test_tensor_creation() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        assert_eq!(t.data, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(t.shape, vec![2, 2]);
    }

    #[test]
    fn test_tensor_add() {
        let t1 = Tensor::new(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::new(vec![3.0, 4.0], vec![2]);
        let t3 = add(&t1, &t2).unwrap();
        assert_eq!(t3.data, vec![4.0, 6.0]);
    }

    #[test]
    fn test_tensor_mul() {
        let t1 = Tensor::new(vec![1.0, 2.0], vec![2]);
        let t2 = Tensor::new(vec![3.0, 4.0], vec![2]);
        let t3 = mul(&t1, &t2).unwrap();
        assert_eq!(t3.data, vec![3.0, 8.0]);
    }
}
