use super::tensor::Tensor;

pub fn add(t1: &Tensor, t2: &Tensor) -> Result<Tensor, String> {
    if t1.shape != t2.shape {
        return Err("Tensors must have the same shape for addition".to_string());
    }
    let result_data: Vec<f32> = t1.data.iter().zip(&t2.data).map(|(&a, &b)| a + b).collect();
    Ok(Tensor::new(result_data, t1.shape.clone()))
}

pub fn mul(t1: &Tensor, t2: &Tensor) -> Result<Tensor, String> {
    if t1.shape != t2.shape {
        return Err("Tensors must have the same shape for multiplication".to_string());
    }
    let result_data: Vec<f32> = t1.data.iter().zip(&t2.data).map(|(&a, &b)| a * b).collect();
    Ok(Tensor::new(result_data, t1.shape.clone()))
}
