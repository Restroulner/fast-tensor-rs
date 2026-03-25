pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        // Basic shape validation
        let total_elements: usize = shape.iter().product();
        assert_eq!(data.len(), total_elements, "Data length must match product of shape dimensions");
        Tensor { data, shape }
    }

    pub fn num_elements(&self) -> usize {
        self.data.len()
    }
}
