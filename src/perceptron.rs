pub struct Perceptron {
    pub weights: Vec<f64>,
    pub bias: f64,
    pub learning_rate: f64,
}

impl Perceptron {
    pub fn new(input_size: usize, learning_rate: f64) -> Perceptron {
        Perceptron {
            weights: vec![0.0; input_size],
            bias: 0.0,
            learning_rate,
        }
    }

    pub fn predict(&self, inputs: &Vec<f64>) -> f64 {
        let weighted_sum: f64 = self
            .weights
            .iter()
            .zip(inputs.iter())
            .map(|(w, &i)| w * i)
            .sum();

        let weighted_sum_with_bias = weighted_sum + self.bias;
        let result = Perceptron::activation(weighted_sum_with_bias);

        result
    }

    /// The Activation Function's job is to compress the state
    /// of the polynomial output into a binary classification
    pub fn activation(polynomial_output: f64) -> f64 {
        if polynomial_output > 0.0 {
            1.0
        } else {
            0.0
        }
    }

    pub fn train(&mut self, training_data: &[(Vec<f64>, f64)]) {
        for &(ref inputs, target) in training_data.iter() {
            let output = self.predict(inputs);
            let error = target - output;
            for (w, &i) in self.weights.iter_mut().zip(inputs.iter()) {
                *w += self.learning_rate * error * i;
            }
            self.bias += self.learning_rate * error;
        }
    }
}
