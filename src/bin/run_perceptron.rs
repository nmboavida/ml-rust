use ml_rust::perceptron::Perceptron;

fn main() {
    let mut perceptron = Perceptron::new(2, 0.1);
    let training_data = vec![
        (vec![0.0, 0.0], 0.0),
        (vec![0.0, 1.0], 1.0),
        (vec![1.0, 0.0], 1.0),
        (vec![1.0, 1.0], 1.0),
    ];

    for _ in 0..100 {
        perceptron.train(&training_data);
    }

    // Test the perceptron after training
    for &(ref inputs, target) in training_data.iter() {
        let output = perceptron.predict(inputs);
        println!(
            "Input: {:?}, Predicted: {}, Actual: {}",
            inputs, output, target
        );
    }
}
