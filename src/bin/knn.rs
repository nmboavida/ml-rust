use std::fs::File;

use anyhow::Result;
use csv::Reader;

use ml_rust::knn::{knn, Point};

// Main function to test KNN
fn main() -> Result<()> {
    // Read file
    let file_path = "datasets/knn_dataset.csv";
    let file = File::open(file_path)?;
    let mut reader = Reader::from_reader(file);

    // Collect all records into a Vec<Point>
    let dataset: Vec<Point> = reader.deserialize().collect::<Result<Vec<_>, _>>()?;
    println!("b");

    // Test the KNN function with a new point
    let test_point = Point {
        x: 2.5,
        y: 3.0,
        class: String::new(),
    };

    println!("c");

    let predicted_class = knn(&test_point, &dataset, 5);
    println!("Predicted class: {}", predicted_class);

    Ok(())
}
