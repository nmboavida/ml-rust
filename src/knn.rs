use serde::Deserialize;
use std::collections::HashMap;

pub type Class = String;

// Define a struct for a data point
#[derive(Debug, Deserialize)]
pub struct Point {
    pub x: f64,
    pub y: f64,
    pub class: Class,
}

/// The distance between the new data point $x$ and each point in the dataset is
/// usually calculated using a distance metric such as Euclidean distance:
/// 
/// $d(p^*, q_i) = \sqrt{\sum_{d=1}^{n} (p^*_{d} - q_{i,d})^2}$
/// 
/// Where n is the number of dimensions, i.e. the number of features per 
/// data point, $p^*_{d}$ is the coordinate of the $p^*$ with respect to the
/// dimension $d$, and $q_{i,d}$ is the $d$-th feature of the $i$-th neighbor.
#[rustfmt::skip]
pub fn euclidian_distance(p_star: &Point, q_i: &Point) -> f64 {
    (
        (p_star.x - q_i.x).powi(2) + (p_star.y - q_i.y).powi(2)
    ).sqrt()
}

pub fn knn(new_point: &Point, data: &[Point], k: usize) -> Class {
    println!("Data: {:?}", data);

    // Compute distances - note that this step can be parallelized
    let mut distances: Vec<(&Point, f64)> = data
        .iter()
        .map(|point| (point, euclidian_distance(new_point, point)))
        .collect();

    println!("Distances: {:?}", distances);

    // Sort distances and take the k-nearest neighbors
    // a.partial_cmp(&b) --> Checks if a > b
    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    println!("Sorted Distances: {:?}", distances);

    // Get K nearest neighbors
    let neighbors = &distances[..k];
    println!("K Neightbors: {:?}", neighbors);

    // Count the frequency in the neighboring classes
    // let class_frenquecy = HasMap::new();

    let mut class_count: HashMap<&Class, i32> = HashMap::new();
    for (point, _) in neighbors {
        *class_count.entry(&point.class).or_insert(0) += 1;
    }

    // Return the most common class
    class_count
        .into_iter()
        .max_by(|a, b| a.1.cmp(&b.1))
        .map(|(class, _)| class.to_string())
        .unwrap()
}
