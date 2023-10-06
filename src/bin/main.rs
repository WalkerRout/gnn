
use rand::prelude::*;

use ::gnn::*;

#[derive(Debug, Default, Clone)]
struct APlusB;

impl APlusB {
  const FITNESS_ITERATIONS: usize = 15;
}

impl Optimizer for APlusB {
  fn fitness(&mut self, nn: &mut NN) -> f64 {
    let mut fitness = 0.0;
    let mut rng = rand::thread_rng();
    
    for _ in 0..Self::FITNESS_ITERATIONS {
      let a = rng.gen_range(0.0..100.0);
      let b = rng.gen_range(0.0..100.0);
      // A plus B!
      let expected = a + b;
      let pred = nn.forward(&[a, b], Output::None);
      fitness += 1.0 / f64::abs(pred[0] - expected);
    }

    fitness / Self::FITNESS_ITERATIONS as f64
  }
}

fn main() {
  let mut gnn = GNNBuilder::new()
    .population(1500)
    .architecture(&[2, 4, 4, 1])
    .build::<APlusB>();

  gnn.evolve_complete(2000);

  let a = 46.1;
  let b = 12.8;
  let avg = gnn.average_fitness();
  let (most_fit, fitness, _) = gnn.most_fit_mut();

  println!("final avg fitness: {}", avg);
  println!("most fit guess of fitness {}: {a} + {b} = {:?}",
    fitness, most_fit.forward(&[a, b], Output::None));

  //println!("Most fit weights: {:?}", most_fit.weights);
  //println!("Most fit biases: {:?}", most_fit.biases);
}