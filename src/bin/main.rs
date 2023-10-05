
use rand::prelude::*;

use ::gnn::*;

#[derive(Debug, Default, Clone)]
struct APlusB;

impl Optimizer for APlusB {
  fn fitness(&mut self, nn: &mut NN) -> f64 {
    let mut fitness = 0.0;
    let mut rng = rand::thread_rng();
    
    for i in 0..100 {
      let a = rng.gen_range(0.0..100.0);
      let b = rng.gen_range(0.0..100.0);
      // A plus B!
      let expected = a + b;
      let pred = nn.forward(&[a, b], Output::None);
      fitness += 1.0 / f64::abs(pred[0] - expected);
    }

    fitness / 100.0
  }
}

fn main() {
  let mut gnn = GNNBuilder::new()
    .population(200)
    .architecture(&[2, 12, 1])
    .build::<APlusB>();

  for i in 0..400 {
    gnn.evolve_generation();
    println!("avg fitness: {}", gnn.average_fitness());
  }
  println!("final avg fitness: {}", gnn.average_fitness());

  let a = 23.0;
  let b = 37.0;
  let (most_fit, fitness) = gnn.most_fit_mut();
  println!("most fit guess of fitness {}: {a} + {b} = {:?}", 
    fitness, most_fit.forward(&[a, b], Output::None));
}