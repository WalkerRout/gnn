
use rand::prelude::*;

use gnn::prelude::*;

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
      let pred = nn.forward([a, b], Output::None);
      fitness += 1.0 / f64::abs(pred[0] - expected);
    }

    fitness / Self::FITNESS_ITERATIONS as f64
  }
}

fn main() {
  let mut gnn = GNNBuilder::new()
    .population(1500)
    .architecture([2, 2, 1])
    .build::<APlusB>();

  gnn.evolve_complete(2000);

  let a = 22.0;
  let b = 23.0;
  let avg = gnn.average_fitness();

  println!("final avg fitness: {}", avg);
  println!("most fit guess of fitness {}: {a} + {b} = {:?}",
    gnn.fitness(), gnn.forward([a, b], Output::None));
}