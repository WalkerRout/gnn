
use rand::prelude::*;

use ::gnn::*;

#[derive(Debug, Default, Clone)]
struct APlusB;

impl Optimizer for APlusB {
  fn fitness(&mut self, nn: &mut NN) -> f64 {
    let mut fitness = 0.0;
    let mut rng = rand::thread_rng();
    
    for _ in 0..10 {
      let a = rng.gen_range(0.0..100.0);
      let b = rng.gen_range(0.0..100.0);
      // A plus B!
      let expected = a + b;
      let pred = nn.forward(&[a, b], Output::None);
      fitness += 1.0 / f64::abs(pred[0] - expected);
    }

    fitness / 10.0
  }
}

fn main() {
  let mut gnn = GNNBuilder::new()
    .population(3000)
    .architecture(&[2, 3, 4, 1])
    .build::<APlusB>();

  gnn.evolve_complete(2250);

  let a = 12.5;
  let b = 12.5;
  let avg = gnn.average_fitness();
  let (most_fit, fitness, _) = gnn.most_fit_mut();

  println!("final avg fitness: {}", avg);
  println!("most fit guess of fitness {}: {a} + {b} = {:?}", 
    fitness, most_fit.forward(&[a, b], Output::None));
}