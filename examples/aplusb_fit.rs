
use ::gnn::*;

use std::sync::Arc;

fn main() {
  let weights = vec![0.9714391640113373, 0.8637838115055051, 0.6905623020105852, 0.9651992923345951, 0.804984277326354, 0.3156783774880433];
  let biases = vec![0.5483480573474827, 0.6067177890976664, -0.6339702105297078];
  let mut nn = NN::new_prefit(Arc::new(vec![2, 2, 1]), weights, biases);

  let a = 30.5;
  let b = 12.5;

  println!("prefit guess: {a} + {b} = {:?}", 
    nn.forward([a, b], Output::None));
}