
use std::sync::Arc;

#[derive(Debug, Clone)]
pub enum Output {
  Softmax,
  //Sigmoid,
  None
}

#[derive(Debug, Clone)]
pub struct NN {
  // input is 1x4, h1-w is 10x4 (*input is 1x10), h2-w is 12x10 (*h1-w is 1x12), output-w is 2x12 (*h2-w is 1x2)
  pub architecture: Arc<Vec<usize>>, // [4, 10, 12, 2]
  pub input_count: usize, // == architecture[0]

  // (curr, prev)
  pub weights_dimensions: Vec<(usize, usize)>, // [(10, 4), (12, 10), (2, 12)]
  pub weights: Vec<f64>, // all weights in network

  pub biases_dimensions: Vec<(usize, usize)>, // [(10, 1), (12, 1), (2, 1)]
  pub biases: Vec<f64> // all biases in network
}

impl NN {
  pub fn new(arch: Arc<Vec<usize>>) -> Self {
    assert!(arch.len() > 1);

    let mut prev = arch[0];
    let input_count = arch[0];

    let weights_dimensions: Vec<_> = arch
      .iter()
      .skip(1)
      .map(|curr| {
        let res = (*curr, prev);
        prev = *curr;
        res
      })
      .collect();
    let biases_dimensions: Vec<_> = arch
      .iter()
      .skip(1)
      .map(|curr| {
        (*curr, 1)
      })
      .collect();

    let weights_count = weights_dimensions
      .iter()
      .fold(0, |acc, &(x, y)| acc + x*y);
    let biases_count = biases_dimensions
      .iter()
      .fold(0, |acc, &(x, y)| acc + x*y);

    let mut weights: Vec<_> = Vec::with_capacity(weights_count);
    for _ in 0..weights.capacity() {
      weights.push((rand::random::<f64>() - 0.5) * 2.0);
    }
    let mut biases: Vec<_> = Vec::with_capacity(biases_count);
    for _ in 0..biases.capacity() {
      biases.push((rand::random::<f64>() - 0.5) * 2.0);
    }

    NN {
      architecture: arch,
      input_count,
      weights_dimensions,
      weights,
      biases_dimensions,
      biases
    }
  }

  pub fn forward<A: AsRef<[f64]>>(&mut self, input: A, output: Output) -> Vec<f64> {
    let mut current_input = input.as_ref().to_owned();

    for i in 0..self.architecture.len() - 1 {  // subtract 1 to exclude input layer
      let layer_weights = self.layer_weights(i);
      let layer_biases = self.layer_biases(i);
      let (current_neuron_count, current_input_count) = self.weights_dimensions[i];

      // hotspot; gonna be a LOT of bounds checking unless we use unsafe{}... good thing this is safe...
      // calculate a = W.T*x + b
      let mut neurons = vec![0.0; current_neuron_count];
      for j in 0..current_neuron_count {
        let mut weighted_sum = 0.0;
        for k in 0..current_input_count {
          unsafe {
            weighted_sum += *layer_weights.get_unchecked(j * current_input_count + k) 
              * *current_input.get_unchecked(k);
          }
        }
        unsafe {
          *neurons.get_unchecked_mut(j) = weighted_sum + *layer_biases.get_unchecked(j);
        }
      }

      // apply the relu activation function ϕ(a), skipping output layer
      if i != self.architecture.len() - 2 {
        neurons = neurons
          .into_iter()
          .map(|x| x.max(0.0))
          .collect();
      }

      // update current input for next layer
      current_input = neurons;
    }

    match output {
      Output::Softmax => {
        assert!(self.architecture[self.architecture.len()-1] > 1);
        self.softmax(current_input)
      },
      //Output::Sigmoid => {
      //  assert!(self.architecture[self.architecture.len()-1] == 1);
      //  self.sigmoid(current_input)
      //},
      Output::None => current_input
    }
  }

  fn softmax(&self, input: Vec<f64>) -> Vec<f64> {
    let max_val = input
      .iter()
      .fold(0.0, |acc, x|  f64::max(acc, *x));
    let exp_sum: f64 = input
      .iter()
      .map(|&x| (x - max_val).exp())
      .sum();

    input
      .into_iter()
      .map(|x| (x - max_val).exp() / exp_sum)
      .collect()
  }

  #[inline]
  fn layer_weights(&self, index: usize) -> &[f64] {
    let ptr = self.weights_dimensions
      .iter()
      .take(index)
      .fold(0, |acc, &(x, y)| acc + x*y);
    let (rows, cols) = self.weights_dimensions[index];

    &self.weights[ptr..(ptr + rows*cols)]
  }

  #[inline]
  fn layer_biases(&self, index: usize) -> &[f64] {
    let ptr = self.biases_dimensions
      .iter()
      .take(index)
      .fold(0, |acc, &(x, y)| acc + x*y);
    let (rows, cols) = self.biases_dimensions[index];

    &self.biases[ptr..(ptr + rows*cols)]
  }
}

#[cfg(test)]
mod test_nn {
  use super::*;

  #[test]
  fn test_nn_new() {
    let arch = Arc::new(vec![2, 8, 8, 4]);
    let nn = NN::new(arch.clone());

    assert_eq!(nn.architecture, arch);
    assert_eq!(nn.weights_dimensions, &[(8, 2), (8, 8), (4, 8)]);
    assert_eq!(nn.weights.len(), 8*2 + 8*8 + 4*8);
    assert_eq!(nn.biases_dimensions, &[(8, 1), (8, 1), (4, 1)]);
    assert_eq!(nn.biases.len(), 8*1 + 8*1 + 4*1);
  }

  #[test]
  fn test_nn_forward_softmax() {
    let mut nn = NN::new(Arc::new(vec![6, 8, 4]));
    let inputs = &[0.0; 6];

    let outputs = nn.forward(inputs, Output::Softmax);
    assert!(outputs.iter().sum::<f64>() - 1.0 < f64::EPSILON+0.00001);
    assert_eq!(outputs.len(), 4);
  }

    #[test]
  fn test_nn_forward_none() {
    let mut nn = NN::new(Arc::new(vec![6, 8, 4]));
    let inputs = &[0.0; 6];

    let outputs = nn.forward(inputs, Output::None);
    assert_eq!(outputs.len(), 4);
  }

  #[test]
  fn test_nn_layer_weights() {
    let nn = NN::new(Arc::new(vec![2, 4, 8, 6, 2]));

    assert_eq!(nn.layer_weights(0), &nn.weights[0..4*2]);
    assert_eq!(nn.layer_weights(1), &nn.weights[4*2..(4*2 + 8*4)]);
    assert_eq!(nn.layer_weights(2), &nn.weights[(4*2 + 8*4)..(4*2 + 8*4 + 6*8)]);
    assert_eq!(nn.layer_weights(3), &nn.weights[(4*2 + 8*4 + 6*8)..(4*2 + 8*4 + 6*8 + 2*6)]);
  }

  #[test]
  fn test_nn_layer_biases() {
    let nn = NN::new(Arc::new(vec![2, 4, 8, 6, 2]));

    assert_eq!(nn.layer_biases(0), &nn.biases[0..4*1]);
    assert_eq!(nn.layer_biases(1), &nn.biases[4*1..(4*1 + 8*1)]);
    assert_eq!(nn.layer_biases(2), &nn.biases[(4*1 + 8*1)..(4*1 + 8*1 + 6*1)]);
    assert_eq!(nn.layer_biases(3), &nn.biases[(4*1 + 8*1 + 6*1)..(4*1 + 8*1 + 6*1 + 2*1)]);
  }
}