
use rand::prelude::*;

use crate::nn::NN;

use std::iter;
use std::sync::Arc;
use std::default::Default;

pub trait Optimizer: Default {
  /// Use output of network forward pass to determine fitness:
  /// let output: Vec<f64> = nn.forward(&[...], Output::...);
  fn fitness(&mut self, nn: &mut NN) -> f64;
}

pub struct GNN<P: Optimizer> {
  architecture: Arc<Vec<usize>>, // population architecture
  /// Tandem vectors
  networks: Vec<NN>,
  fitnesses: Vec<f64>,
  population: Vec<P>
}

impl<P: Optimizer> GNN<P> {
  pub fn new<A: AsRef<[usize]>>(population_count: usize, arch: A) -> Self {
    assert!(population_count >= 10, "Population must be greater than or equal to 10");
    assert!(population_count % 2 == 0, "Population must be an even number");

    let architecture = Arc::new(Vec::from(arch.as_ref()));
    let networks = iter::repeat_with(|| NN::new(architecture.clone()))
      .take(population_count)
      .collect();
    let fitnesses = iter::repeat_with(f64::default)
      .take(population_count)
      .collect();
    let population = iter::repeat_with(P::default)
      .take(population_count)
      .collect();

    GNN {
      architecture,
      networks,
      fitnesses,
      population
    }
  }

  #[inline]
  pub fn evolve_generation(&mut self) {
    self.fit();
    let genes = self.gene_pool();
    self.crossover(genes);
    self.mutate();
  }

  #[inline]
  pub fn evolve_complete(&mut self, epochs: usize) {
    for _ in 0..epochs {
      self.evolve_generation();
    }
  }

  #[inline]
  pub fn most_fit(&self) -> (&NN, f64) {
    let indices_by_fitness = self.indices_by_fitness();
    (&self.networks[indices_by_fitness[0]], self.fitnesses[indices_by_fitness[0]])
  }

  #[inline]
  pub fn most_fit_mut(&mut self) -> (&mut NN, f64) {
    let indices_by_fitness = self.indices_by_fitness();
    (&mut self.networks[indices_by_fitness[0]], self.fitnesses[indices_by_fitness[0]])
  }

  #[inline]
  pub fn average_fitness(&self) -> f64 {
    self.fitnesses.iter().sum::<f64>() / self.fitnesses.len() as f64
  }

  #[inline]
  fn fit(&mut self) -> f64 {
    assert_eq!(self.networks.len(), self.fitnesses.len());
    assert_eq!(self.fitnesses.len(), self.population.len());

    self.fitnesses = self.population
      .iter_mut()
      .zip(self.networks.iter_mut())
      .map(|(p, nn)| p.fitness(nn))
      .collect();

    self.average_fitness()
  }

  #[inline]
  fn gene_pool(&self) -> Vec<usize> {
    let indices_by_fitness = self.indices_by_fitness();
    let elites_count = self.elites_count();

    indices_by_fitness
      .into_iter()
      .take(elites_count)
      .collect()
  }

  fn crossover(&mut self, elites: Vec<usize>) {
    // |A|AA|AA| -> |A|BB|AA|
    // |B|BB|BB|    |B|AA|BB|
    let mut rng = rand::thread_rng();
    // same architecture for all, just select first one
    let weights_count = self.networks[0].weights.len();
    let biases_count = self.networks[0].biases.len();
    let elites_count = self.elites_count();

    // eventually add some random crap networks for variability
    let elites: Vec<(Vec<f64>, Vec<f64>)> = elites
      .into_iter()
      .map(|i| (
        self.networks[i].weights.clone(),
        self.networks[i].biases.clone()
      ))
      .collect();

    // room for so many optimizations here...
    self.networks
      .chunks_exact_mut(2)
      .skip(elites_count/2)
      .for_each(|nets| {
        // select 2 random elites (.0 is weights, .1 is biases)
        let elite_a = elites.choose(&mut rng).expect("Error - elites empty..");
        let elite_b = elites.choose(&mut rng).expect("Error - elites empty..");

        // |A|BB|AA|
        let mut nn_a_weights = Vec::with_capacity(weights_count);
        let mut nn_a_biases  = Vec::with_capacity(biases_count);
        // |B|AA|BB|
        let mut nn_b_weights = Vec::with_capacity(weights_count);
        let mut nn_b_biases  = Vec::with_capacity(biases_count);

        // nn_a_weights
        let lower = rng.gen_range(0..weights_count / 2);
        let upper = rng.gen_range(weights_count / 2..weights_count);
        nn_a_weights.extend_from_slice(&elite_a.0[0..lower]);
        nn_a_weights.extend_from_slice(&elite_b.0[lower..upper]);
        nn_a_weights.extend_from_slice(&elite_a.0[upper..weights_count]);

        // nn_a_biases
        let lower = rng.gen_range(0..biases_count / 2);
        let upper = rng.gen_range(biases_count / 2..biases_count);
        nn_a_biases.extend_from_slice(&elite_a.1[0..lower]);
        nn_a_biases.extend_from_slice(&elite_b.1[lower..upper]);
        nn_a_biases.extend_from_slice(&elite_a.1[upper..biases_count]);

        // nn_b_weights
        let lower = rng.gen_range(0..weights_count / 2);
        let upper = rng.gen_range(weights_count / 2..weights_count);
        nn_b_weights.extend_from_slice(&elite_b.0[0..lower]);
        nn_b_weights.extend_from_slice(&elite_a.0[lower..upper]);
        nn_b_weights.extend_from_slice(&elite_b.0[upper..weights_count]);

        // nn_b_biases
        let lower = rng.gen_range(0..biases_count / 2);
        let upper = rng.gen_range(biases_count / 2..biases_count);
        nn_b_biases.extend_from_slice(&elite_b.1[0..lower]);
        nn_b_biases.extend_from_slice(&elite_a.1[lower..upper]);
        nn_b_biases.extend_from_slice(&elite_b.1[upper..biases_count]);

        let nn_genes = [(nn_a_weights, nn_a_biases), (nn_b_weights, nn_b_biases)];
        
        nets
          .iter_mut()
          .zip(IntoIterator::into_iter(nn_genes))
          .for_each(|(nn, (weights_gene, biases_gene))| {
            nn.weights = weights_gene;
            nn.biases  = biases_gene;
          });
      });

    // copy over elites
    self.networks
      .iter_mut()
      .zip(elites.into_iter())
      .take(elites_count)
      .for_each(|(nn, (weights_gene, biases_gene))| {
        nn.weights = weights_gene;
        nn.biases  = biases_gene;
      });
  }

  fn mutate(&mut self) {
    let mutation_rate = 0.35;
    let weights_mutation_rate = 0.12;
    let biases_mutation_rate  = 0.09;

    let mut rng = rand::thread_rng();

    // select random networks in population to mutate
    self.networks
      .iter_mut()
      .filter(|_| rand::random::<f64>() < mutation_rate)
      .for_each(|nn| {
        // select random weights to mutate
        nn.weights
          .iter_mut()
          .filter(|_| rand::random::<f64>() < weights_mutation_rate)
          .for_each(|w| {
            *w += rng.gen_range(-0.05..=0.05);
          });
        // select random biases to mutate
        nn.biases
          .iter_mut()
          .filter(|_| rand::random::<f64>() < biases_mutation_rate)
          .for_each(|b| {
            *b += rng.gen_range(-0.05..=0.05);
          });
      });
  }

  fn indices_by_fitness(&self) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..self.fitnesses.len()).collect();
    
    // safe; indices always fall within size of self.fitnesses
    unsafe {
      indices.sort_by(|&a, &b| { 
        self.fitnesses.get_unchecked(b)
          .partial_cmp(&self.fitnesses.get_unchecked(a)).unwrap() 
      });
    }

    indices
  }

  #[inline]
  fn elites_count(&self) -> usize {
    (0.3 * self.networks.len() as f64).ceil() as usize
  }
}

#[cfg(test)]
mod test_gnn {
  use super::*;

  #[derive(Default)]
  struct OneOptimizer;

  impl Optimizer for OneOptimizer {
    fn fitness(&mut self, _: &mut NN) -> f64 {
      1.0
    }
  }

  #[test]
  fn test_gnn_new() {
    let arch = &[2, 6, 2];
    let gnn: GNN<OneOptimizer> = GNN::new(1000, arch);

    assert_eq!(*gnn.architecture, arch);
    assert_eq!(gnn.networks.len(), 1000);
    assert_eq!(gnn.fitnesses.len(), 1000);
    assert_eq!(gnn.population.len(), 1000);
    assert_eq!(gnn.fitnesses, vec![0.0; 1000]);
  }

  #[test]
  fn test_gnn_fit() {
    let mut gnn: GNN<OneOptimizer> = GNN::new(1000, &[1, 6, 2]);
    let _avg_fitness = gnn.fit();
    
    assert_ne!(gnn.fitnesses, vec![0.0; 1000]);
  }

  #[test]
  fn test_gnn_gene_pool() {
    let gnn: GNN<OneOptimizer> = GNN::new(1000, &[1, 6, 2]);
    let genes = gnn.gene_pool();

    // 30% == 3/10
    assert_eq!(genes.len(), 1000 * 3/10);
  }

  #[test]
  fn test_gnn_crossover() {
    let mut gnn: GNN<OneOptimizer> = GNN::new(10, &[1, 6, 2]);
    let prev_weights = gnn.networks[0].weights.clone();
    gnn.crossover(vec![2, 4, 8]); //0..=9
    let post_weights = gnn.networks[0].weights.clone();

    assert_ne!(prev_weights, post_weights);
  }

  #[test]
  fn test_gnn_mutate() {
    let mut gnn: GNN<OneOptimizer> = GNN::new(10, &[1, 6, 2]);
    let nets = gnn.networks.clone();
    let mut diff = false;

    for i in 0..nets.len() {
      let prev_weights = nets[i].weights.clone();
      gnn.mutate();
      let post_weights = gnn.networks[i].weights.clone();

      diff = prev_weights != post_weights;
      if diff { break; }
    }

    assert!(diff);
  }
}

pub struct GNNBuilder {
  population: usize,
  architecture: Option<Vec<usize>>
}

impl GNNBuilder {
  pub fn new() -> Self {
    GNNBuilder {
      population: 500,
      architecture: None,
    }
  }

  pub fn population(mut self, population: usize) -> Self {
    self.population = population;
    self
  }

  pub fn architecture<A: AsRef<[usize]>>(mut self, architecture: A) -> Self {
    self.architecture = Some(architecture.as_ref().to_owned());
    self
  }

  pub fn build<P: Optimizer>(self) -> GNN<P> {
    GNN::new(self.population, self.architecture.expect("Error - must set architecture for GNN"))
  }
}

#[cfg(test)]
mod test_gnnbuilder {
  use super::*;
  
  #[derive(Default)]
  struct OneOptimizer;

  impl Optimizer for OneOptimizer {
    fn fitness(&mut self, _: &mut NN) -> f64 {
      1.0
    }
  }

  #[test]
  fn test_gnnbuilder_new() {
    let builder = GNNBuilder::new();

    assert_eq!(builder.population, 500);
    assert_eq!(builder.architecture, None);
  }

  #[test]
  fn test_gnnbuilder_population() {
    let builder = GNNBuilder::new()
      .population(1000);

    assert_eq!(builder.population, 1000);
  }

  #[test]
  fn test_gnnbuilder_architecture() {
    let builder = GNNBuilder::new()
      .architecture(&[2, 6, 1]);

    assert_eq!(builder.architecture, Some(vec![2, 6, 1]));
  }

  #[test]
  fn test_gnnbuilder_build() {
    let gnn = GNNBuilder::new()
      .population(300)
      .architecture(&[2, 12, 1])
      .build::<OneOptimizer>();

    assert_eq!(gnn.population.len(), 300);
    assert_eq!(*gnn.architecture, &[2, 12, 1]);
  }
}
