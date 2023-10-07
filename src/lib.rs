
pub mod nn;
pub mod gnn;

// visible in prelude::*;
pub mod prelude {
  pub use crate::nn::{NN, Output};
  pub use crate::gnn::{GNN, GNNBuilder, Optimizer};
}
