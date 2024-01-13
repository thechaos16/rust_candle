use candle_core::{Device, Result, Tensor};

mod mnist_inference;
use mnist_inference::mnist_inference::run_mnist;


fn main() -> Result<()> {
    run_mnist()
}