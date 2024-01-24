use candle_core::Result;

mod mnist_inference;
use mnist_inference::mnist_inference::run_mnist;

//mod hf_models;
//use hf_models::run_hf_models::download_and_load_model;


fn main() -> Result<()> {
    run_mnist()
    //download_and_load_model("bert-base-uncased".to_string())
}