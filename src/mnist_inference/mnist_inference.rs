use candle_core::{Device, Result, Tensor};
use candle_nn;

struct Model {
    first: Tensor,
    second: Tensor,
}

impl Model {
    fn forward(&self, image: &Tensor) -> Result<Tensor> {
        let x = image.matmul(&self.first)?;
        let x = x.relu()?;
        x.matmul(&self.second)
    }
}

pub fn run_mnist() -> Result<()> {
    // Use Device::new_cuda(0)?; to use the GPU.
    let device = Device::Cpu;

    let first = Tensor::randn(0f32, 1.0, (784, 100), &device)?;
    let second = Tensor::randn(0f32, 1.0, (100, 10), &device)?;
    let model = Model { first, second };

    let dummy_image = Tensor::randn(0f32, 1.0, (1, 784), &device)?;

    let digit = model.forward(&dummy_image)?;
    println!("Digit {digit:?} digit");
    println!("{:?}", digit.to_vec2::<f32>());
    println!("{:?}", candle_nn::ops::softmax(&digit, 1)?.to_vec2::<f32>());
    let softmax_res = candle_nn::ops::softmax(&digit, 1)?;
    println!("{:?}", softmax_res.argmax(1));
    Ok(())
}
