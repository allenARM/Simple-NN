use activation::TANH;
use network::Network;

pub mod activation;
pub mod matrix;
pub mod network;

fn main() {
	let inputs = vec![vec![-3.0], vec![72.0], vec![71.0], vec![40.0], vec![105.0], vec![-143.0], vec![-84.0], vec![-97.0], vec![-112.0], vec![-33.0], vec![92.0], vec![-79.0], vec![63.0], vec![-45.0], vec![-148.0], vec![-106.0], vec![117.0], vec![-14.0], vec![-103.0], vec![83.0]];

	let inputs_divided_by_1000: Vec<Vec<f64>> = inputs.iter().map(|input| {
		input.iter().map(|&val| val / 1000.0).collect()
	}).collect();
	
	let targets = inputs.iter().map(|f| vec![((f[0] - 32.0) * 5.0/9.0)]).collect::<Vec<Vec<f64>>>();

	let targets_divided_by_1000: Vec<Vec<f64>> = targets.iter().map(|input| {
		input.iter().map(|&val| val / 1000.0).collect()
	}).collect();
	
	let mut network = Network::new(vec![1, 10, 10, 1], 0.1, TANH);

	network.train(inputs_divided_by_1000, targets_divided_by_1000, 1000);

	println!("{}\tf is about {:?}", 45, network.feed_forward(vec![45.0/1000.0])[0] * 1000.0);
	println!("{}\tf is about {:?}", 120, network.feed_forward(vec![120.0/1000.0])[0] * 1000.0);
	println!("{}\tf is about {:?}", 70, network.feed_forward(vec![70.0/1000.0])[0] * 1000.0);
	println!("{}\tf is about {:?}", -2, network.feed_forward(vec![-2.0/1000.0])[0] * 1000.0);
	println!("{}\tf is about {:?}", -70, network.feed_forward(vec![-70.0/1000.0])[0] * 1000.0);
}