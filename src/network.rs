// use activation::Activation;
// use matrix::Matrix;

use super::{activation::Activation, matrix::Matrix};

pub struct Network<'a> {
    layers: Vec<usize>,
    weights: Vec<Matrix>,
    biases: Vec<Matrix>,
    data: Vec<Matrix>,
    learning_rate: f64,
    activation: Activation<'a>,
}

impl Network<'_> {
    pub fn new<'a>(
        layers: Vec<usize>,
        learning_rate: f64,
        activation: Activation<'a>,
    ) -> Network<'a> {
        let mut weights = vec![];
        let mut biases = vec![];

        for i in 0..layers.len() - 1 {
            weights.push(Matrix::random(layers[i + 1], layers[i]));
            biases.push(Matrix::random(layers[i + 1], 1));
        }

        Network {
            layers,
            weights,
            biases,
            data: vec![],
            learning_rate,
            activation,
        }
    }

    pub fn feed_forward(&mut self, inputs: Vec<f64>) -> Vec<f64> {
        if (inputs.len() != self.layers[0]) {
            panic!("Input vector must be the same size as the first layer");
        }

        let mut current = Matrix::transpose(Matrix::from(vec![inputs]));

        self.data = vec![current.clone()];

        for i in 0..self.layers.len() - 1 {
            current = Matrix::multiply(self.weights[i].clone(), current.clone());
            current = Matrix::add(current.clone(), self.biases[i].clone());
            current = current.map(self.activation.function);

            self.data.push(current.clone());
        }

        return Matrix::transpose(current).data[0].to_owned();
    }

    pub fn back_propogate(&mut self, outputs: Vec<f64>, expected: Vec<f64>) {
        if expected.len() != self.layers[self.layers.len() - 1] {
            panic!("Error in back_propogate");
        }

        let mut parsed = Matrix::from(vec![outputs]);
        parsed = Matrix::transpose(parsed.clone());

        let mut errors = Matrix::from(vec![expected]);
        errors = Matrix::transpose(errors.clone());
        errors = Matrix::subtract(errors.clone(), parsed.clone());

        let mut gradients = parsed.map(self.activation.derivative);

        for i in (0..self.layers.len() - 1).rev() {
            gradients = Matrix::dot_multiply(gradients.clone(), errors.clone());
            gradients = gradients.map(&|x| x * self.learning_rate);

            self.weights[i] = Matrix::add(
                self.weights[i].clone(),
                Matrix::multiply(gradients.clone(), Matrix::transpose(self.data[i].clone())),
            );

            self.biases[i] = Matrix::add(self.biases[i].clone(), gradients.clone());

            errors = Matrix::multiply(Matrix::transpose(self.weights[i].clone()), errors.clone());

            gradients = self.data[i].map(self.activation.derivative);
        }
    }

	pub fn train(&mut self, inputs: Vec<Vec<f64>>, targets: Vec<Vec<f64>>, epochs: u16) {
		for i in 1..=epochs {
			if epochs < 100 || i % (epochs / 100) == 0 {
				println!("Epoch {} of {}", i, epochs);
			}
			for j in 0..inputs.len() {
				let outputs = self.feed_forward(inputs[j].clone());
				self.back_propogate(outputs, targets[j].clone());
			}
		}
	}
}
