use rand::{thread_rng, Rng};

#[derive(Debug, Clone)]
pub struct Matrix {
    pub rows: usize,
    pub columns: usize,
    pub data: Vec<Vec<f64>>,
}

impl Matrix {
    pub fn new(rows: usize, columns: usize) -> Matrix {
        Matrix {
            rows: rows,
            columns: columns,
            data: vec![vec![0.0; columns]; rows],
        }
    }

    pub fn random(rows: usize, columns: usize) -> Matrix {
        let mut rng = thread_rng();

        let mut res = Matrix::new(rows, columns);

        for i in 0..rows {
            for j in 0..columns {
                res.data[i][j] = rng.gen::<f64>() * 2.0 - 1.0;
            }
        }
        return res;
    }

    pub fn from(data: Vec<Vec<f64>>) -> Matrix {
        Matrix {
            rows: data.len(),
            columns: data[0].len(),
            data,
        }
    }

    pub fn add(first: Matrix, second: Matrix) -> Matrix {
        if (first.rows != second.rows) || (first.columns != second.columns) {
            panic!("Matrices must be the same size");
        }

        let mut res = Matrix::new(first.rows, first.columns);

        for i in 0..res.rows {
            for j in 0..res.columns {
                res.data[i][j] = first.data[i][j] + second.data[i][j];
            }
        }
        return res;
    }

    pub fn subtract(first: Matrix, second: Matrix) -> Matrix {
        if (first.rows != second.rows) || (first.columns != second.columns) {
            panic!("Matrices must be the same size");
        }

        let mut res = Matrix::new(first.rows, first.columns);

        for i in 0..res.rows {
            for j in 0..res.columns {
                res.data[i][j] = first.data[i][j] - second.data[i][j];
            }
        }
        return res;
    }

    pub fn scale(matrix: Matrix, scalar: f64) -> Matrix {
        let mut res = Matrix::new(matrix.rows, matrix.columns);

        for i in 0..res.rows {
            for j in 0..res.columns {
                res.data[i][j] = matrix.data[i][j] * scalar;
            }
        }
        return res;
    }

    pub fn multiply(first: Matrix, second: Matrix) -> Matrix {
        if (first.columns != second.rows) {
            panic!("Matrices must be compatible");
        }

        let mut res = Matrix::new(first.rows, second.columns);

        for i in 0..res.rows {
            for j in 0..res.columns {
                res.data[i][j] = Matrix::vec_multi_add(
                    first.data[i].clone(),
                    second.data.iter().map(|row| row[j]).collect(),
                )
            }
        }

        return res;
    }

    pub fn dot_multiply(first: Matrix, second: Matrix) -> Matrix {
        if (first.columns != second.columns) || (first.rows != second.rows) {
            panic!("Dot Matrices must be compatible");
        }

        let mut res = Matrix::new(first.rows, first.columns);
		
        for i in 0..res.rows {
            for j in 0..res.columns {
                res.data[i][j] = first.data[i][j] * second.data[i][j];
			}
        }

        return res;
    }

    pub fn transpose(matrix: Matrix) -> Matrix {
        let mut res = Matrix::new(matrix.columns, matrix.rows);

        for i in 0..res.rows {
            for j in 0..res.columns {
                res.data[i][j] = matrix.data[j][i];
            }
        }

        return res;
    }

    fn vec_multi_add(first: Vec<f64>, second: Vec<f64>) -> f64 {
        let mut res = 0.0;
        for i in 0..first.len() {
            res += first[i] * second[i];
        }
        return res;
    }

    pub fn map(&self, function: &dyn Fn(f64) -> f64) -> Matrix {
        Matrix::from(
            self.data
                .clone()
                .into_iter()
                .map(|row| row.into_iter().map(|value| function(value)).collect())
                .collect(),
        )
    }

    pub fn print(&self) {
        for i in 0..self.rows {
            for j in 0..self.columns {
                print!("{} ", self.data[i][j]);
            }
            println!();
        }
    }
}
