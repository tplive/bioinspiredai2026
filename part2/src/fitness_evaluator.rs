use linfa::prelude::*;
use linfa_linear::LinearRegression;
use ndarray::{Array1, Array2};
use crate::chromosome::Chromosome;

pub struct FitnessEvaluator {
    x_train: Array2<f64>,
    y_train: Array1<f64>,
    x_text: Array2<f64>,
    y_text: Array1<f64>,
}

impl FitnessEvaluator {
    pub fn new(
        x_train: Array2<f64>,
        y_train: Array1<f64>,
        x_text: Array2<f64>,
        y_text: Array1<f64>,
    ) -> Self {
        Self { x_train, y_train, x_text, y_text }
    }

    pub fn evaluate(&self, chromosome: &Chromosome) -> f64 {
        let genes_in_the_sack: Vec<usize> = chromosome.genes
            .iter()
            .enumerate()
            .filter(|&selected| *selected.1)
            .map(|(i, _)| i)
            .collect();
        
        if genes_in_the_sack.is_empty() {
            return f64::MAX;
        }

        let x_train_selected = self.select_columns(&self.x_train, &genes_in_the_sack);
        let x_test_selected = self.select_columns(&self.x_text, &genes_in_the_sack);

        let dataset = Dataset::new(x_train_selected, self.y_train.clone());
        let model = LinearRegression::default().fit(&dataset).unwrap();

        let pred = model.predict(&x_test_selected);

        self.rmse(&pred, &self.y_text)
    }

    fn select_columns(&self, data_array: &Array2<f64>, indexes: &[usize]) -> Array2<f64> {
        let rows = data_array.nrows();
        let cols = indexes.len();
        let mut result = Array2::zeros((rows, cols));

        // 
        for (new_col, &old_col) in indexes.iter().enumerate() {
            for row in 0..rows {
                result[[row, new_col]] = data_array[[row, old_col]];
            }
        }

        result
    }

    // From 
    fn rmse(&self, predictions: &Array1<f64>, actual: &Array1<f64>) -> f64 {
        let n = predictions.len() as f64;
        let sum_squared_error: f64 = predictions.iter()
            .zip(actual.iter())
            .map(|(pred, actual)| (pred - actual).powi(2))
            .sum();

        (sum_squared_error / n).sqrt()
    }
}