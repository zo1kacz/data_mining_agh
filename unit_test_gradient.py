import unittest
import numpy as np

class TestLinearRegressionGradient(unittest.TestCase):
    def setUp(self):
        self.X_train = np.array([[1], [2], [3], [4], [5]])
        self.y_train = np.array([2, 4, 6, 8, 10])  # y = 2x
        self.model = LinearRegression_gradient(learning_rate=0.01, iterations=1000)

    def test_initialization(self):
        self.assertIsNone(self.model.weights, "Initial weights should be None.")
        self.assertIsNone(self.model.bias, "Initial bias should be None.")
        self.assertEqual(self.model.learning_rate, 0.01, "Learning rate not correctly initialized")
        self.assertEqual(self.model.iterations, 1000, "Number of iterations not correctly initialized")


    def test_fit(self):
        try:
            self.model.fit(self.X_train, self.y_train)
            self.assertIsNotNone(self.model.weights, "Weights should be initialized after fitting")
            self.assertIsNotNone(self.model.bias, "Bias should be initialized after fitting")
        except Exception as e:
            self.fail(f"Model training failed with exception: {e}")

    def test_predict(self):
        self.model.fit(self.X_train, self.y_train)
        predictions = self.model.predict(self.X_train)

        self.assertEqual(predictions.shape, self.y_train.shape, "Predicted shape does not match target shape.")

    def test_fit_predict_accuracy(self):
        self.model.fit(self.X_train, self.y_train)
        predictions = self.model.predict(self.X_train)
        mse = np.mean((predictions - self.y_train) ** 2)
        self.assertLess(mse, 0.1, "Model MSE is too high")

        for pred, true in zip(predictions, self.y_train):
            self.assertAlmostEqual(pred, true, delta=1, msg=f"Prediction {pred} is not close to true value {true}")


if __name__ == '__main__':
    unittest.main(argv=[''], verbosity=2, exit=False)
