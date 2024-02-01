import os
import sys
import argparse
import time
import itertools
import numpy as np
import pandas as pd


class KnnClassifier:
    def __init__(self, k: int, p: float):
        """
        Constructor for the KnnClassifier.

        :param k: Number of nearest neighbors to use.
        :param p: p parameter for Minkowski distance calculation.
        """
        self.k = k
        self.p = p

        self.ids = (319044434, 314779166)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        This method trains a k-NN classifier on a given training set X with label set y.

        :param X: A 2-dimensional numpy array of m rows and d columns. It is guaranteed that m >= 1 and d >= 1.
            Array datatype is guaranteed to be np.float32.
        :param y: A 1-dimensional numpy array of m rows. it is guaranteed to match X's rows in length (|m_x| == |m_y|).
            Array datatype is guaranteed to be np.uint8.
        """
        self.x_train = X
        self.y_train = y

    def minkowski_distance(self, x1, x2, axis=None):
        ''' This function calculates the Minkowski distance between two vectors or matrices.
            if axis=None, it calculates the distance between two vectors,
            if axis=2, it calculates the distance between two matrices (each row of the first matrix from each row of the second matrix). '''
        return np.power(np.sum(np.power(np.abs(x1 - x2), self.p), axis=axis), 1/self.p)


    def break_tie_between_labels(self, col, tied_labels):
        ''' This function breaks the tie between the labels of the k nearest neighbors.
            It is called recursively until the tie is broken.
            Also it checks for distance equality between more than one test point and chooses the label according to lexicographic order. '''
        closest_index = self.sorted_indices[self.i][col]
        closest_label = self.y_train[closest_index]

        if closest_label in tied_labels:
            closest_points = np.where(self.distances[self.i] == self.distances[self.i][closest_index])  # Indices of the points who are nearest to the test point with equal distance.
            closest_labels = self.y_train[closest_points]  # The labels of the points who are nearest to the test point with equal distance.
            closest_tied_labels = np.intersect1d(closest_labels, tied_labels)  # Only keeping the labels with maximum count.
            return np.min(closest_tied_labels)  # Choose a label according to lexicographic order.
        else:
            return self.break_tie_between_labels(col+1, tied_labels)  # If the label of the nearest neighbor is not in the tied labels, move to the next neighbor.


    def tie_breaking(self, labels):
        unique, counts = np.unique(labels, return_counts=True)  # Calculate the count of each label in the k nearest neighbors.
        max_count_of_label = np.max(counts)
        max_count_of_label_idx = np.where(counts == max_count_of_label)
        tied_labels = unique[max_count_of_label_idx]  # Only taking the labels with maximum count.
        pred_label = self.break_tie_between_labels(0, tied_labels) if len(tied_labels) > 1 else tied_labels[0]

        self.i += 1
        return pred_label


    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        This method predicts the y labels of a given dataset X, based on a previous training of the model.
        It is mandatory to call KnnClassifier.fit before calling this method.

        :param X: A 2-dimensional numpy array of m rows and d columns. It is guaranteed that m >= 1 and d >= 1.
            Array datatype is guaranteed to be np.float32.
        :return: A 1-dimensional numpy array of m rows. Should be of datatype np.uint8.
        """
        self.distances = self.minkowski_distance(self.x_train, X[:, np.newaxis], axis=2)
        # Distances is a 2d matrix containing the distances of each test point (in the rows) from each train point (in the columns).
        self.sorted_indices = np.argsort(self.distances, axis=1) # Sorts the indices of the distances matrix column in each row (test points), in ascending order.
        k_nearest_indices = self.sorted_indices[:, :self.k] # Takes the k nearest neighbors for each test point.
        k_nearest_labels = self.y_train[k_nearest_indices]
        self.i = 0  # Saves the row number of the test point we are currently predicting.
        predictions = np.apply_along_axis(self.tie_breaking, axis=1, arr=k_nearest_labels)  # TODO: Add another tie-breaker.
        return predictions        

       

def main():

    print("*" * 20)
    print("Started HW1_ID1_ID2.py")
    # Parsing script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', type=str, help='Input csv file path')
    parser.add_argument('k', type=int, help='k parameter')
    parser.add_argument('p', type=float, help='p parameter')
    args = parser.parse_args()

    print("Processed input arguments:")
    print(f"csv = {args.csv}, k = {args.k}, p = {args.p}")

    print("Initiating KnnClassifier")
    model = KnnClassifier(k=args.k, p=args.p)
    print(f"Student IDs: {model.ids}")
    print(f"Loading data from {args.csv}...")
    data = pd.read_csv(args.csv, header=None)
    print(f"Loaded {data.shape[0]} rows and {data.shape[1]} columns")
    X = data[data.columns[:-1]].values.astype(np.float32)
    y = pd.factorize(data[data.columns[-1]])[0].astype(np.uint8)

    print("Fitting...")
    model.fit(X, y)
    print("Done")
    print("Predicting...")
    y_pred = model.predict(X)
    print("Done")
    accuracy = np.sum(y_pred == y) / len(y)
    print(f"Train accuracy: {accuracy * 100 :.2f}%")
    print("*" * 20)


if __name__ == "__main__":
    main()
