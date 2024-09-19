package com.github.echo;

import com.github.echo.network.structure.layer.OutputLayer;
import com.github.echo.types.Activations;
import com.github.echo.network.NeuralNetwork;
import com.github.echo.network.structure.layer.DenseLayer;
import com.github.echo.training.DataRow;
import com.github.echo.training.DataSet;
import com.github.echo.training.models.Backpropagation;
import com.github.echo.types.loss.LossFunctions;

import java.util.Arrays;

public class Main {
    public static void main(String[] args) {
        NeuralNetwork network = new NeuralNetwork()
                .addLayer(new DenseLayer(2, Activations.LINEAR))
                .addLayer(new DenseLayer(16, Activations.RELU))
                .addLayer(new DenseLayer(16, Activations.RELU))
                .addLayer(new DenseLayer(16, Activations.RELU))
                .addLayer(new OutputLayer(1, Activations.SIGMOID, LossFunctions.BCE))
                .build();

        double[] input = new double[]{0, 1};

        DataRow sampleA = new DataRow(new double[]{0, 0}, 0);
        DataRow sampleB = new DataRow(new double[]{0, 1}, 1);
        DataRow sampleC = new DataRow(new double[]{1, 0}, 0);
        DataRow sampleD = new DataRow(new double[]{1, 1}, 1);

        DataSet dataSet = new DataSet(sampleA, sampleB, sampleC, sampleD);
        Backpropagation method = new Backpropagation(network);

        double value = 0.01;

        method.setLearningRate(value);
        method.setBatches(1);
        method.train(dataSet, value);

        System.out.println(Arrays.toString(network.calculateOutput(input)));
    }
}