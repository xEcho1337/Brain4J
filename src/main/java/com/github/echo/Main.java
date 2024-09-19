package com.github.echo;

import com.github.echo.network.NeuralConfiguration;
import com.github.echo.network.structure.layer.OutputLayer;
import com.github.echo.types.Activations;
import com.github.echo.network.NeuralNetwork;
import com.github.echo.network.structure.layer.DenseLayer;
import com.github.echo.training.DataRow;
import com.github.echo.training.DataSet;
import com.github.echo.training.impl.BackPropagation;
import com.github.echo.types.loss.LossFunctions;

import java.util.Arrays;

public class Main {
    public static void main(String[] args) {
        NeuralConfiguration configuration = new NeuralConfiguration()
                .layer(new DenseLayer(2, Activations.LINEAR))
                .layer(new DenseLayer(16, Activations.RELU))
                .layer(new DenseLayer(16, Activations.RELU))
                .layer(new DenseLayer(16, Activations.RELU))
                .layer(new OutputLayer(1, Activations.SIGMOID, LossFunctions.BCE));

        NeuralNetwork network = new NeuralNetwork(configuration);

        double[] input = new double[]{0, 1};

        DataRow sampleA = new DataRow(new double[]{0, 0}, 0);
        DataRow sampleB = new DataRow(new double[]{0, 1}, 1);
        DataRow sampleC = new DataRow(new double[]{1, 0}, 1);
        DataRow sampleD = new DataRow(new double[]{1, 1}, 0);

        DataSet dataSet = new DataSet(sampleA, sampleB, sampleC, sampleD);
        BackPropagation method = new BackPropagation(network);

        double value = 0.01;

        method.setLearningRate(value);
        method.setBatches(2);
        method.train(dataSet, value);

        System.out.println(Arrays.toString(network.calculateOutput(input)));
    }
}