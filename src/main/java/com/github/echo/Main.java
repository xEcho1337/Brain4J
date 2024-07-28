package com.github.echo;

import com.github.echo.activations.Activations;
import com.github.echo.network.NeuralNetwork;
import com.github.echo.network.structure.layer.Layer;
import com.github.echo.network.structure.layer.impl.InputLayer;

import java.util.Arrays;

public class Main {
    public static void main(String[] args) {
        NeuralNetwork network = new NeuralNetwork()
                .addLayer(new InputLayer(2))
                .addLayer(new Layer(3, Activations.RELU))
                .addLayer(new Layer(3, Activations.SIGMOID))
                .build();

        double[] input = new double[]{0.1, 0.2};
        double[] output = network.calculateOutput(input);

        System.out.println(Arrays.toString(output));
        System.out.println("Hello world!");
    }
}