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
                .addLayer(new Layer(300, Activations.RELU))
                .addLayer(new Layer(300, Activations.RELU))
                .addLayer(new Layer(3, Activations.RELU))
                .addLayer(new Layer(3, Activations.RELU))
                .addLayer(new Layer(10, Activations.SIGMOID))
                .build();

        double[] input = new double[]{0.1, 0.2};

        long start = System.nanoTime();
        for (int i = 0; i < 500; i++) {
            double[] output = network.calculateOutput(input);

            System.out.println(Arrays.toString(output));
            network.resetSynapses();
        }
        long end = System.nanoTime();

        System.out.println("TOOK " + (end - start) + " nano seconds or approximately " + (end - start) / 1e6 + " milli seconds");
    }
}