package net.echo.brain4j.nlp.attention;

import net.echo.brain4j.activation.Activations;
import net.echo.brain4j.layer.Layer;

import java.util.ArrayList;
import java.util.List;

public class MultiHeadAttention extends Layer {

    private final List<AttentionHead> heads;
    private final double temperature;
    private final int headCount;
    private final int contextSize;
    private final int dimension;

    public MultiHeadAttention(int headCount, int contextSize, int dimension, double temperature) {
        super(0, Activations.LINEAR);

        this.heads = new ArrayList<>();
        this.headCount = headCount;
        this.contextSize = contextSize;
        this.dimension = dimension;
        this.temperature = temperature;

        initializeHeads();
    }

    public double[] attend(double[] input) {
        List<double[]> attendedChanges = new ArrayList<>();

        heads.parallelStream().forEach(head -> {
            double[] result = head.attend(input);

            attendedChanges.add(result);
        });

        double[] result = input.clone();

        for (double[] changes : attendedChanges) {
            for (int i = 0; i < result.length; i++) {
                result[i] = result[i] + changes[i];
            }
        }

        return result;
    }

    private void initializeHeads() {
        for (int i = 0; i < headCount; i++) {
            heads.add(new AttentionHead(contextSize, dimension, temperature));
        }
    }
}
