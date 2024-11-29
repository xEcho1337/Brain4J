package net.echo.brain4j.nlp.agents.attention.impl;

import net.echo.brain4j.activation.Activations;
import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.nlp.agents.attention.AttentionMechanism;
import net.echo.brain4j.nlp.agents.attention.score.AttentionScorer;

import java.util.Random;

public class SelfAttention extends Layer implements AttentionMechanism {
    private final AttentionScorer scorer;
    private final int headDim;
    private final double[][] queryWeights;
    private final double[][] keyWeights;
    private final double[][] valueWeights;

    public SelfAttention(int embeddingDim, double temperature, double topK) {
        super(embeddingDim, Activations.LINEAR);

        this.scorer = new AttentionScorer(temperature, topK);
        this.headDim = embeddingDim;

        this.queryWeights = new double[embeddingDim][headDim];
        this.keyWeights = new double[embeddingDim][headDim];
        this.valueWeights = new double[embeddingDim][headDim];

        initializeWeights();
    }

    private void initializeWeights() {
        Random random = new Random();

        for (int i = 0; i < headDim; i++) {
            for (int j = 0; j < headDim; j++) {
                queryWeights[i][j] = random.nextGaussian() * 0.02;
                keyWeights[i][j] = random.nextGaussian() * 0.02;
                valueWeights[i][j] = random.nextGaussian() * 0.02;
            }
        }
    }

    @Override
    public double[] attend(double[] input, String contextKey) {
        double[] query = projectVector(input, queryWeights);
        double[] key = projectVector(input, keyWeights);
        double[] value = projectVector(input, valueWeights);

        double[] attentionScores = scorer.score(query, key, contextKey);
        return computeWeightedSum(attentionScores, value);
    }

    private double[] projectVector(double[] input, double[][] weights) {
        double[] output = new double[headDim];
        for (int i = 0; i < headDim; i++) {
            for (int j = 0; j < input.length; j++) {
                output[i] += input[j] * weights[j][i];
            }
        }
        return output;
    }

    private double[] computeWeightedSum(double[] scores, double[] values) {
        double[] output = new double[values.length];
        for (int i = 0; i < values.length; i++) {
            output[i] = scores[i] * values[i];
        }
        return output;
    }
}

