package net.echo.brain4j.nlp.agents.attention.impl;

import net.echo.brain4j.activation.Activations;
import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.nlp.agents.attention.AttentionMechanism;
import net.echo.brain4j.nlp.agents.attention.score.AttentionScorer;

import java.util.Random;

public class MultiHeadAttention extends Layer implements AttentionMechanism {
    private final int numHeads;
    private final int headDim;
    private final AttentionScorer scorer;
    private final double[][] projectionWeights;
    private final double[][] outputWeights;

    public MultiHeadAttention(int numHeads, int embeddingDim, double temperature, double topK) {
        super(embeddingDim, Activations.LINEAR);
        this.numHeads = numHeads;
        this.headDim = embeddingDim / numHeads;
        this.scorer = new AttentionScorer(temperature, topK);
        this.projectionWeights = new double[numHeads][embeddingDim];
        this.outputWeights = new double[embeddingDim][embeddingDim];

        initializeWeights();
    }

    private void initializeWeights() {
        Random random = new Random();
        for (int i = 0; i < numHeads; i++) {
            for (int j = 0; j < headDim; j++) {
                projectionWeights[i][j] = random.nextGaussian() * 0.02;
            }
        }

        for (int i = 0; i < outputWeights.length; i++) {
            for (int j = 0; j < outputWeights[0].length; j++) {
                outputWeights[i][j] = random.nextGaussian() * 0.02;
            }
        }
    }

    @Override
    public double[] attend(double[] input, String contextKey) {
        double[][] headOutputs = new double[numHeads][];

        for (int head = 0; head < numHeads; head++) {
            double[] projectedInput = projectToHead(input, head);
            headOutputs[head] = scorer.score(projectedInput, projectedInput, contextKey + "_head_" + head);
        }

        return concatenateAndProject(headOutputs);
    }

    private double[] projectToHead(double[] input, int head) {
        double[] projected = new double[headDim];
        for (int i = 0; i < headDim; i++) {
            for (int j = 0; j < input.length; j++) {
                projected[i] += input[j] * projectionWeights[head][j];
            }
        }
        return projected;
    }

    private double[] concatenateAndProject(double[][] headOutputs) {
        double[] concatenated = new double[headDim * numHeads];
        int offset = 0;

        for (double[] headOutput : headOutputs) {
            System.arraycopy(headOutput, 0, concatenated, offset, headDim);
            offset += headDim;
        }

        double[] output = new double[getNeurons().size()];
        for (int i = 0; i < output.length; i++) {
            for (int j = 0; j < concatenated.length; j++) {
                output[i] += concatenated[j] * outputWeights[i][j];
            }
        }

        return output;
    }
}


