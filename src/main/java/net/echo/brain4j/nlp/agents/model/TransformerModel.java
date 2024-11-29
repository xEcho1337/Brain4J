package net.echo.brain4j.nlp.agents.model;

import net.echo.brain4j.activation.Activations;
import net.echo.brain4j.layer.impl.DenseLayer;
import net.echo.brain4j.layer.impl.DropoutLayer;
import net.echo.brain4j.model.impl.FeedForwardModel;
import net.echo.brain4j.nlp.agents.attention.impl.MultiHeadAttention;

public class TransformerModel extends FeedForwardModel {
    public TransformerModel(int maxSequenceLength, int numHeads, int embeddingDim, double temperature, double topK) {
        super(
                new DenseLayer(maxSequenceLength, Activations.SIGMOID),
                new MultiHeadAttention(numHeads, embeddingDim, temperature, topK),
                new DenseLayer(4098, Activations.RELU),
                new DropoutLayer(0.1),
                new DenseLayer(embeddingDim, Activations.RELU)
        );
    }
}


