package net.echo.brain4j.nlp.agents.model;

import net.echo.brain4j.activation.Activations;
import net.echo.brain4j.layer.impl.DenseLayer;
import net.echo.brain4j.layer.impl.DropoutLayer;
import net.echo.brain4j.model.Model;
import net.echo.brain4j.nlp.agents.attention.impl.MultiHeadAttention;

public class TransformerModel extends Model {
    public TransformerModel(int maxSequenceLength, int numHeads, int embeddingDim, double temperature, double topK) {
        super();
        for (int i = 0; i < 3; i++) {
            add(new DenseLayer(maxSequenceLength, Activations.SIGMOID));
            add(new MultiHeadAttention(numHeads, embeddingDim, temperature, topK));
            add(new DenseLayer(1024, Activations.RELU));
            add(new DropoutLayer(0.1));
            add(new DenseLayer(embeddingDim, Activations.SOFTMAX));
        }
    }
}


