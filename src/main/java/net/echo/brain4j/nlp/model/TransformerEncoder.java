package net.echo.brain4j.nlp.model;

import net.echo.brain4j.activation.Activations;
import net.echo.brain4j.layer.impl.DenseLayer;
import net.echo.brain4j.layer.impl.LayerNorm;
import net.echo.brain4j.model.Model;
import net.echo.brain4j.nlp.attention.MultiHeadAttention;

public class TransformerEncoder extends Model {

    public TransformerEncoder(int layers, int numHeads, int contextSize, int dimension, double temperature) {
        for (int i = 0; i < layers; i++) {
            add(
                    new LayerNorm(),
                    new MultiHeadAttention(numHeads, contextSize, dimension, temperature),
                    // Add
                    new LayerNorm(),
                    new DenseLayer(1024, Activations.RELU),
                    new DenseLayer(contextSize, Activations.LINEAR)
                    // Add (again)
            );
        }

        add(new LayerNorm());
    }
}


