package net.echo.brain4j.nlp.agents.model;

import net.echo.brain4j.activation.Activations;
import net.echo.brain4j.layer.impl.DenseLayer;
import net.echo.brain4j.layer.impl.LayerNorm;
import net.echo.brain4j.model.Model;
import net.echo.brain4j.nlp.agents.attention.impl.MultiHeadAttention;

public class TransformerEncoder extends Model {

    public TransformerEncoder(int layers, int numHeads, int embeddingDim, double temperature, double topK) {
        for (int i = 0; i < layers; i++) {
            add(
                    new LayerNorm(),
                    new MultiHeadAttention(numHeads, embeddingDim, temperature, topK),
                    // Add
                    new LayerNorm(),
                    new DenseLayer(1024, Activations.RELU),
                    new DenseLayer(embeddingDim, Activations.LINEAR)
                    // Add (again)
            );
        }

        add(new LayerNorm());
    }
}


