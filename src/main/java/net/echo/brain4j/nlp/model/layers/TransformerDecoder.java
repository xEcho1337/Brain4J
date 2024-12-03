package net.echo.brain4j.nlp.model.layers;

import net.echo.brain4j.activation.Activations;
import net.echo.brain4j.layer.impl.DenseLayer;
import net.echo.brain4j.layer.impl.LayerNorm;
import net.echo.brain4j.model.Model;
import net.echo.brain4j.nlp.attention.MultiHeadAttention;

public class TransformerDecoder extends Model {

    public TransformerDecoder(int layers, int numHeads, int contextSize, int embeddingDim, double temperature) {
        for (int i = 0; i < layers; i++) {
            add(
                    new LayerNorm(),
                    new MultiHeadAttention(numHeads, contextSize, embeddingDim, temperature), // TODO Masked MHSA
                    // Add
                    new LayerNorm(),
                    new MultiHeadAttention(numHeads, contextSize, embeddingDim, temperature), // TODO: Multi Headed Cross-Attention
                    // Add (again)
                    new LayerNorm(),
                    new DenseLayer(1024, Activations.RELU),
                    new DenseLayer(embeddingDim, Activations.LINEAR)
                    // Add (again)
            );
        }

        add(new LayerNorm(), new DenseLayer(embeddingDim, Activations.SOFTMAX));
    }
}


