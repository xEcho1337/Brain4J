package net.echo.brain4j.nlp.model;

import net.echo.brain4j.activation.Activations;
import net.echo.brain4j.layer.impl.DenseLayer;
import net.echo.brain4j.layer.impl.LayerNorm;
import net.echo.brain4j.model.Model;
import net.echo.brain4j.nlp.attention.MultiHeadAttention;
import net.echo.brain4j.utils.Vector;

import java.util.ArrayList;
import java.util.List;

public class TransformerEncoder extends Model {

    private Model feedForward;
    private LayerNorm normalizer;
    private MultiHeadAttention attention;
    private int contextSize;

    public TransformerEncoder(int layers, int numHeads, int contextSize, int dimension, double temperature) {
        this.normalizer = new LayerNorm();
        this.attention = new MultiHeadAttention(numHeads, contextSize, dimension, temperature);
        this.contextSize = contextSize;
        this.feedForward = new Model(
                new DenseLayer(dimension, Activations.LINEAR),
                new DenseLayer(4 * dimension, Activations.RELU),
                new DenseLayer(dimension, Activations.LINEAR)
        );
    }

    public List<Vector> transform(List<Vector> embeddings) {
        List<Vector> resulting = new ArrayList<>();

        for (Vector vector : embeddings) {
            Vector embedding = Vector.of(vector.toArray());
            normalizer.normalize(embedding);

            Vector attended = attention.attend(embedding.toArray());
            normalizer.normalize(attended);

            Vector result = feedForward.predict(attended);
            normalizer.normalize(result);

            resulting.add(result);
        }

        return resulting;
    }
}


