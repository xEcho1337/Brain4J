package net.echo.brain4j.nlp.model;

import net.echo.brain4j.activation.Activations;
import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.layer.impl.DenseLayer;
import net.echo.brain4j.layer.impl.LayerNorm;
import net.echo.brain4j.model.Model;
import net.echo.brain4j.nlp.attention.MultiHeadAttention;
import net.echo.brain4j.utils.Vector;

import java.util.ArrayList;
import java.util.List;

public class TransformerEncoder extends Model {

    private List<Model> denseLayers;
    private LayerNorm normalizer;
    private MultiHeadAttention attention;
    private int contextSize;

    public TransformerEncoder(int layers, int numHeads, int contextSize, int dimension, double temperature) {
        this.normalizer = new LayerNorm();
        this.attention = new MultiHeadAttention(numHeads, contextSize, dimension, temperature);
        this.contextSize = contextSize;
        this.denseLayers = new ArrayList<>();

        for (int i = 0; i < contextSize; i++) {
            Model model = new Model(
                    new DenseLayer(dimension, Activations.LINEAR),
                    new DenseLayer(1024, Activations.RELU),
                    new DenseLayer(contextSize, Activations.LINEAR)
            );

            this.denseLayers.add(model);
        }

        /*for (int i = 0; i < layers; i++) {
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

        add(new LayerNorm());*/
    }

    public List<Vector> transform(List<Vector> embeddings) {
        List<Vector> resulting = new ArrayList<>();

        for (int i = 0; i < embeddings.size(); i++) {
            Vector embedding = Vector.of(embeddings.get(i).toArray());

            normalizer.normalize(embedding);

            Vector attended = attention.attend(embedding.toArray());

            normalizer.normalize(attended);

            Vector result = denseLayers.get(i).predict(attended);

            normalizer.normalize(result);

            resulting.add(result);
        }

        return resulting;
    }
}


