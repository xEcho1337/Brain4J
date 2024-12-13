package antiswear;

import net.echo.brain4j.activation.Activations;
import net.echo.brain4j.layer.impl.DenseLayer;
import net.echo.brain4j.loss.LossFunctions;
import net.echo.brain4j.model.initialization.WeightInitialization;
import net.echo.brain4j.nlp.encoding.PositionalEncoding;
import net.echo.brain4j.nlp.model.Transformer;
import net.echo.brain4j.nlp.model.layers.TransformerEncoder;
import net.echo.brain4j.training.optimizers.impl.Adam;
import net.echo.brain4j.training.updater.impl.StochasticUpdater;
import net.echo.brain4j.utils.Vector;
import org.apache.commons.io.FileUtils;

import java.io.File;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class ToxicCommentClassification {

    private final static int CONTEXT_SIZE = 12;
    private final static int EMBEDDING_DIM = 6;
    private final static double TEMPERATURE = 0.6;

    public static void main(String[] args) {
        Transformer transformer = new Transformer(
                new TransformerEncoder(4, CONTEXT_SIZE, EMBEDDING_DIM, TEMPERATURE)
        );

        transformer.concat(
                new DenseLayer(EMBEDDING_DIM, Activations.LINEAR), // Subsampling layer, required to make the model work
                new DenseLayer(6, Activations.SIGMOID)
        );

        transformer.compile(
                WeightInitialization.XAVIER,
                LossFunctions.MEAN_SQUARED_ERROR,
                new Adam(0.001),
                new StochasticUpdater()
        );

        var vectors = loadVocab();

        String phrase = "the pen is on the table";
        var embeddings = getEmbeddings(vectors, phrase);

        List<Vector> output = transformer.transform(embeddings);

        for (Vector vector : output) {
            System.out.println(vector);
        }
    }

    private static List<Vector> getEmbeddings(Map<String, Vector> vectors, String phrase) {
        String[] tokens = phrase.split("\\s+");

        PositionalEncoding encoder = new PositionalEncoding(100, EMBEDDING_DIM);
        List<Vector> embeddings = new ArrayList<>();

        for (int i = 0; i < tokens.length; i++) {
            Vector vector = vectors.get(tokens[i].toLowerCase());

            double[] encoded = encoder.encode(vector.toArray(), i);

            embeddings.add(Vector.of(encoded));
        }

        return embeddings;
    }

    private static Map<String, Vector> loadVocab() {
        Map<String, Vector> vectors = new HashMap<>();

        try {
            List<String> content = FileUtils.readLines(new File("vocab.txt"), StandardCharsets.UTF_8);

            for (String token : content) {
                if (token.startsWith("[") || token.startsWith("##")) continue;

                vectors.put(token, Vector.random(EMBEDDING_DIM).scale(10));
            }
        } catch (Exception e) {
            e.printStackTrace(System.err);
        }

        return vectors;
    }
}
