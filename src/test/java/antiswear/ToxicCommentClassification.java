package antiswear;

import net.echo.brain4j.activation.Activations;
import net.echo.brain4j.layer.impl.DenseLayer;
import net.echo.brain4j.loss.LossFunctions;
import net.echo.brain4j.model.initialization.InitializationType;
import net.echo.brain4j.nlp.encoding.PositionalEncoding;
import net.echo.brain4j.nlp.model.TransformerEncoder;
import net.echo.brain4j.training.optimizers.impl.Adam;
import net.echo.brain4j.utils.Vector;
import org.apache.commons.io.FileUtils;

import java.io.File;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class ToxicCommentClassification {

    private final static int EMBEDDING_DIM = 6;

    public static void main(String[] args) {
        TransformerEncoder model = new TransformerEncoder(2, 4,128, EMBEDDING_DIM, 0.6);

        model.add(new DenseLayer(6, Activations.SIGMOID));

        model.compile(InitializationType.XAVIER, LossFunctions.MEAN_SQUARED_ERROR, new Adam(0.001));

        var vectors = loadVocab();

        String phrase = "You are dumb";
        var embeddings = getEmbeddings(vectors, phrase);

        for (var embed : embeddings) {
            System.out.println(embed);
        }
    }

    private static List<Vector> getEmbeddings(Map<String, Vector> vectors, String phrase) {
        String[] tokens = phrase.split("\\s+");

        PositionalEncoding encoder = new PositionalEncoding(100, EMBEDDING_DIM);
        List<Vector> embeddings = new ArrayList<>();

        for (int i = 0; i < tokens.length; i++) {
            Vector vector = vectors.get(tokens[i].toLowerCase());

            double[] encoded = encoder.encode(vector.toArray(), i);

            embeddings.add(new Vector(encoded));
        }

        return embeddings;
    }

    private static Map<String, Vector> loadVocab() {
        Map<String, Vector> vectors = new HashMap<>();

        try {
            List<String> content = FileUtils.readLines(new File("vocab.txt"), StandardCharsets.UTF_8);

            for (String token : content) {
                if (token.startsWith("[") || token.startsWith("##")) continue;

                Vector embedding = new Vector(EMBEDDING_DIM);

                embedding.fill(Math::random).scale(10);

                vectors.put(token, embedding);
            }
        } catch (Exception e) {
            e.printStackTrace(System.err);
        }

        return vectors;
    }
}