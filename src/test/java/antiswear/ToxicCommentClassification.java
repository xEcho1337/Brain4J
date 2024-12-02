package antiswear;

import net.echo.brain4j.activation.Activations;
import net.echo.brain4j.layer.impl.DenseLayer;
import net.echo.brain4j.loss.LossFunctions;
import net.echo.brain4j.model.initialization.InitializationType;
import net.echo.brain4j.nlp.agents.model.TransformerEncoder;
import net.echo.brain4j.training.optimizers.impl.Adam;
import org.apache.commons.io.FileUtils;

import java.io.File;
import java.nio.charset.StandardCharsets;
import java.util.List;

public class ToxicCommentClassification {

    private final static int EMBEDDING_DIM = 128;

    public static void main(String[] args) {
        TransformerEncoder model = new TransformerEncoder(2, 128, EMBEDDING_DIM, 0.6, 0.95);

        model.add(new DenseLayer(6, Activations.SIGMOID));

        model.compile(InitializationType.XAVIER, LossFunctions.BINARY_CROSS_ENTROPY, new Adam(0.001));

        loadVocab();
    }

    private static void loadVocab() {
        try {
            List<String> content = FileUtils.readLines(new File("vocab.txt"), StandardCharsets.UTF_8);

            for (String token : content) {
                if (token.startsWith("[") || token.startsWith("##")) continue;

                System.out.println("Adding " + token);
            }
        } catch (Exception e) {
            e.printStackTrace(System.err);
        }
    }
}
