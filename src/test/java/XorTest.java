import net.echo.brain4j.activation.Activations;
import net.echo.brain4j.training.data.DataRow;
import net.echo.brain4j.training.data.DataSet;
import net.echo.brain4j.model.initialization.InitializationType;
import net.echo.brain4j.layer.impl.DenseLayer;
import net.echo.brain4j.loss.LossFunctions;
import net.echo.brain4j.model.Model;
import net.echo.brain4j.model.impl.FeedForwardModel;
import net.echo.brain4j.training.optimizers.impl.Adam;

import java.util.Arrays;

public class XorTest {

    public static void main(String[] args) {
        Model network = new FeedForwardModel();

        network.add(new DenseLayer(2, Activations.LINEAR));
        network.add(new DenseLayer(256, Activations.RELU));
        network.add(new DenseLayer(256, Activations.RELU));
        network.add(new DenseLayer(256, Activations.RELU));
        network.add(new DenseLayer(1, Activations.SIGMOID));

        network.compile(InitializationType.XAVIER, LossFunctions.BINARY_CROSS_ENTROPY, new Adam(0.001));

        System.out.println(network.getStats());

        DataRow first = new DataRow(new double[]{0.0, 0.0}, 0.0);
        DataRow second = new DataRow(new double[]{0.0, 1.0}, 1.0);
        DataRow third = new DataRow(new double[]{1.0, 0.0}, 1.0);
        DataRow fourth = new DataRow(new double[]{1.0, 1.0}, 0.0);

        DataSet training = new DataSet(first, second, third, fourth);

        long start = System.nanoTime();
        int epoches = 0;

        while (epoches < 500) {
            epoches++;

            double error = network.evaluate(training);

            System.out.println("Epoch " + epoches + " with error " + error);

            network.fit(training);
        }

        double error = network.evaluate(training);
        double took = (System.nanoTime() - start) / 1e6;

        System.out.println("Took " + took + " ms with an average of " + (took / epoches) + " ms per epoch and error " + error);

        for (DataRow row : training.getDataRows()) {
            double[] output = network.predict(row.inputs());

            System.out.println("Input: " + Arrays.toString(row.inputs()) + " Output: " + output[0]);
        }

        network.save("xor-test.json");
    }
}
