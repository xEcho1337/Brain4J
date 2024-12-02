import net.echo.brain4j.activation.Activations;
import net.echo.brain4j.training.data.DataRow;
import net.echo.brain4j.training.data.DataSet;
import net.echo.brain4j.model.initialization.InitializationType;
import net.echo.brain4j.layer.impl.DenseLayer;
import net.echo.brain4j.loss.LossFunctions;
import net.echo.brain4j.model.Model;
import net.echo.brain4j.training.optimizers.impl.Adam;
import net.echo.brain4j.utils.Vector;

import java.util.Arrays;

public class XorTest {

    public static void main(String[] args) {
        Model network = new Model(
                new DenseLayer(2, Activations.LINEAR),
                new DenseLayer(32, Activations.RELU),
                new DenseLayer(32, Activations.RELU),
                new DenseLayer(32, Activations.RELU),
                new DenseLayer(1, Activations.SIGMOID)
        );

        network.compile(InitializationType.HE, LossFunctions.BINARY_CROSS_ENTROPY, new Adam(0.001));

        System.out.println(network.getStats());

        DataRow first = new DataRow(new Vector(0.0, 0.0), new Vector(0.0));
        DataRow second = new DataRow(new Vector(0, 1), new Vector(1.0));
        DataRow third = new DataRow(new Vector(1, 0), new Vector(1.0));
        DataRow fourth = new DataRow(new Vector(1, 1), new Vector(0.0));

        DataSet training = new DataSet(first, second, third, fourth);

        long start = System.nanoTime();
        int epoches = 0;
        double currError;

        do {
            epoches++;

            currError = network.evaluate(training);

            System.out.println("Epoch " + epoches + " with error " + currError);

            network.fit(training, 1);
        } while (currError > 0.01);

        double error = network.evaluate(training);
        double took = (System.nanoTime() - start) / 1e6;

        System.out.println("Took " + took + " ms with an average of " + (took / epoches) + " ms per epoch and error " + error);

        for (DataRow row : training.getDataRows()) {
            Vector output = network.predict(row.inputs());

            System.out.println("Input: " + row.inputs() + " Output: " + output.get(0));
        }

        network.save("xor-test.json");
    }
}
