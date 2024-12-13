import net.echo.brain4j.activation.Activations;
import net.echo.brain4j.training.data.DataRow;
import net.echo.brain4j.training.data.DataSet;
import net.echo.brain4j.model.initialization.WeightInitialization;
import net.echo.brain4j.layer.impl.DenseLayer;
import net.echo.brain4j.loss.LossFunctions;
import net.echo.brain4j.model.Model;
import net.echo.brain4j.training.optimizers.impl.GradientDescent;
import net.echo.brain4j.training.updater.impl.NormalUpdater;
import net.echo.brain4j.utils.Vector;

public class XorTest {

    public static void main(String[] args) {
        Model network = new Model(
                new DenseLayer(2, Activations.LINEAR),
                new DenseLayer(16, Activations.RELU),
                new DenseLayer(16, Activations.RELU),
                new DenseLayer(1, Activations.SIGMOID)
        );

        network.compile(
                WeightInitialization.HE,
                LossFunctions.BINARY_CROSS_ENTROPY,
                new GradientDescent(0.1),
                new NormalUpdater()
        );

        System.out.println(network.getStats());

        DataRow first = new DataRow(Vector.of(0, 0), Vector.of(0));
        DataRow second = new DataRow(Vector.of(0, 1), Vector.of(1));
        DataRow third = new DataRow(Vector.of(1, 0), Vector.of(1));
        DataRow fourth = new DataRow(Vector.of(1, 1), Vector.of(0));

        DataSet training = new DataSet(first, second, third, fourth);

        long start = System.nanoTime();

        int epoch = 0;
        double error = Double.MAX_VALUE;

        do {
            network.fit(training, 1);

            if (epoch % 100 == 0) {
                error = network.evaluate(training);
                System.out.println("Epoch #" + epoch + " error: " + error);
            }

            epoch++;
        } while (error > 0.01);

        System.out.println("completed in " + epoch + " with error " + error);
        /*int steps = 1000;

        for (int i = 0; i < steps; i++) {
            network.fit(training, 1);
        }

        long took = System.nanoTime() - start;
        double error = network.evaluate(training);

        System.out.println("Took " + took + " ns or " + (took / 1e6) + " ms, with an average of " + (took / 1e6 / steps) + " ms per step");
        System.out.println("Completed with error " + error);*/

        for (DataRow row : training.getDataRows()) {
            Vector output = network.predict(row.inputs());

            System.out.println("Input: " + row.inputs() + " Output: " + output.get(0));
        }
    }
}
