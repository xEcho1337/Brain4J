import net.echo.brain4j.activation.Activations;
import net.echo.brain4j.training.data.DataRow;
import net.echo.brain4j.training.data.DataSet;
import net.echo.brain4j.model.initialization.WeightInitialization;
import net.echo.brain4j.layer.impl.DenseLayer;
import net.echo.brain4j.loss.LossFunctions;
import net.echo.brain4j.model.Model;
import net.echo.brain4j.training.optimizers.impl.Adam;
import net.echo.brain4j.utils.Vector;

public class XorTest {

    public static void main(String[] args) {
        Model network = new Model(
                new DenseLayer(2, Activations.LINEAR),
                new DenseLayer(16, Activations.RELU),
                new DenseLayer(16, Activations.RELU),
                new DenseLayer(1, Activations.SIGMOID)
        );

        network.compile(WeightInitialization.HE, LossFunctions.BINARY_CROSS_ENTROPY, new Adam(0.01));

        System.out.println(network.getStats());

        DataRow first = new DataRow(Vector.of(0, 0), Vector.of(0));
        DataRow second = new DataRow(Vector.of(0, 1), Vector.of(1));
        DataRow third = new DataRow(Vector.of(1, 0), Vector.of(1));
        DataRow fourth = new DataRow(Vector.of(1, 1), Vector.of(0));

        DataSet training = new DataSet(first, second, third, fourth);

        double error;

        do {
            network.fit(training, 1);
            error = network.evaluate(training);
        } while (error > 0.01);

        System.out.println("Completed with error " + error);

        for (DataRow row : training.getDataRows()) {
            Vector output = network.predict(row.inputs());

            System.out.println("Input: " + row.inputs() + " Output: " + output.get(0));
        }
    }
}
