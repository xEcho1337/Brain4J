import net.echo.brain4j.activation.Activations;
import net.echo.brain4j.training.data.DataRow;
import net.echo.brain4j.training.data.DataSet;
import net.echo.brain4j.model.initialization.WeightInitialization;
import net.echo.brain4j.layer.impl.DenseLayer;
import net.echo.brain4j.loss.LossFunctions;
import net.echo.brain4j.model.Model;
import net.echo.brain4j.training.optimizers.impl.Adam;
import net.echo.brain4j.training.optimizers.impl.GradientDescent;
import net.echo.brain4j.training.updater.impl.NormalUpdater;
import net.echo.brain4j.training.updater.impl.StochasticUpdater;
import net.echo.brain4j.utils.Vector;

import java.util.Random;

public class XorTest {

    public static void main(String[] args) {
        Model model = new Model(
                new DenseLayer(2, Activations.LINEAR),
                new DenseLayer(16, Activations.RELU),
                new DenseLayer(16, Activations.RELU),
                new DenseLayer(1, Activations.SIGMOID)
        );

        model.compile(
                WeightInitialization.HE,
                LossFunctions.BINARY_CROSS_ENTROPY,
                new Adam(0.1),
                new NormalUpdater()
        );

        DataRow first = new DataRow(Vector.of(0, 0), Vector.of(0));
        DataRow second = new DataRow(Vector.of(0, 1), Vector.of(1));
        DataRow third = new DataRow(Vector.of(1, 0), Vector.of(1));
        DataRow fourth = new DataRow(Vector.of(1, 1), Vector.of(0));

        DataSet training = new DataSet(first, second, third, fourth);

        trainForBenchmark(model, training);
    }

    private static void trainForBenchmark(Model model, DataSet data) {
        long start = System.nanoTime();

        for (int i = 0; i < 5000; i++) {
            model.fit(data, 1);
        }

        long end = System.nanoTime();

        double took = (end - start) / 1e6;
        double error = model.evaluate(data);

        System.out.println("Completed 5000 epoches in " + took + " ms with error: " + error);
    }

    private static void trainTillError(Model model, DataSet data) {
        double error;
        int epoches = 0;

        do {
            model.fit(data, 1);

            error = model.evaluate(data);
            epoches++;

            System.out.println("Epoch " + epoches + " error: " + error);
        } while (error > 0.01);
    }
}
