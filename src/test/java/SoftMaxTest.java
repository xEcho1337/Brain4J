import net.echo.brain4j.activation.Activations;
import net.echo.brain4j.layer.impl.DenseLayer;
import net.echo.brain4j.loss.LossFunctions;
import net.echo.brain4j.model.Model;
import net.echo.brain4j.model.initialization.InitializationType;
import net.echo.brain4j.training.data.DataRow;
import net.echo.brain4j.training.data.DataSet;
import net.echo.brain4j.training.optimizers.impl.Adam;
import net.echo.brain4j.utils.Vector;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class SoftMaxTest {

    private final static int BOUND = 1000;

    public static void main(String[] args) {
        Model model = new Model(
                new DenseLayer(1, Activations.LINEAR),
                new DenseLayer(32, Activations.SIGMOID),
                new DenseLayer(32, Activations.SIGMOID),
                new DenseLayer(1, Activations.SIGMOID)
        );

        model.compile(InitializationType.XAVIER, LossFunctions.MEAN_SQUARED_ERROR, new Adam(0.001));

        System.out.println(model.getStats());

        DataSet set = getDataSet();

        double error;

        int epoch = 0;

        do {
            epoch++;

            error = model.evaluate(set);

            if (epoch % 10 == 0) {
                System.out.println("Epoch: " + epoch + " | Error: " + error);
            }

            model.fit(set, 1);
        } while(error > 0.01);
    }

    public static DataSet getDataSet() {
        List<DataRow> rows = new ArrayList<>();

        for (int i = 0; i < BOUND * 2; i++) {
            double x = Math.random() * BOUND * 2 - BOUND;
            double y = Math.sin(Math.toRadians(x));

            System.out.println("(" + x + ", " + y + ") ");
            rows.add(new DataRow(new Vector(x), new Vector(y)));
        }

        return new DataSet(rows);
    }
}
