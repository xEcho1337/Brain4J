import net.echo.brain4j.activation.Activations;
import net.echo.brain4j.layer.impl.DenseLayer;
import net.echo.brain4j.layer.impl.DropoutLayer;
import net.echo.brain4j.loss.LossFunctions;
import net.echo.brain4j.model.Model;
import net.echo.brain4j.model.impl.FeedForwardModel;
import net.echo.brain4j.model.initialization.InitializationType;
import net.echo.brain4j.training.data.DataRow;
import net.echo.brain4j.training.data.DataSet;
import net.echo.brain4j.training.optimizers.impl.Adam;

import java.util.Arrays;

public class AntiCheatClassifier {

    public static void main(String[] args) {
        int amount = 50;

        Model network = new FeedForwardModel(
                new DenseLayer(amount, Activations.LINEAR),
                new DropoutLayer(0.5),
                new DenseLayer(32, Activations.RELU),
                new DropoutLayer(0.5),
                new DenseLayer(32, Activations.RELU),
                new DropoutLayer(0.5),
                new DenseLayer(32, Activations.RELU),
                new DropoutLayer(0.5),
                new DenseLayer(5, Activations.SIGMOID)
        );

        network.compile(InitializationType.XAVIER, LossFunctions.MEAN_SQUARED_ERROR, new Adam(0.001));

        System.out.println(network.getStats());

        DataRow grim = new DataRow(getGrimConfig(amount), 1, 0, 0, 0, 0);
        DataRow polar = new DataRow(getPolarConfig(amount), 0, 1, 0, 0, 0);
        DataRow karhu = new DataRow(getKarhuConfig(amount), 0, 0, 1, 0, 0);
        DataRow intave = new DataRow(getIntaveConfig(amount), 0, 0, 0, 1, 0);
        DataRow matrix = new DataRow(getMatrixConfig(amount), 0, 0, 0, 0, 1);

        DataSet training = new DataSet(grim, polar, karhu, intave, matrix);
        double error;
        long start = System.nanoTime();

        int epoches = 0;

        do {
            epoches++;

            error = network.fit(training);

            if (epoches % 100 == 0) {
                System.out.println("Epoch #" + epoches + " has error " + error);
            }
        } while (error > 0.01);

        network.save("anticheat.json");

        double took = (System.nanoTime() - start) / 1e6;

        System.out.println("Took " + took + " ms with an average of " + (took / epoches) + " ms per epoch and error " + error);

        for (DataRow row : training.getDataRows()) {
            double[] output = network.predict(row.inputs());

            System.out.println("Output: " + Arrays.toString(output));
        }

        network.load("anticheat.json");

        for (DataRow row : training.getDataRows()) {
            double[] output = network.predict(row.inputs());

            System.out.println("Output 2: " + Arrays.toString(output));
        }

    }
    
    private static double[] getGrimConfig(int num) {
        double[] transactions = new double[num];

        for (int i = 0; i < num; i++) {
            transactions[i] = -i;
        }

        return transactions;
    }

    private static double[] getMatrixConfig(int num) {
        double[] transactions = new double[num];

        for (int i = 0; i < num; i++) {
            transactions[i] = 100 + i;
        }

        return transactions;
    }

    private static double[] getIntaveConfig(int num) {
        double[] transactions = new double[num];

        for (int i = 0; i < num; i++) {
            transactions[i] = Math.random() * -10;
        }

        return transactions;
    }

    private static double[] getVulcanConfig(int num) {
        double[] transactions = new double[num];

        for (int i = 0; i < num; i++) {
            transactions[i] = -23767 + i;
        }

        return transactions;
    }

    private static double[] getPolarConfig(int num) {
        double[] transactions = new double[num];

        transactions[0] = Math.random() * -1500;

        for (int i = 1; i < num; i++) {
            transactions[i] = -100 - i;
        }

        return transactions;
    }

    private static double[] getKarhuConfig(int num) {
        double[] transactions = new double[num];

        for (int i = 0; i < num; i++) {
            transactions[i] = -i - 3000;
        }

        return transactions;
    }
}
