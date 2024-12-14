package net.echo.brain4j.training;

import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.layer.impl.DropoutLayer;
import net.echo.brain4j.model.Model;
import net.echo.brain4j.structure.Neuron;
import net.echo.brain4j.training.data.DataRow;
import net.echo.brain4j.training.data.DataSet;
import net.echo.brain4j.training.optimizers.Optimizer;
import net.echo.brain4j.training.updater.Updater;
import net.echo.brain4j.utils.Vector;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;

public class BackPropagation {

    private final Model model;
    private final Optimizer optimizer;
    private final Updater updater;

    public BackPropagation(Model model, Optimizer optimizer, Updater updater) {
        this.model = model;
        this.optimizer = optimizer;
        this.updater = updater;
    }

    private List<DataRow> partition(List<DataRow> rows, int batches, int offset) {
        return rows.subList(offset * batches, Math.max((offset + 1) * batches, rows.size()));
    }

    public void iterate(DataSet dataSet, int batches) {
        /*List<DataRow> rows = dataSet.getDataRows();
        
        int rowsPerBatch = rows.size() / batches;

        List<Thread> threads = new ArrayList<>();

        for (int i = 0; i < batches; i++) {
            List<DataRow> batch = partition(rows, rowsPerBatch, i);

            Thread thread = Thread.startVirtualThread(() -> {
                for (DataRow row : batch) {
                    Vector output = model.predict(row.inputs());
                    Vector target = row.outputs();

                    backpropagate(target.toArray(), output.toArray());
                }
            });

            threads.add(thread);
        }

        for (Thread thread : threads) {
            try {
                thread.join();
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
        }*/
        for (DataRow row : dataSet.getDataRows()) {
            Vector output = model.predict(row.inputs());
            Vector target = row.outputs();

            backpropagate(target.toArray(), output.toArray());
        }

        List<Layer> layers = model.getLayers();
        updater.postFit(layers, optimizer.getLearningRate());
    }

    public void backpropagate(double[] targets, double[] outputs) {
        List<Layer> layers = model.getLayers();
        initialDelta(layers, targets, outputs);

        for (int l = layers.size() - 2; l > 0; l--) {
            Layer layer = layers.get(l);

            if (layer instanceof DropoutLayer dropoutLayer) {
                Layer previous = layers.get(l - 1);
                dropoutLayer.backward(previous.getNeurons());
                continue;
            }

            layer.propagate(updater, optimizer);
        }

        optimizer.postIteration(updater, layers);
        updater.postIteration(layers, optimizer.getLearningRate());
    }

    private void initialDelta(List<Layer> layers, double[] targets, double[] outputs) {
        Layer outputLayer = layers.get(layers.size() - 1);

        for (int i = 0; i < outputLayer.getNeurons().size(); i++) {
            Neuron neuron = outputLayer.getNeuronAt(i);

            // Calculates the error of the output
            double output = outputs[i];
            double error = targets[i] - output;

            // Calculates the delta using the error and the derivative of the output
            double delta = error * outputLayer.getActivation().getFunction().getDerivative(output);
            neuron.setDelta(delta);
        }
    }
}