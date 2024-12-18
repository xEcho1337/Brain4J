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

    private List<DataRow> partition(List<DataRow> rows, double batches, int offset) {
        int start = (int) Math.min(offset * batches, rows.size());
        int stop = (int) Math.min((offset + 1) * batches, rows.size());

        return rows.subList(start, stop);
    }

    public void iterate(DataSet dataSet, int batches) {
        List<DataRow> rows = dataSet.getDataRows();
        double rowsPerBatch = (double) rows.size() / batches;

        for (int i = 0; i < batches; i++) {
            List<DataRow> batch = partition(dataSet.getDataRows(), rowsPerBatch, i);
            List<Thread> threads = new ArrayList<>();

            for (DataRow row : batch) {
                Thread thread = Thread.startVirtualThread(() -> {
                    Vector output = model.predict(row.inputs());
                    Vector target = row.outputs();

                    backpropagate(target.toArray(), output.toArray());
                });

                threads.add(thread);
            }

            waitAll(threads);

            List<Layer> layers = model.getLayers();
            updater.postFit(layers, optimizer.getLearningRate());
        }
    }

    private void waitAll(List<Thread> threads) {
        for (Thread thread : threads) {
            try {
                thread.join();
            } catch (InterruptedException e) {
                e.printStackTrace(System.err);
            }
        }
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

            double output = outputs[i];
            double error = targets[i] - output;

            double delta = error * outputLayer.getActivation().getFunction().getDerivative(output);
            neuron.setDelta(delta);
        }
    }
}