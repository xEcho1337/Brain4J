package net.echo.brain4j.model.impl;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.google.gson.reflect.TypeToken;
import net.echo.brain4j.adapters.LayerAdapter;
import net.echo.brain4j.adapters.OptimizerAdapter;
import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.layer.impl.DenseLayer;
import net.echo.brain4j.layer.impl.DropoutLayer;
import net.echo.brain4j.loss.LossFunction;
import net.echo.brain4j.loss.LossFunctions;
import net.echo.brain4j.model.Model;
import net.echo.brain4j.model.initialization.InitializationType;
import net.echo.brain4j.structure.Neuron;
import net.echo.brain4j.structure.Synapse;
import net.echo.brain4j.training.BackPropagation;
import net.echo.brain4j.training.data.DataRow;
import net.echo.brain4j.training.data.DataSet;
import net.echo.brain4j.training.optimizers.Optimizer;
import net.echo.brain4j.training.optimizers.impl.Adam;
import net.echo.brain4j.training.optimizers.impl.SGD;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class FeedForwardModel implements Model {

    private static final OptimizerAdapter OPTIMIZER_ADAPTER = new OptimizerAdapter();
    private static final LayerAdapter LAYER_ADAPTER = new LayerAdapter();
    private static final Gson GSON = new Gson()
            .newBuilder()
            .setPrettyPrinting()
            .excludeFieldsWithoutExposeAnnotation()
            .registerTypeAdapter(DenseLayer.class, LAYER_ADAPTER)
            .registerTypeAdapter(DropoutLayer.class, LAYER_ADAPTER)
            .registerTypeAdapter(Adam.class, OPTIMIZER_ADAPTER)
            .registerTypeAdapter(SGD.class, OPTIMIZER_ADAPTER)
            .create();

    private List<Layer> layers;
    private LossFunctions function;
    private Optimizer optimizer;
    private BackPropagation propagation;

    public FeedForwardModel(Layer... layers) {
        this.layers = new ArrayList<>(Arrays.asList(layers));
    }

    private void connect(InitializationType type) {
        for (int i = 0; i < layers.size() - 1; i++) {
            Layer layer = layers.get(i);

            if (layer instanceof DropoutLayer) continue;

            Layer nextLayer = layers.get(i + 1);

            if (nextLayer instanceof DropoutLayer) {
                nextLayer = layers.get(i + 2);
            }

            int nIn = layer.getNeurons().size();
            int nOut = nextLayer.getNeurons().size();

            double bound = type.getInitializer().getBound(nIn, nOut);

            layer.connectAll(nextLayer, bound);
        }
    }

    @Override
    public void compile(InitializationType type, LossFunctions function, Optimizer optimizer) {
        this.function = function;
        this.optimizer = optimizer;
        this.propagation = new BackPropagation(this, optimizer);

        connect(type);
    }

    @Override
    public void fit(DataSet set) {
        propagation.iterate(set, optimizer.getLearningRate());
    }

    @Override
    public double evaluate(DataSet set) {
        double totalError = 0.0;

        for (DataRow row : set.getDataRows()) {
            double[] inputs = row.inputs();
            double[] targets = row.outputs();

            double[] outputs = predict(inputs);

            totalError += function.getFunction().calculate(targets, outputs);
        }

        return totalError;
    }

    @Override
    public double[] predict(double ... input) {
        Layer inputLayer = layers.get(0);

        if (input.length != inputLayer.getNeurons().size()) {
            throw new IllegalArgumentException("Input size does not match model's input dimension! " +
                    input.length + " != " + inputLayer.getNeurons().size());
        }

        for (Layer layer : layers) {
            for (Neuron neuron : layer.getNeurons()) {
                neuron.setValue(0);
            }
        }

        for (int i = 0; i < input.length; i++) {
            inputLayer.getNeuronAt(i).setValue(input[i]);
        }

        for (int l = 0; l < layers.size() - 1; l++) {
            Layer layer = layers.get(l);

            if (layer instanceof DropoutLayer) continue;

            Layer nextLayer = layers.get(l + 1);

            if (nextLayer instanceof DropoutLayer) {
                nextLayer = layers.get(l + 2);
            }

            for (Synapse synapse : layer.getSynapses()) {
                Neuron inputNeuron = synapse.getInputNeuron();
                Neuron outputNeuron = synapse.getOutputNeuron();

                // Weighted sum
                outputNeuron.setValue(outputNeuron.getValue() + inputNeuron.getValue() * synapse.getWeight());
            }

            // Apply the activation function
            nextLayer.applyFunction();
        }

        Layer outputLayer = layers.get(layers.size() - 1);
        double[] output = new double[outputLayer.getNeurons().size()];

        for (int i = 0; i < output.length; i++) {
            output[i] = outputLayer.getNeuronAt(i).getValue();
        }

        return output;
    }

    @Override
    public LossFunction getLossFunction() {
        return function.getFunction();
    }

    @Override
    public List<Layer> getLayers() {
        return layers;
    }

    @Override
    public String getStats() {
        StringBuilder stats = new StringBuilder();
        stats.append(String.format("%-7s %-15s %-10s %-12s\n", "Index", "Layer name", "nIn, nOut", "TotalParams"));
        stats.append("================================================\n");

        int params = 0;

        for (int i = 0; i < layers.size(); i++) {
            Layer layer = layers.get(i);
            Layer next = layers.get(Math.min(i, layers.size() - 1));

            if (next instanceof DropoutLayer) {
                next = layers.get(Math.min(i + 1, layers.size() - 1));
            }

            String layerType = layer.getClass().getSimpleName();

            int nIn = layer.getNeurons().size();
            int nOut = i == layers.size() - 1 ? 0 : next.getNeurons().size();

            int totalParams = layer.getTotalParams();

            String formatNin = layer instanceof DropoutLayer ? "-" : String.valueOf(nIn);
            String formatNout = layer instanceof DropoutLayer ? "-" : String.valueOf(nOut);

            stats.append(String.format("%-7d %-15s %-10s %-12d\n",
                    i, layerType, formatNin + ", " + formatNout, totalParams));

            params += totalParams;
        }

        stats.append("================================================\n");
        stats.append("Total parameters: ").append(params).append("\n");
        stats.append("================================================\n");
        return stats.toString();
    }

    @Override
    public void load(String path) {
        File file = new File(path);

        if (!file.exists()) {
            throw new IllegalArgumentException("File does not exist: " + path);
        }

        try {
            JsonObject parent = JsonParser.parseReader(new FileReader(file)).getAsJsonObject();

            this.optimizer = GSON.fromJson(parent.get("optimizer"), Optimizer.class);
            this.function = LossFunctions.valueOf(parent.get("lossFunction").getAsString());

            Type listType = new TypeToken<ArrayList<Layer>>(){}.getType();

            this.layers = GSON.fromJson(parent.get("layers"), listType);

            connect(InitializationType.NORMAL);

            double[][] weights = GSON.fromJson(parent.get("weights"), double[][].class);

            for (int i = 0; i < weights.length; i++) {
                double[] layerWeights = weights[i];
                Layer layer = layers.get(i);

                for (int j = 0; j < layerWeights.length; j++) {
                    layer.getSynapses().get(j).setWeight(layerWeights[j]);
                }
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public void save(String path) {
        File file = new File(path);

        JsonObject parent = new JsonObject();
        JsonObject optimizerObject = GSON.toJsonTree(optimizer).getAsJsonObject();

        parent.addProperty("lossFunction", function.name());
        parent.add("optimizer", optimizerObject);

        List<JsonObject> layerObjects = new ArrayList<>();

        for (Layer layer : layers) {
            layerObjects.add(GSON.toJsonTree(layer).getAsJsonObject());
        }

        parent.add("layers", GSON.toJsonTree(layerObjects).getAsJsonArray());

        double[][] weights = new double[layers.size()][];

        for (int i = 0; i < layers.size(); i++) {
            Layer layer = layers.get(i);
            weights[i] = new double[layer.getSynapses().size()];

            for (int j = 0; j < layer.getSynapses().size(); j++) {
                Synapse synapse = layer.getSynapses().get(j);

                weights[i][j] = synapse.getWeight();
            }
        }

        parent.add("weights", GSON.toJsonTree(weights));

        try (BufferedWriter writer = new BufferedWriter(new FileWriter(file))) {
            GSON.toJson(parent, writer);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public void add(Layer layer) {
        layers.add(layer);
    }
}
