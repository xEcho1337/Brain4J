package net.echo.brain4j.nlp.agents.impl;

import net.echo.brain4j.loss.LossFunctions;
import net.echo.brain4j.model.initialization.InitializationType;
import net.echo.brain4j.nlp.agents.Agent;
import net.echo.brain4j.nlp.agents.attention.AttentionMechanism;
import net.echo.brain4j.nlp.agents.encoding.PositionalEncoding;
import net.echo.brain4j.nlp.agents.model.TransformerModel;
import net.echo.brain4j.nlp.token.weight.TokenWeighter;
import net.echo.brain4j.training.data.DataSet;
import net.echo.brain4j.training.optimizers.impl.Adam;

import java.util.ArrayList;
import java.util.List;

public class ChatAgent implements Agent {
    double temperature, topK;
    private final TransformerModel model;
    private final TokenWeighter weighter;
    private final PositionalEncoding encoder;
    private final AttentionMechanism attentionMechanism;
    private final int contextWindow;
    private final List<String> conversationHistory;

    public ChatAgent(AttentionMechanism attentionMechanism, int contextWindow, int embeddingDim, double temperature, double topK) {
        this.temperature = temperature;
        this.topK = topK;
        this.attentionMechanism = attentionMechanism;
        this.contextWindow = contextWindow;
        this.model = new TransformerModel(contextWindow, 128, embeddingDim, temperature, topK);
        this.weighter = new TokenWeighter(0.1);
        this.encoder = new PositionalEncoding(contextWindow, embeddingDim);
        this.conversationHistory = new ArrayList<>();

        initializeModel();
    }

    private void initializeModel() {
        model.compile(
                InitializationType.XAVIER,
                LossFunctions.MEAN_SQUARED_ERROR,
                new Adam(0.001)
        );
    }

    public String generateResponse(String userInput) {
        String processedInput = preprocessInput(userInput);
        double[] encodedInput = processInput(processedInput);

        String contextKey = String.valueOf(conversationHistory.size());
        double[] attendedInput = attentionMechanism.attend(encodedInput, contextKey);

        double[] modelOutput = model.predict(attendedInput);
        String response = decodeResponse(modelOutput);
        updateContext(userInput, response);

        return formatResponse(response);
    }

    private String preprocessInput(String input) {
        return input.toLowerCase()
                .replaceAll("[^a-z0-9\\s]", "")
                .trim();
    }

    private String formatResponse(String response) {
        return response.substring(0, 1).toUpperCase() +
                response.substring(1) +
                (response.endsWith(".") ? "" : ".");
    }

    @Override
    public String process(String input) {
        updateContext(input);
        double[] weightedInput = processInput(input);
        double[] response = model.predict(weightedInput);
        String output = decodeResponse(response);
        updateContext(output);
        return output;
    }

    @Override
    public void train(DataSet conversationData) {
        int maxEpochs = 150;  // Further reduced for testing
        double errorThreshold = 0.001;

        System.out.println("Starting training loop");
        for(int epoch = 0; epoch < maxEpochs; epoch++) {
            System.out.printf("Epoch %d/%d%n", epoch + 1, maxEpochs);
            model.fit(conversationData);
            double error = model.evaluate(conversationData);
            if (Double.isNaN(error)) {
                throw new RuntimeException("Error is NaN");
            }

            System.out.printf("Error: %.4f%n", error);
            if(error < errorThreshold) break;
        }
    }

    @Override
    public double evaluate(DataSet testData) {
        return model.evaluate(testData);
    }

    @Override
    public void save(String path) {
        model.save(path);
    }

    @Override
    public void load(String path) {
        model.load(path);
    }

    private void updateContext(String text) {
        conversationHistory.add(text);
        if (conversationHistory.size() > contextWindow) {
            conversationHistory.remove(0);
        }
    }

    private void updateContext(String userInput, String response) {
        updateContext("User: " + userInput);
        updateContext("Assistant: " + response);
    }

    private double[] processInput(String input) {
        String[] tokens = input.split("\\s+");
        double[] weighted = new double[contextWindow];

        for (int i = 0; i < tokens.length && i < contextWindow; i++) {
            double weight = weighter.getWeight(tokens[i]);
            double[] posEncoded = encoder.encode(new double[]{weight}, i);
            weighted[i] = posEncoded[0];
        }

        return weighted;
    }

    private String decodeResponse(double[] response) {
        StringBuilder output = new StringBuilder();
        for (double value : response) {
            int index = (int) Math.round(value);
            if (index >= 0 && index < 26) {
                output.append((char) (index + 'a'));
            }
        }
        return output.toString();
    }

    public TransformerModel getModel() {
        return model;
    }
}
