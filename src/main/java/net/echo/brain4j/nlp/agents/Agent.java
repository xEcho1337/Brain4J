package net.echo.brain4j.nlp.agents;

import net.echo.brain4j.training.data.DataSet;

public interface Agent {
    String process(String input);
    void train(DataSet conversationData);
    double evaluate(DataSet testData);
    void save(String path);
    void load(String path);
}
