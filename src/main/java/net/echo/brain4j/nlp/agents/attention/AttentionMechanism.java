package net.echo.brain4j.nlp.agents.attention;

public interface AttentionMechanism {
    double[] attend(double[] input, String contextKey);
}

