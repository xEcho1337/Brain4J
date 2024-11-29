package net.echo.brain4j.nlp.agents.attention.score;

import java.util.HashMap;
import java.util.Map;

public class AttentionScorer {
    private final double temperature;
    private final double topK;
    private final Map<String, double[]> attentionCache;

    public AttentionScorer(double temperature, double topK) {
        this.temperature = temperature;
        this.topK = topK;
        this.attentionCache = new HashMap<>();
    }

    public double[] score(double[] query, double[] key, String contextKey) {
        double[] cachedScore = attentionCache.get(contextKey);
        if (cachedScore != null) return cachedScore;

        double[] scores = computeAttentionScores(query, key);
        attentionCache.put(contextKey, scores);
        return scores;
    }

    private double[] computeAttentionScores(double[] query, double[] key) {
        double[] scores = new double[query.length];
        double maxScore = Double.NEGATIVE_INFINITY;

        for (int i = 0; i < query.length; i++) {
            scores[i] = (query[i] * key[i]) / Math.sqrt(query.length);
            maxScore = Math.max(maxScore, scores[i]);
        }

        double sum = 0.0;
        for (int i = 0; i < scores.length; i++) {
            scores[i] = Math.exp((scores[i] - maxScore) / temperature);
            sum += scores[i];
        }

        for (int i = 0; i < scores.length; i++) {
            scores[i] = scores[i] / sum;
            if (scores[i] < topK) {
                scores[i] = 0;
            }
        }

        return scores;
    }

    public void clearCache() {
        attentionCache.clear();
    }
}

