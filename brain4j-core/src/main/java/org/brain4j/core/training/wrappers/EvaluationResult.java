package org.brain4j.core.training.wrappers;

import java.util.Map;

import org.brain4j.math.Tensors;
import org.brain4j.math.commons.Commons;
import org.brain4j.math.tensor.Tensor;

/**
 * Holds loss and classification metrics for an evaluated dataset.
 * Use it to inspect accuracy, precision, recall and confusion data.
 */
public class EvaluationResult {

    private final double loss;
    private final int classes;
    private final Map<Integer, Tensor> classifications;

    private int totalCorrect;
    private int totalIncorrect;
    private double accuracy;
    private double precision;
    private double recall;
    private double f1Score;
    private double totalDeviation;

    /**
     * Creates a result and derives metrics eagerly.
     */
    public EvaluationResult(double loss, int classes, Map<Integer, Tensor> classifications) {
        this.loss = loss;
        this.classes = classes;
        this.classifications = classifications;
        calculateStats();
    }

    private void calculateStats() {
        int[] truePositives = new int[classes];
        int[] falsePositives = new int[classes];
        int[] falseNegatives = new int[classes];

        for (int i = 0; i < classifications.size(); i++) {
            Tensor vector = classifications.get(i);

            for (int j = 0; j < vector.elements(); j++) {
                int value = (int) vector.get(j);

                if (i == j) {
                    totalCorrect += value;
                    truePositives[i] += value;
                } else {
                    totalIncorrect += value;
                    falsePositives[j] += value;
                    falseNegatives[i] += value;
                }
            }
        }

        double precisionSum = 0, recallSum = 0;

        for (int i = 0; i < classes; i++) {
            double precision = (truePositives[i] + falsePositives[i]) > 0 ?
                    (double) truePositives[i] / (truePositives[i] + falsePositives[i]) : 0;

            double recall = (truePositives[i] + falseNegatives[i]) > 0 ?
                    (double) truePositives[i] / (truePositives[i] + falseNegatives[i]) : 0;

            precisionSum += precision;
            recallSum += recall;
        }

        this.accuracy = (double) totalCorrect / (totalCorrect + totalIncorrect);
        this.precision = precisionSum / classes;
        this.recall = recallSum / classes;
        this.f1Score = (precision + recall) > 0 ? 2 * (precision * recall) / (precision + recall) : 0;
        this.totalDeviation = calculateDeviation();
    }

    private double calculateDeviation() {
        double sum = 0.0;
        double total = 0.0;
        int count = 0;

        for (Tensor tensor : classifications.values()) {
            for (int i = 0; i < tensor.elements(); i++) {
                total += tensor.get(i);
                count++;
            }
        }

        if (count == 0) {
            return 0.0;
        }

        double mean = total / count;

        for (Tensor tensor : classifications.values()) {
            for (int i = 0; i < tensor.elements(); i++) {
                double diff = tensor.get(i) - mean;
                sum += diff * diff;
            }
        }

        return sum;
    }

    /**
     * Formats loss, metrics and confusion matrix for console output.
     *
     * @return the formatted report
     */
    public String results() {
        StringBuilder matrix = new StringBuilder();
        String divider = Commons.getHeader(" Evaluation Results ", Commons.HEADER_CHAR);

        matrix.append(divider);
        matrix.append("Out of ").append(classifications.size()).append(" classes\n\n");

        String secondary = "%-12s %-10s\n";
        matrix.append(secondary.formatted("Loss:", "%.4f".formatted(loss)));
        matrix.append(secondary.formatted("Accuracy:", "%.4f".formatted(accuracy)));
        matrix.append(secondary.formatted("Precision:", "%.4f".formatted(precision)));
        matrix.append(secondary.formatted("Recall:", "%.4f".formatted(recall)));
        matrix.append(secondary.formatted("F1-score:", "%.4f".formatted(f1Score)));

        if (!classifications.isEmpty()) {
            divider = Commons.getHeader(" Confusion Matrix ", Commons.HEADER_CHAR);
            matrix.append(divider);
            matrix.append("First column is the actual class, top row are the predicted classes.\n\n");

            int maxValue = 0;

            for (Tensor tensor : classifications.values()) {
                for (int i = 0; i < tensor.elements(); i++) {
                    maxValue = Math.max(maxValue, (int) tensor.get(i));
                }
            }

            int cellWidth = Math.max(String.valueOf(maxValue).length(), 5) + 1;

            matrix.repeat(" ", 7);

            for (int i = 0; i < classes; i++) {
                matrix.append(("%" + cellWidth + "d").formatted(i));
            }

            matrix.append("\n  ");
            matrix.repeat("-", 6 + cellWidth * classes).append("\n");

            for (int i = 0; i < classes; i++) {
                matrix.append("%4d | ".formatted(i));
                Tensor predictions = classifications.get(i);

                for (int j = 0; j < predictions.elements(); j++) {
                    int prediction = (int) predictions.get(j);
                    matrix.append(("%" + cellWidth + "d").formatted(prediction));
                }

                matrix.append("\n");
            }

            matrix.append("\n");
        }

        matrix.append(Commons.getHeader("", Commons.HEADER_CHAR));

        return matrix.toString();
    }

    /**
     * Builds the confusion matrix tensor.
     *
     * @return a {@code classes x classes} tensor
     */
    public Tensor confusionMatrix() {
        Tensor result = Tensors.matrix(classes, classes);

        for (int i = 0; i < classes; i++) {
            Tensor predictions = classifications.get(i);

            for (int j = 0; j < predictions.elements(); j++) {
                result.set(predictions.get(j), i, j);
            }
        }

        return result;
    }

    /**
     * Returns per-class prediction counts.
     *
     * @return the classification map
     */
    public Map<Integer, Tensor> classifications() {
        return classifications;
    }

    /**
     * Returns the average loss.
     *
     * @return the loss value
     */
    public double loss() {
        return loss;
    }

    /**
     * Returns the class count.
     *
     * @return the number of classes
     */
    public int classes() {
        return classes;
    }

    /**
     * Returns the correct prediction count.
     *
     * @return the correct total
     */
    public int totalCorrect() {
        return totalCorrect;
    }

    /**
     * Returns the incorrect prediction count.
     *
     * @return the incorrect total
     */
    public int totalIncorrect() {
        return totalIncorrect;
    }

    /**
     * Returns the overall accuracy.
     *
     * @return the accuracy in [0, 1]
     */
    public double accuracy() {
        return accuracy;
    }

    /**
     * Returns the macro-averaged precision.
     *
     * @return the precision in [0, 1]
     */
    public double precision() {
        return precision;
    }

    /**
     * Returns the macro-averaged recall.
     *
     * @return the recall in [0, 1]
     */
    public double recall() {
        return recall;
    }

    /**
     * Returns the F1 score from precision and recall.
     *
     * @return the F1 score in [0, 1]
     */
    public double f1Score() {
        return f1Score;
    }

    /**
     * Returns the squared deviation of confusion counts.
     *
     * @return the total deviation
     */
    public double totalDeviation() {
        return totalDeviation;
    }
}
