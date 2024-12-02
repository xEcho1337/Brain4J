package net.echo.brain4j.utils;

public class Vector {

    private final double[] data;

    public Vector(int size) {
        this.data = new double[size];
    }

    public Vector(double... data) {
        this.data = data;
    }

    public void set(int index, double value) {
        data[index] = value;
    }

    public double get(int index) {
        return data[index];
    }

    public double lengthSquared() {
        double sum = 0;

        for (double value : data) {
            sum += value * value;
        }

        return sum;
    }

    public double length() {
        return Math.sqrt(lengthSquared());
    }

    public double sum() {
        double sum = 0;

        for (double value : data) {
            sum += value;
        }

        return sum;
    }

    public Vector normalizeSquared() {
        double length = lengthSquared();

        for (int i = 0; i < data.length; i++) {
            data[i] = data[i] / length;
        }

        return this;
    }

    public Vector normalize() {
        double length = length();

        for (int i = 0; i < data.length; i++) {
            data[i] = data[i] / length;
        }

        return this;
    }

    public double distanceSquared(Vector vector) {
        if (data.length != vector.data.length) {
            throw new IllegalArgumentException("Vectors must be of the same length.");
        }

        double sum = 0;

        for (int i = 0; i < data.length; i++) {
            sum += (data[i] - vector.data[i]) * (data[i] - vector.data[i]);
        }

        return sum;
    }

    public double distance(Vector vector) {
        return Math.sqrt(distanceSquared(vector));
    }

    public Vector convoluted(Vector other) {
        double[] result = new double[data.length + other.data.length - 1];

        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < other.data.length; j++) {
                result[i + j] += data[i] * other.data[j];
            }
        }

        return new Vector(result);
    }

    public double[] toArray() {
        return data;
    }

    public void add(Vector other) {
        for (int i = 0; i < data.length; i++) {
            data[i] += other.data[i];
        }
    }

    public void scale(double value) {
        for (int i = 0; i < data.length; i++) {
            data[i] *= value;
        }
    }
}
