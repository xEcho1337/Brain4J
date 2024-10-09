package net.echo.brain4j.initialization;

public interface WeightInitializer {

    double SQRT_OF_6 = Math.sqrt(6);

    double getBound(int nIn, int nOut);
}
