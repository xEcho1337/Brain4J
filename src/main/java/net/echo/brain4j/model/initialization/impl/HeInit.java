package net.echo.brain4j.model.initialization.impl;

import net.echo.brain4j.model.initialization.WeightInitializer;

public class HeInit implements WeightInitializer {

    @Override
    public double getBound(int nIn, int nOut) {
        return SQRT_OF_6 / Math.sqrt(nIn);
    }
}
