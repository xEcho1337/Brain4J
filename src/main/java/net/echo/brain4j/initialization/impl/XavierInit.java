package net.echo.brain4j.initialization.impl;

import net.echo.brain4j.initialization.WeightInitializer;

public class XavierInit implements WeightInitializer {

    @Override
    public double getBound(int nIn, int nOut) {
        return SQRT_OF_6 / Math.sqrt(nIn + nOut);
    }
}
