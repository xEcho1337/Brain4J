package net.echo.brain4j.initialization;

import net.echo.brain4j.initialization.impl.HeInit;
import net.echo.brain4j.initialization.impl.NormalInit;
import net.echo.brain4j.initialization.impl.XavierInit;

public enum InitializationType {

    NORMAL(new NormalInit()),
    HE(new HeInit()),
    XAVIER(new XavierInit());

    private final WeightInitializer initializer;

    InitializationType(WeightInitializer initializer) {
        this.initializer = initializer;
    }

    public WeightInitializer getInitializer() {
        return initializer;
    }
}
