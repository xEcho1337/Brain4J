package net.echo.brain4j.model.initialization;

import net.echo.brain4j.model.initialization.impl.HeInit;
import net.echo.brain4j.model.initialization.impl.LeCunInit;
import net.echo.brain4j.model.initialization.impl.NormalInit;
import net.echo.brain4j.model.initialization.impl.XavierInit;

public enum InitializationType {

    NORMAL(new NormalInit()),
    HE(new HeInit()),
    XAVIER(new XavierInit()),
    LECUN(new LeCunInit()),;

    private final WeightInitializer initializer;

    InitializationType(WeightInitializer initializer) {
        this.initializer = initializer;
    }

    public WeightInitializer getInitializer() {
        return initializer;
    }
}
