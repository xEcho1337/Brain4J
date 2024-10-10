package net.echo.brain4j.model;

import net.echo.brain4j.training.data.DataSet;
import net.echo.brain4j.model.initialization.InitializationType;
import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.loss.LossFunction;
import net.echo.brain4j.loss.LossFunctions;
import net.echo.brain4j.training.optimizers.Optimizer;

import java.util.List;

public interface Model {

    void compile(InitializationType type, LossFunctions function, Optimizer optimizer);

    double fit(DataSet set);

    double[] predict(double ... input);

    LossFunction getLossFunction();

    List<Layer> getLayers();

    String getStats();
}
