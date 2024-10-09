package net.echo.brain4j.model;

import net.echo.brain4j.data.DataSet;
import net.echo.brain4j.initialization.InitializationType;
import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.loss.LossFunction;
import net.echo.brain4j.loss.LossFunctions;

import java.util.List;

public interface Model {

    void compile(InitializationType type, LossFunctions function);

    double fit(DataSet set, double learningRate);

    double[] predict(double ... input);

    LossFunction getLossFunction();

    List<Layer> getLayers();

    String getStats();
}
