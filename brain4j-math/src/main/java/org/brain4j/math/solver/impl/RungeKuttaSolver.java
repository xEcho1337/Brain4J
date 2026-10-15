package org.brain4j.math.solver.impl;

import java.util.function.Function;

import org.brain4j.math.activation.Activations;
import org.brain4j.math.solver.NumericalSolver;
import org.brain4j.math.tensor.Tensor;

/**
 * Classical fourth-order Runge-Kutta solver. Use it for accurate stable ODE steps.
 */
public class RungeKuttaSolver implements NumericalSolver {

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor update(Tensor deltaTimestep, Tensor tauT, Tensor projInput,
                         Tensor hidden, Function<Tensor, Tensor> hiddenFunction) {

        Tensor alpha = deltaTimestep.broadcastLike(tauT).divGrad(tauT);
        Tensor k1 = computeF(hidden, projInput, hiddenFunction);
        Tensor k2 = computeF(hidden.addGrad(alpha.times(k1).times(0.5f)),
            projInput, hiddenFunction);
        Tensor k3 = computeF(hidden.addGrad(alpha.times(k2).times(0.5f)),
            projInput, hiddenFunction);
        Tensor k4 = computeF(hidden.addGrad(alpha.times(k3)),
            projInput, hiddenFunction);
        Tensor increment = k1.addGrad(k2.times(2)).addGrad(k3.times(2)).addGrad(k4)
            .times(1.0f / 6.0f);
        hidden = hidden.addGrad(alpha.times(increment));

        return hidden;
    }

    private Tensor computeF(Tensor h, Tensor projInput, Function<Tensor, Tensor> hiddenFunction) {
        Tensor projHidden = hiddenFunction.apply(h);
        Tensor z = projInput.addGrad(projHidden).activateGrad(Activations.TANH.function());

        return z.addGrad(h.times(-1));
    }
}
