package org.brain4j.math.solver;

import java.util.function.Function;

import org.brain4j.math.solver.impl.BogackiShampineSolver;
import org.brain4j.math.solver.impl.EulerSolver;
import org.brain4j.math.solver.impl.RungeKuttaSolver;
import org.brain4j.math.solver.utils.StepResult;
import org.brain4j.math.tensor.Tensor;

/**
 * Advances ODE hidden states through time.
 * Use it for liquid networks trading speed against accuracy.
 */
public interface NumericalSolver {

    /**
     * Advances the hidden state by one integration step.
     *
     * @param deltaT timestep delta, shape [batch, 1]
     * @param tauT projected time constant, shape [batch, hidden_dim]
     * @param projInput projected input for this step, shape [batch, hidden_dim]
     * @param hidden current hidden state
     * @param hiddenFunction projects the hidden state
     * @return the next hidden state
     */
    Tensor update(Tensor deltaT, Tensor tauT, Tensor projInput, Tensor hidden, Function<Tensor, Tensor> hiddenFunction);

    /**
     * Advances the hidden state with error estimation and step control.
     * Rejects the step when the error exceeds tolerance.
     *
     * @param deltaT timestep delta, shape [batch, 1]
     * @param tauT projected time constant, shape [batch, hidden_dim]
     * @param projInput projected input for this step, shape [batch, hidden_dim]
     * @param hidden current hidden state
     * @param hiddenFunction projects the hidden state
     * @param tolerance error tolerance for step acceptance
     * @return the next state, recommended timestep, and acceptance flag
     */
    default StepResult updateAdaptive(Tensor deltaT, Tensor tauT, Tensor projInput,
                                      Tensor hidden, Function<Tensor, Tensor> hiddenFunction,
                                      float tolerance) {
        Tensor nextHidden = update(deltaT, tauT, projInput, hidden, hiddenFunction);

        return StepResult.accepted(nextHidden, deltaT);
    }

    /**
     * Clears solver caches before a new sequence or batch size.
     */
    default void resetCache() {
    }
}
