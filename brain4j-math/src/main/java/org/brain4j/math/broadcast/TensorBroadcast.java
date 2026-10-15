package org.brain4j.math.broadcast;

import org.brain4j.math.broadcast.impl.BroadcastAdd;
import org.brain4j.math.broadcast.impl.BroadcastDiv;
import org.brain4j.math.broadcast.impl.BroadcastMul;
import org.brain4j.math.broadcast.impl.BroadcastPow;
import org.brain4j.math.broadcast.impl.BroadcastSub;
import org.brain4j.math.tensor.Tensor;

/**
 * Entry point for broadcast arithmetic. Use it to add, subtract, or scale mismatched tensors.
 */
public class TensorBroadcast {

    private final static BroadcastAdd ADD_OP = new BroadcastAdd();
    private final static BroadcastSub SUB_OP = new BroadcastSub();
    private final static BroadcastMul MUL_OP = new BroadcastMul();
    private final static BroadcastDiv DIV_OP = new BroadcastDiv();
    private final static BroadcastPow POW_OP = new BroadcastPow();

    /**
     * Adds B into A with broadcasting.
     *
     * @return the mutated tensor A
     */
    public static Tensor add(Tensor A, Tensor B) {
        return ADD_OP.defaultOp(A, B);
    }

    /**
     * Subtracts B from A with broadcasting.
     *
     * @return the mutated tensor A
     */
    public static Tensor sub(Tensor A, Tensor B) {
        return SUB_OP.defaultOp(A, B);
    }

    /**
     * Multiplies A by B with broadcasting.
     *
     * @return the mutated tensor A
     */
    public static Tensor mul(Tensor A, Tensor B) {
        return MUL_OP.defaultOp(A, B);
    }

    /**
     * Divides A by B with broadcasting.
     *
     * @return the mutated tensor A
     */
    public static Tensor div(Tensor A, Tensor B) {
        return DIV_OP.defaultOp(A, B);
    }

    /**
     * Raises A to B with broadcasting.
     *
     * @return the mutated tensor A
     */
    public static Tensor pow(Tensor A, Tensor B) {
        return POW_OP.defaultOp(A, B);
    }

    /**
     * Runs a custom broadcast op on A and B.
     *
     * @return the mutated tensor A
     */
    public static Tensor forward(BroadcastOperation operation, Tensor A, Tensor B) {
        return operation.defaultOp(A, B);
    }
}
