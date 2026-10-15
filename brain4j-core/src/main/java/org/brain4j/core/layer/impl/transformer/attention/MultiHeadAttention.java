package org.brain4j.core.layer.impl.transformer.attention;

import java.util.List;
import java.util.random.RandomGenerator;

import org.brain4j.core.layer.Layer;
import org.brain4j.math.Tensors;
import org.brain4j.math.activation.impl.Softmax;
import org.brain4j.math.commons.Commons;
import org.brain4j.math.commons.Range;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.tensor.Shape;
import org.brain4j.math.tensor.Tensor;

/**
 * Computes scaled dot-product attention over multiple heads.
 * Use it in encoders where every position may attend to all others.
 */
public class MultiHeadAttention extends Layer {

    protected int headDim;
    protected Config config;

    /**
     * Holds embedding width, head count and bias flags.
     */
    public record Config(int embedDim, int heads, boolean qkvBias, boolean outBias) {
        /**
         * Return the per-head dimension.
         *
         * @return the head size
         */
        public int headDim() {
            return embedDim / heads;
        }
    }

    /**
     * Create attention without biases.
     */
    public MultiHeadAttention(int heads, int embedDim) {
        this(new Config(embedDim, heads, false, false));
    }

    /**
     * Create attention with the given config.
     */
    public MultiHeadAttention(Config config) {
        if (config.embedDim % config.heads != 0) {
            throw Commons.illegalArgument(
                    "Embedding dimension must be divisible by head count! (%s %% %s = %s)",
                    config.embedDim, config.heads, config.embedDim % config.heads
            );
        }

        this.headDim = config.embedDim / config.heads;
        this.config = config;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void build(List<Shape> inputShapes) {
        Shape inputShape = inputShapes.getFirst();

        if (inputShape.rank() != 2) {
            throw Commons.illegalArgument("Attention requires tensors with rank 2 but %s were given!", inputShape.rank());
        }

        if (inputShape.last() != config.embedDim) {
            throw Commons.illegalArgument("Expected embedding dim %s but got %s", config.embedDim, inputShape.last());
        }

        registerParam("weights", Tensors.zeros(config.embedDim, 3 * config.embedDim));
        registerParam("out_proj", Tensors.zeros(config.embedDim, config.embedDim));

        if (config.qkvBias) {
            registerParam("bias", Tensors.zeros(3 * config.embedDim));
        }

        if (config.outBias) {
            registerParam("out_bias", Tensors.zeros(config.embedDim));
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void initWeights(List<Shape> inputShapes, RandomGenerator rng) {
        generateWeights("weights", rng, config.embedDim, 3 * config.embedDim);
        generateWeights("out_proj", rng, config.embedDim, config.embedDim);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public List<Shape> inferOutputShapes(List<Shape> inputShapes) {
        if (inputShapes.size() != 1) {
            throw Commons.illegalArgument("Layer requires 1 input but %s were given!", inputShapes.size());
        }

        Shape inputShape = inputShapes.getFirst();

        if (inputShape.rank() != 2) {
            throw Commons.illegalArgument("Attention requires tensors with rank 2 but %s were given!", inputShape.rank());
        }

        return List.of(inputShape);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor[] forward(StatesCache cache, Tensor... inputs) {
        Tensor input = inputs[0];
        int batch = input.shapeAt(0);
        int seqLength = input.shapeAt(1);

        Tensor weights = getParam("weights");
        Tensor outProj = getParam("out_proj");
        Tensor bias = getParam("bias");
        Tensor outBias = getParam("out_bias");
        Tensor qkv = input.matmulGrad(weights);

        if (config.qkvBias) {
            qkv = qkv.addGrad(bias);
        }

        Tensor reshaped = qkv.reshapeGrad(batch, seqLength, config.heads, 3, headDim);
        reshaped = reshaped.transposeGrad(1, 2);

        Tensor[] qkvs = new Tensor[3];
        Range all = Range.all();

        for (int i = 0; i < qkvs.length; i++) {
            qkvs[i] = reshaped.sliceGrad(all, all, all, Range.point(i), all);
            qkvs[i] = qkvs[i].squeezeGrad(3);
        }

        Tensor q = qkvs[0];
        Tensor k = qkvs[1];
        Tensor v = qkvs[2];

        double normalizer = Math.sqrt(headDim);

        Tensor kT = k.transposeGrad();
        Tensor scores = q.matmulGrad(kT).divGrad(Tensors.scalar(normalizer));
        Tensor attentionWeights = scores.activateGrad(new Softmax());
        Tensor context = attentionWeights.matmulGrad(v);
        context = context.transposeGrad(1, 2);
        Tensor output = context.reshapeGrad(batch, seqLength, config.embedDim);

        Tensor result = output.matmulGrad(outProj);

        if (config.outBias) {
            result = result.addGrad(outBias);
        }

        return new Tensor[]{result};
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Layer copy() {
        MultiHeadAttention copy = new MultiHeadAttention(config);
        copyParameters(copy);

        return copy;
    }

    /**
     * Return the layer config.
     *
     * @return the config
     */
    public Config config() {
        return config;
    }
}
