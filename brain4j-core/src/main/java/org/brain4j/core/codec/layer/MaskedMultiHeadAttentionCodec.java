package org.brain4j.core.codec.layer;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import org.brain4j.core.codec.JsonCodec;
import org.brain4j.core.layer.impl.transformer.attention.MaskedMultiHeadAttention;

/**
 * Codec for {@link MaskedMultiHeadAttention}.
 */
public class MaskedMultiHeadAttentionCodec implements JsonCodec<MaskedMultiHeadAttention> {

    /**
     * {@inheritDoc}
     */
    @Override
    public String type() {
        return "masked_multi_head_attention";
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Class<MaskedMultiHeadAttention> targetClass() {
        return MaskedMultiHeadAttention.class;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void write(MaskedMultiHeadAttention layer, ObjectNode out) {
        out.put("head_count", layer.config().heads());
        out.put("embedding_dim", layer.config().embedDim());
        out.put("qkv_bias", layer.config().qkvBias());
        out.put("out_bias", layer.config().outBias());
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public MaskedMultiHeadAttention parse(JsonNode in) {
        int heads = in.get("head_count").asInt();
        int dim = in.get("embedding_dim").asInt();

        boolean qkv = in.get("qkv_bias").asBoolean();
        boolean out = in.get("out_bias").asBoolean();

        var config = new MaskedMultiHeadAttention.Config(dim, heads, qkv, out);
        return new MaskedMultiHeadAttention(config);
    }
}
