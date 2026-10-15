package org.brain4j.core.codec.layer.transformer;

import com.fasterxml.jackson.databind.JsonNode;

import org.brain4j.core.layer.impl.transformer.Transformer;

/**
 * Codec for {@link Transformer.Decoder}.
 */
public class TransformerDecoderCodec implements TransformerCodec<Transformer.Decoder> {

    /**
     * {@inheritDoc}
     */
    @Override
    public String type() {
        return "transformer_decoder";
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Class<Transformer.Decoder> targetClass() {
        return Transformer.Decoder.class;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Transformer.Decoder parse(JsonNode in) {
        return new Transformer.Decoder(readConfig(in));
    }
}
