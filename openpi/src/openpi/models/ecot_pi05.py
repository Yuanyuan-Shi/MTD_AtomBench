"""Reasoning-capable pi0.5 while preserving the upstream flow objective.

The first Gemma expert generates released ECoT text autoregressively.  The
second expert predicts continuous actions and can attend to that text.  No
reasoning/action alignment loss is implemented here.
"""

import dataclasses

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0
from openpi.models import pi0_config
from openpi.shared import array_typing as at


EOS_TOKEN = 1
IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb")


@dataclasses.dataclass(frozen=True)
class EcotPi05Config(pi0_config.Pi0Config):
    """Shape-compatible pi0.5 config with ECoT methods enabled."""

    @override
    def create(self, rng: at.KeyArrayLike) -> "EcotPi05":
        return EcotPi05(self, rngs=nnx.Rngs(rng))


class EcotPi05(pi0.Pi0):
    def _embed_ecot_prefix(self, obs: _model.Observation):
        if obs.tokenized_prompt is None or obs.tokenized_prompt_mask is None:
            raise ValueError("ECoT token sequence and mask are required")
        if obs.token_ar_mask is None:
            raise ValueError("ECoT autoregressive mask is required")

        tokens = []
        masks = []
        ar_masks = []
        image_tokens = 0
        for name in IMAGE_KEYS:
            embedded, _ = self.PaliGemma.img(obs.images[name], train=False)
            tokens.append(embedded)
            masks.append(einops.repeat(obs.image_masks[name], "b -> b s", s=embedded.shape[1]))
            ar_masks.append(jnp.zeros(embedded.shape[:2], dtype=jnp.bool_))
            image_tokens += embedded.shape[1]

        text = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
        tokens.append(text)
        masks.append(obs.tokenized_prompt_mask)
        ar_masks.append(obs.token_ar_mask.astype(jnp.bool_))
        return (
            jnp.concatenate(tokens, axis=1),
            jnp.concatenate(masks, axis=1),
            jnp.concatenate(ar_masks, axis=1),
            image_tokens,
        )

    def _reasoning_nll(self, prefix_out, image_tokens, obs, target_local_ids, vocabulary_ids):
        # Hidden token i predicts token i+1.  Decode only against the audited
        # annotation vocabulary, avoiding a 257k-way dense tensor.
        text_out = prefix_out[:, image_tokens : image_tokens + obs.tokenized_prompt.shape[1]]
        logits = self.PaliGemma.llm(text_out[:, :-1], vocabulary_ids, method="decode_subset")
        targets = jnp.clip(target_local_ids[:, 1:], 0)
        selected = jnp.take_along_axis(jax.nn.log_softmax(logits, axis=-1), targets[..., None], axis=-1)[..., 0]
        mask = obs.token_loss_mask[:, 1:].astype(logits.dtype)
        return -jnp.sum(selected * mask, axis=-1) / jnp.maximum(jnp.sum(mask, axis=-1), 1)

    def compute_reasoning_loss(
        self, rng, observation, target_local_ids, vocabulary_ids, *, train=False
    ):
        observation = _model.preprocess_observation(
            rng, observation, train=train, image_keys=IMAGE_KEYS
        )
        prefix, mask, ar_mask, image_tokens = self._embed_ecot_prefix(observation)
        attention = pi0.make_attn_mask(mask, ar_mask)
        positions = jnp.cumsum(mask, axis=1) - 1
        (prefix_out, _), _ = self.PaliGemma.llm(
            [prefix, None], mask=attention, positions=positions, adarms_cond=[None, None]
        )
        return self._reasoning_nll(
            prefix_out, image_tokens, observation, target_local_ids, vocabulary_ids
        )

    def compute_joint_loss(
        self, rng, observation, actions, target_local_ids, vocabulary_ids, *, train=False
    ):
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(
            preprocess_rng, observation, train=train, image_keys=IMAGE_KEYS
        )
        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        expanded = time[..., None, None]
        x_t = expanded * noise + (1 - expanded) * actions
        target_velocity = noise - actions

        prefix, prefix_mask, prefix_ar, image_tokens = self._embed_ecot_prefix(observation)
        suffix, suffix_mask, suffix_ar, adarms = self.embed_suffix(observation, x_t, time)
        full_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        full_ar = jnp.concatenate([prefix_ar, jnp.broadcast_to(suffix_ar, suffix_mask.shape)], axis=1)
        attention = pi0.make_attn_mask(full_mask, full_ar)
        positions = jnp.cumsum(full_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix, suffix], mask=attention, positions=positions, adarms_cond=[None, adarms]
        )
        velocity = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        flow = jnp.mean(jnp.square(velocity - target_velocity), axis=-1)
        reason = self._reasoning_nll(
            prefix_out, image_tokens, observation, target_local_ids, vocabulary_ids
        )
        return reason, flow

    def compute_conditioned_flow_loss(self, rng, observation, actions, *, train=False):
        """Unchanged flow matching conditioned on the teacher-forced ECoT sequence."""
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(
            preprocess_rng, observation, train=train, image_keys=IMAGE_KEYS
        )
        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        expanded = time[..., None, None]
        x_t = expanded * noise + (1 - expanded) * actions
        target_velocity = noise - actions
        prefix, prefix_mask, prefix_ar, _ = self._embed_ecot_prefix(observation)
        suffix, suffix_mask, suffix_ar, adarms = self.embed_suffix(observation, x_t, time)
        full_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        full_ar = jnp.concatenate(
            [prefix_ar, jnp.broadcast_to(suffix_ar, suffix_mask.shape)], axis=1
        )
        attention = pi0.make_attn_mask(full_mask, full_ar)
        positions = jnp.cumsum(full_mask, axis=1) - 1
        (_, suffix_out), _ = self.PaliGemma.llm(
            [prefix, suffix], mask=attention, positions=positions, adarms_cond=[None, adarms],
            # Stage 2 must expose the action expert to the learned ECoT/VLM
            # representation. The action expert itself remains frozen.
            lora_enabled=[True, False],
        )
        velocity = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        return jnp.mean(jnp.square(velocity - target_velocity), axis=-1)

    def generate_reasoning(self, observation, vocabulary_ids, *, max_decoding_steps=416):
        """Greedy constrained-vocabulary ECoT generation with a fixed KV cache.

        The static default complements the fixed 160-token inference prefix and
        keeps inference inside the 576-token context used during training.
        """
        observation = _model.preprocess_observation(
            None, observation, train=False, image_keys=IMAGE_KEYS
        )
        prefix, prefix_mask, prefix_ar, _ = self._embed_ecot_prefix(observation)
        attention = pi0.make_attn_mask(prefix_mask, prefix_ar)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_out, _), cache = self.PaliGemma.llm(
            [prefix, None], mask=attention, positions=positions, adarms_cond=[None, None]
        )
        # Inference prefixes are right-aligned, so the final physical token is
        # also the final valid token for every example.
        last_hidden = prefix_out[:, -1:]
        cache = jax.tree.map(
            lambda x: jnp.pad(x, ((0, 0), (0, 0), (0, max_decoding_steps), (0, 0), (0, 0))), cache
        )
        batch = prefix.shape[0]
        prefill_size = prefix.shape[1]
        valid_prefix = jnp.sum(prefix_mask, axis=-1)
        output = jnp.zeros((batch, max_decoding_steps), dtype=jnp.int32)

        def body(carry):
            hidden, cache, output, finished, step = carry
            logits = self.PaliGemma.llm(hidden, vocabulary_ids, method="decode_subset")[:, -1]
            local = jnp.argmax(logits, axis=-1)
            token = vocabulary_ids[local]
            token = jnp.where(finished, EOS_TOKEN, token)
            output = output.at[:, step].set(token)
            finished = jnp.logical_or(finished, token == EOS_TOKEN)

            embedded = self.PaliGemma.llm(token[:, None], method="embed")
            key_mask = jnp.concatenate(
                [
                    prefix_mask,
                    jnp.broadcast_to(
                        jnp.arange(max_decoding_steps)[None, :] <= step,
                        (batch, max_decoding_steps),
                    ),
                ],
                axis=1,
            )
            query_mask = key_mask[:, None, :]
            token_positions = valid_prefix[:, None] + step
            (hidden, _), cache = self.PaliGemma.llm(
                [embedded, None],
                mask=query_mask,
                positions=token_positions,
                kv_cache=cache,
                cache_position=jnp.asarray(prefill_size + step, dtype=jnp.int32),
                adarms_cond=[None, None],
            )
            return hidden, cache, output, finished, step + 1

        def cond(carry):
            return jnp.logical_and(jnp.logical_not(jnp.all(carry[3])), carry[4] < max_decoding_steps)

        _, _, output, _, _ = jax.lax.while_loop(
            cond, body, (last_hidden, cache, output, jnp.zeros((batch,), dtype=jnp.bool_), 0)
        )
        return output

    @override
    def sample_actions(self, rng, observation, *, num_steps=10, noise=None):
        observation = _model.preprocess_observation(
            None, observation, train=False, image_keys=IMAGE_KEYS
        )
        dt = -1.0 / num_steps
        batch = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch, self.action_horizon, self.action_dim))
        prefix, prefix_mask, prefix_ar, _ = self._embed_ecot_prefix(observation)
        attention = pi0.make_attn_mask(prefix_mask, prefix_ar)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, cache = self.PaliGemma.llm(
            [prefix, None], mask=attention, positions=positions, adarms_cond=[None, None],
            lora_enabled=[True, False],
        )

        def step(carry):
            x_t, time = carry
            suffix, suffix_mask, suffix_ar, adarms = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch)
            )
            suffix_attention = pi0.make_attn_mask(suffix_mask, suffix_ar)
            prefix_attention = einops.repeat(prefix_mask, "b p -> b s p", s=suffix.shape[1])
            full_attention = jnp.concatenate([prefix_attention, suffix_attention], axis=-1)
            suffix_positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
            (_, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix], mask=full_attention, positions=suffix_positions,
                kv_cache=cache, adarms_cond=[None, adarms], lora_enabled=[True, False]
            )
            velocity = self.action_out_proj(suffix_out[:, -self.action_horizon :])
            return x_t + dt * velocity, time + dt

        def cond(carry):
            return carry[1] >= -dt / 2

        result, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return result
