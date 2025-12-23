# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hydra
import torch

from rlinf.models import get_model, get_vla_model_config_and_processor


@hydra.main(
    version_base="1.1", config_path="config", config_name="maniskill_ppo_openvlaoft"
)
def main(cfg) -> None:
    model = get_model(cfg.actor.checkpoint_load_path, cfg.actor.model)
    model_config, input_processor = get_vla_model_config_and_processor(cfg.actor)
    model.setup_config_and_processor(model_config, cfg, input_processor)

    _train_sampling_params = {
        "temperature": cfg.algorithm.sampling_params["temperature_train"],
        "top_k": cfg.algorithm.sampling_params["top_k"],
        "top_p": cfg.algorithm.sampling_params["top_p"],
        "max_new_tokens": cfg.algorithm.length_params["max_new_token"],
        "use_cache": True,
    }
    kwargs = _train_sampling_params
    kwargs["do_sample"] = cfg.actor.model.get("do_sample", True)

    env_obs = torch.load(
        "/mnt/mnt/public_zgc/home/mjwei/repo/RLinf/outputs/extracted_obs.pt"
    )
    with torch.no_grad():
        actions, result = model.predict_action_batch(
            env_obs=env_obs,
            **kwargs,
        )
    # do_sample = True

    # if env_obs is not None:
    #     task_descriptions = [
    #         f"In: What action should the robot take to {t.lower()}?\nOut: "
    #         for t in env_obs["task_descriptions"]
    #     ]

    #     all_images = [env_obs["images"]]
    #     if self.vision_backbone.get_num_images_in_input() > 1:
    #         wrist_imgs = env_obs["wrist_images"]  # [B, N_IMG, C, H, W]
    #         all_images.extend(
    #             [wrist_imgs[:, i] for i in range(wrist_imgs.shape[1])]
    #         )

    #     max_length = self.max_prompt_length
    #     device = next(self.parameters()).device
    #     precision = next(self.parameters()).dtype

    #     primary_image = all_images.pop(0)
    #     if primary_image.ndim == 4:
    #         primary_image = primary_image.unsqueeze(1)
    #     assert primary_image.ndim == 5
    #     images = {"images": primary_image}
    #     inputs = self.input_processor(
    #         text=task_descriptions,
    #         images=images,
    #         proprio_states=env_obs["states"],
    #         padding="max_length",
    #         max_length=max_length,
    #     )

    #     if all_images:
    #         all_wrist_inputs = [
    #             self.input_processor(
    #                 text=task_descriptions,
    #                 images={"images": wrist_image.unsqueeze(1)},
    #                 proprio_states=env_obs["states"],
    #                 padding="max_length",
    #                 max_length=max_length,
    #             )
    #             for wrist_image in all_images
    #         ]

    #         # Concatenate all images
    #         primary_pixel_values = inputs["pixel_values"]
    #         all_wrist_pixel_values = [
    #             wrist_inputs["pixel_values"] for wrist_inputs in all_wrist_inputs
    #         ]
    #         inputs["pixel_values"] = torch.cat(
    #             [primary_pixel_values] + all_wrist_pixel_values, dim=1
    #         )

    #     input_ids = inputs["input_ids"].to(device=device, dtype=torch.long)
    #     attention_mask = inputs["attention_mask"].to(
    #         device=device, dtype=torch.bool
    #     )
    #     pixel_values = inputs["pixel_values"].to(device=device, dtype=precision)

    #     B, N, C, H, W = pixel_values.shape
    #     pixel_values = pixel_values.reshape(B, N * C, H, W)

    # forward_inputs = {
    #     "input_ids": input_ids,
    #     "attention_mask": attention_mask,
    #     "pixel_values": pixel_values,
    # }

    # # assert first token is 1
    # assert torch.all(input_ids[:, 0] == 1)
    # assert torch.all(attention_mask[:, 0] == 1)
    # # last token is space ` `
    # assert torch.all(input_ids[:, -1] == 29871)
    # assert torch.all(attention_mask[:, -1] == 1)

    # n_prompt_tokens = input_ids.shape[-1] - 1
    # # Calculate number of patches (including proprio token and/or diffusion timestep embedding if present)
    # n_patches = (
    #     self.vision_backbone.get_num_patches()
    #     * self.vision_backbone.get_num_images_in_input()
    # )

    # # llm inputs
    # input_ids, attention_mask = self._prepare_input_for_action_prediction(
    #     input_ids, attention_mask
    # )
    # assert torch.all(input_ids[:, -1] == STOP_INDEX)  # [B, L + act + 1, D]
    # assert torch.all(
    #     attention_mask[:, -1 - self.action_dim * self.num_action_chunks :] == 1
    # )  # [B, L + act + 1]

    # # multimodal
    # mm_embeddings, mm_attention_mask = self._build_embedding(
    #     input_ids, attention_mask, pixel_values
    # )
    # multimodal_position_ids = mm_attention_mask.cumsum(dim=1) - 1

    # # Forward pass through language model
    # outputs = self.language_model(
    #     input_ids=None,
    #     attention_mask=mm_attention_mask,
    #     position_ids=multimodal_position_ids,
    #     past_key_values=None,
    #     inputs_embeds=mm_embeddings,
    #     labels=None,
    #     use_cache=None,
    #     output_attentions=False,
    #     output_hidden_states=True,
    #     return_dict=True,
    # )

    # # Extract hidden states for action tokens
    # last_hidden_states = outputs.hidden_states[-1]  # (B, seq_len, D)
    # assert last_hidden_states.shape[1] == mm_embeddings.shape[1]

    # logits_tensor = outputs.logits[
    #     :,
    #     n_patches + n_prompt_tokens : n_patches
    #     + n_prompt_tokens
    #     + self.action_dim * self.num_action_chunks,
    #     :,
    # ]  # [B, act, vocab_size + 64]

    # last_hidden_states = last_hidden_states[
    #     :, -self.action_dim * self.num_action_chunks - 1 : -1
    # ]

    # logits_tensor[..., : self.vocab_size - self.config.n_action_bins] = -torch.inf
    # logits_tensor[..., self.vocab_size :] = -torch.inf

    # processed_logits_tensor = logits_tensor / kwargs["temperature"]
    # top_k = min(kwargs["top_k"], processed_logits_tensor.size(-1))  # Safety check
    # if top_k > 0:
    #     logits_warper = TopKLogitsWarper(
    #         top_k
    #     )  # since here is logprob instead of logits, we use 0 instead of -inf
    #     processed_logits_tensor = logits_warper(None, processed_logits_tensor)

    # processed_logprob_tensor = F.log_softmax(
    #     processed_logits_tensor, dim=-1
    # )  # [B, act, vocab_size + 64]

    # if do_sample:
    #     probs_tensor = torch.exp(
    #         processed_logprob_tensor
    #     )  # [B, act, vocab_size + 64]
    #     probs_flat = probs_tensor.view(
    #         -1, processed_logprob_tensor.shape[-1]
    #     )  # [B * act, vocab_size + 64]

    #     sample_flat = torch.multinomial(
    #         probs_flat, num_samples=1, replacement=True
    #     )  # [B * act, 1]
    #     idxs = sample_flat.view(
    #         processed_logprob_tensor.shape[0], processed_logprob_tensor.shape[1]
    #     )  # [B, act]
    # else:
    #     idxs = processed_logprob_tensor.argmax(dim=-1)  # [B, act]

    # # assert torch.all(idxs >= 0) and torch.all(idxs < self.config.n_action_bins)
    # # generated_ids = idxs + (self.vocab_size - self.config.n_action_bins)
    # assert torch.all(
    #     idxs >= self.vocab_size - self.config.n_action_bins
    # ) and torch.all(idxs < self.vocab_size)

    # chunk_action_tokens = idxs.reshape(-1, self.action_dim)
    # predicted_action_token_ids = chunk_action_tokens.cpu().numpy()
    # discretized_actions = self.vocab_size - predicted_action_token_ids
    # discretized_actions = np.clip(
    #     discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1
    # )
    # # normalized_actions = self.bin_centers[discretized_actions]
    # normalized_actions = np.asarray(
    #     [self.bin_centers[da] for da in discretized_actions]
    # )  # [B, dim]
    # normalized_actions = normalized_actions.reshape(-1, self.action_dim)

    # # Unnormalize predicted actions
    # actions = self._unnormalize_actions(normalized_actions, self.unnorm_key)
    # actions = actions.reshape(idxs.shape)

    # action_logits = processed_logits_tensor.permute(
    #     0, 2, 1
    # )  # [B, vocab-size, action-dim]
    # action_logits[:, : self.vocab_size - self.config.n_action_bins] = -torch.inf
    # action_logits[:, self.vocab_size :] = -torch.inf

    # chunk_logprobs = compute_logprobs_from_logits(logits=action_logits, target=idxs)

    # if hasattr(self, "value_head") and calulate_values:
    #     hidden_features = last_hidden_states[
    #         :, -self.action_dim * self.num_action_chunks
    #     ]  # [batch_size, hidden_dim]

    #     chunk_values = self.value_head(hidden_features)  # [batch_size, 1]
    # else:
    #     chunk_values = torch.zeros_like(chunk_logprobs[..., :1])

    # chunk_actions = actions.reshape(-1, self.num_action_chunks, self.action_dim)
    # chunk_action_tokens = idxs.reshape(-1, self.num_action_chunks, self.action_dim)

    # forward_inputs["action_tokens"] = chunk_action_tokens

    # result = {
    #     "prev_logprobs": chunk_logprobs,
    #     "prev_values": chunk_values,
    #     "forward_inputs": forward_inputs,
    # }

    # return chunk_actions, result


if __name__ == "__main__":
    main()
