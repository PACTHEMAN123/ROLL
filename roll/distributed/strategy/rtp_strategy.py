import os
import math
import shutil
import tempfile
from typing import Dict, List, Optional

import torch
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence
from transformers import set_seed

from roll.distributed.executor.worker import Worker
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.strategy.strategy import InferenceStrategy
from roll.platforms import current_platform
from roll.utils.collective import collective
from roll.utils.cuda_ipc_utils import MultiprocessingSerializer
from roll.utils.functionals import concatenate_input_and_output, gather_unpadded_input_ids
from roll.utils.logging import get_logger
from roll.utils.offload_states import OffloadStateType, clear_memory
from roll.utils.send_recv_utils import monkey_patch_torch_reductions, named_tensors_from_bucket

logger = get_logger()


class RtpStrategy(InferenceStrategy):
    strategy_name = "rtp"

    def __init__(self, worker: Worker):
        super().__init__(worker)
        self.auto_model = None
        self._weight_buffer: Dict[str, torch.Tensor] = {}
        self.is_model_in_gpu = False
        self._device = "cuda"

    async def initialize(self, model_provider):
        set_seed(seed=self.worker.pipeline_config.seed)

        from rtp_llm.models_py.standalone.auto_model import AutoModel

        model_path = self.worker_config.model_args.model_name_or_path
        strategy_config = self.worker_config.strategy_args.strategy_config
        max_total_tokens = strategy_config.get("max_total_tokens", 2048)
        tokens_per_block = strategy_config.get("tokens_per_block", 64)

        logger.info(f"Loading model from {model_path} with rtp-llm AutoModel")
        self.auto_model = AutoModel.from_pretrained(
            model_path,
            max_total_tokens=max_total_tokens,
            tokens_per_block=tokens_per_block,
        )
        self.tokenizer = self.auto_model.tokenizer
        self._device = self.auto_model.device
        self.is_model_in_gpu = True
        logger.info(f"RtpStrategy initialized on device={self._device}")

    async def generate(self, batch: DataProto, generation_config: Dict) -> torch.Tensor:
        input_ids = batch.batch["input_ids"]
        attention_mask = batch.batch["attention_mask"]
        prompts = gather_unpadded_input_ids(input_ids=input_ids, attention_mask=attention_mask)

        max_new_tokens = generation_config["max_new_tokens"]
        n = generation_config.get("num_return_sequences", 1)
        stop_token_ids = generation_config.get("eos_token_id", [])
        if isinstance(stop_token_ids, int):
            stop_token_ids = [stop_token_ids]

        sampling_params = {
            "temperature": generation_config.get("temperature", 1.0),
            "top_p": generation_config.get("top_p", 1.0),
            "top_k": generation_config.get("top_k", 0),
        }

        all_output_ids = []
        for prompt_ids in prompts:
            for _ in range(n):
                output_ids = self._generate_with_sampling(
                    prompt_ids, max_new_tokens, sampling_params, stop_token_ids
                )
                all_output_ids.append(torch.tensor(output_ids, device=input_ids.device))

        output_ids_tensor = pad_sequence(
            all_output_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        output = concatenate_input_and_output(
            input_ids=input_ids, output_ids=output_ids_tensor, num_return_sequences=n
        )
        return output

    async def generate_request(self, payload: Dict) -> Dict:
        input_ids = payload["input_ids"]
        sp = payload.get("sampling_params", {})

        max_new_tokens = sp.get("max_new_tokens", sp.get("max_tokens", 128))
        stop_token_ids = sp.get("stop_token_ids", [])
        if isinstance(stop_token_ids, int):
            stop_token_ids = [stop_token_ids]

        sampling_params = {
            "temperature": sp.get("temperature", 1.0),
            "top_p": sp.get("top_p", 1.0),
            "top_k": sp.get("top_k", 0),
        }

        output_ids = self._generate_with_sampling(
            input_ids, max_new_tokens, sampling_params, stop_token_ids
        )

        return {
            "output_token_ids": [output_ids],
            "finish_reasons": ["stop"],
            "output_logprobs": [],
        }

    async def abort_requests(self, request_ids=None):
        pass

    def _generate_with_sampling(
        self, input_ids: list, max_new_tokens: int, sampling_params: dict, stop_token_ids: list = None
    ) -> list:
        from rtp_llm.ops.compute_ops import PyModelInputs

        am = self.auto_model
        output_ids = []
        input_length = len(input_ids)
        input_ids_tensor = torch.tensor(input_ids, dtype=torch.int32, device=am.device)

        # Prefill
        attn_inputs = am._prepare_prefill_attention_inputs(input_length)
        model_inputs = PyModelInputs(input_ids=input_ids_tensor, attention_inputs=attn_inputs)
        model_outputs = am.model.forward(model_inputs)
        next_token_id = self._sample_next_token(model_outputs, sampling_params)
        next_token_cpu = next_token_id.cpu().item()

        if stop_token_ids and next_token_cpu in stop_token_ids:
            return output_ids
        output_ids.append(next_token_cpu)

        # Decode loop
        gen_tokens = 1
        while gen_tokens < max_new_tokens:
            attn_inputs = am._prepare_decode_attention_inputs(attn_inputs, input_length + gen_tokens)
            model_inputs = PyModelInputs(input_ids=next_token_id, attention_inputs=attn_inputs)
            model_outputs = am.model.forward(model_inputs)
            next_token_id = self._sample_next_token(model_outputs, sampling_params)
            next_token_cpu = next_token_id.cpu().item()
            gen_tokens += 1

            if stop_token_ids and next_token_cpu in stop_token_ids:
                break
            output_ids.append(next_token_cpu)

        return output_ids

    def _sample_next_token(self, model_outputs, sampling_params: dict) -> torch.Tensor:
        am = self.auto_model
        hidden_states = model_outputs.hidden_states[-1:, :]
        logits = torch.matmul(
            hidden_states.to(am.lm_head_weight.dtype), am.lm_head_weight.t()
        ).to(torch.float32)

        temperature = sampling_params.get("temperature", 1.0)
        if temperature == 0:
            return torch.argmax(logits, dim=-1)

        logits = logits / temperature

        top_k = sampling_params.get("top_k", 0)
        if top_k and top_k > 0:
            top_k = min(top_k, logits.size(-1))
            topk_values, _ = torch.topk(logits, top_k, dim=-1)
            logits[logits < topk_values[..., -1:]] = float("-inf")

        top_p = sampling_params.get("top_p", 1.0)
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float("-inf")

        probs = torch.softmax(logits, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1).squeeze(-1)
        return next_token_id

    async def load_states(self, *args, **kwargs):
        if not self.is_model_in_gpu:
            self.auto_model.model.to(self._device)
            self.is_model_in_gpu = True
            logger.info("RtpStrategy model loaded to GPU")

    async def offload_states(self, include=None, non_blocking=False):
        if include is None or OffloadStateType.model_params in include:
            if self.is_model_in_gpu and self.worker.pipeline_config.is_actor_infer_colocated:
                self.auto_model.model.to("cpu")
                self.is_model_in_gpu = False
                logger.info("RtpStrategy model offloaded to CPU")
        clear_memory()

    async def setup_collective_group(
        self, master_address, master_port, rank_offset, world_size, group_name, backend=None
    ):
        logger.info(f"setup_collective_group {group_name=} rank={rank_offset + 1} world_size={world_size}")
        backend = backend if backend is not None else current_platform.communication_backend
        collective.init_collective_group(
            world_size, rank_offset + 1, backend=backend, group_name=group_name,
            master_addr=master_address, master_port=master_port,
        )
        collective.allreduce(torch.zeros(1).to(current_platform.device_type), group_name=group_name)

    async def broadcast_parameter(self, names, dtypes, shapes, group_name, is_lora=False):
        assert not is_lora, "LoRA not supported in rtp strategy"

        weights_and_handles = []
        for name, dtype, shape in zip(names, dtypes, shapes):
            target_dtype = dtype if isinstance(dtype, torch.dtype) else getattr(torch, dtype)
            weight = torch.empty(shape, dtype=target_dtype, device=current_platform.device_type)
            handle = collective.broadcast(
                tensor=weight, src_rank=0, group_name=group_name, async_op=True
            )
            weights_and_handles.append((name, weight, handle))

        for name, weight, handle in weights_and_handles:
            handle.wait()
            self._weight_buffer[name] = weight

    async def update_parameter_in_bucket(self, serialized_named_tensors, is_lora=False):
        assert not is_lora, "LoRA not supported in rtp strategy"
        monkey_patch_torch_reductions()

        for serialized in serialized_named_tensors:
            if serialized is None:
                continue
            bucket_with_meta = MultiprocessingSerializer.deserialize(serialized)
            named_tensors = named_tensors_from_bucket(**bucket_with_meta)
            for name, tensor in named_tensors:
                self._weight_buffer[name] = tensor

    async def process_weights_after_loading(self, *args, **kwargs):
        if not self._weight_buffer:
            return

        logger.info(f"Applying {len(self._weight_buffer)} weight updates to model")

        try:
            self._update_weights_direct()
            logger.info("Direct weight update succeeded")
        except Exception as e:
            logger.warning(f"Direct weight update failed: {e}")
            self._update_weights_via_checkpoint()

        self._weight_buffer.clear()
        clear_memory()

    def _update_weights_direct(self):
        weight = self.auto_model.model.weight
        updated = 0
        for name, new_tensor in self._weight_buffer.items():
            try:
                current = weight.get_global_weight(name)
                current.data.copy_(new_tensor.to(current.dtype).to(current.device))
                updated += 1
            except Exception as e:
                logger.debug(f"Failed to update weight '{name}' directly: {e}")
                raise
        logger.info(f"Updated {updated}/{len(self._weight_buffer)} weights directly")

    def _update_weights_via_checkpoint(self):
        from safetensors.torch import save_file

        tmpdir = tempfile.mkdtemp(prefix="rtp_weight_update_")
        try:
            state_dict = {}
            for name, tensor in self._weight_buffer.items():
                state_dict[name] = tensor.cpu().contiguous()
            save_file(state_dict, os.path.join(tmpdir, "model.safetensors"))

            original_path = self.worker_config.model_args.model_name_or_path
            for fname in os.listdir(original_path):
                if fname.endswith((".json", ".txt", ".model")) or "token" in fname.lower():
                    shutil.copy2(os.path.join(original_path, fname), os.path.join(tmpdir, fname))

            from rtp_llm.models_py.standalone.auto_model import AutoModel

            strategy_config = self.worker_config.strategy_args.strategy_config
            logger.info(f"Reloading model from checkpoint: {tmpdir}")
            self.auto_model = AutoModel.from_pretrained(
                tmpdir,
                max_total_tokens=strategy_config.get("max_total_tokens", 2048),
                tokens_per_block=strategy_config.get("tokens_per_block", 64),
            )
            self._device = self.auto_model.device
            self.is_model_in_gpu = True
            logger.info("Model reloaded from checkpoint successfully")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def get_metrics(self, metric_names: Optional[List[str]] = None) -> Dict[str, float]:
        return {}
