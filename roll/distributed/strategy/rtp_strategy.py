import asyncio
import sys
import time
from typing import Dict, List, Optional

import torch
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
    """RTP-LLM inference strategy with continuous batching via the C++ engine."""

    strategy_name = "rtp"

    def __init__(self, worker: Worker):
        super().__init__(worker)
        self.model = None
        self.rtp_op = None
        self.rpc_client = None
        self.token_processor = None
        self._weight_buffer: Dict[str, torch.Tensor] = {}
        self.is_model_in_gpu = False
        self._device = "cuda"
        self._request_id_counter = 0

    def _next_request_id(self) -> int:
        rid = self._request_id_counter
        self._request_id_counter += 1
        return rid

    async def initialize(self, model_provider):
        set_seed(seed=self.worker.pipeline_config.seed)

        from rtp_llm.config.engine_config import EngineConfig
        from rtp_llm.config.py_config_modules import PyEnvConfigs
        from rtp_llm.cpp.model_rpc.model_rpc_client import ModelRpcClient
        from rtp_llm.frontend.token_processor import TokenProcessor
        from rtp_llm.model_factory import ModelFactory
        from rtp_llm.ops.rtp_llm.rtp_llm_op import RtpLLMOp
        from rtp_llm.tools.api.hf_model_helper import get_model_info_from_hf

        model_path = self.worker_config.model_args.model_name_or_path
        strategy_config = self.worker_config.strategy_args.strategy_config
        max_total_tokens = strategy_config.get("max_total_tokens", 2048)
        tokens_per_block = strategy_config.get("tokens_per_block", 64)
        kv_cache_mem_mb = strategy_config.get("kv_cache_mem_mb", 4096)

        logger.info(f"Loading model from {model_path} with rtp-llm engine (continuous batching)")

        # rtp-llm's config parser reads sys.argv during PyEnvConfigs() and
        # EngineConfig.create(); replace it before any config construction.
        saved_argv = sys.argv
        sys.argv = ["rtp_strategy"]
        try:
            py_env_configs = PyEnvConfigs()

            model_path_resolved, model_type = get_model_info_from_hf(model_path, None)

            py_env_configs.model_args.model_type = model_type
            py_env_configs.model_args.ckpt_path = model_path_resolved
            py_env_configs.model_args.max_seq_len = max_total_tokens
            py_env_configs.kv_cache_config.seq_size_per_block = tokens_per_block
            py_env_configs.kv_cache_config.kv_cache_mem_mb = kv_cache_mem_mb
            if not py_env_configs.model_args.tokenizer_path:
                py_env_configs.model_args.tokenizer_path = model_path_resolved

            engine_config = EngineConfig.create(py_env_configs, nccl_comm_config=None)

            model_config = ModelFactory.create_model_config(
                model_args=py_env_configs.model_args,
                lora_config=py_env_configs.lora_config,
                kv_cache_config=engine_config.kv_cache_config,
                profiling_debug_logging_config=engine_config.profiling_debug_logging_config,
                generate_env_config=py_env_configs.generate_env_config,
                embedding_config=py_env_configs.embedding_config,
                quantization_config=py_env_configs.quantization_config,
                render_config=py_env_configs.render_config,
                vit_config=py_env_configs.vit_config,
            )

            ModelFactory.update_engine_config_from_model_config(
                engine_config=engine_config,
                model_config=model_config,
            )

            self.model = ModelFactory._create_model(
                model_config=model_config,
                engine_config=engine_config,
                vit_config=py_env_configs.vit_config,
                merge_lora=False,
            )
            self.model.load()

            self.token_processor = TokenProcessor(
                self.model.tokenizer,
                self.model.model_config.special_tokens,
            )

            self.rtp_op = RtpLLMOp(
                engine_config=engine_config,
                model=self.model,
                propose_model=None,
                token_processor=self.token_processor,
                mm_process_engine=None,
            )
            self.rtp_op.start()
        finally:
            sys.argv = saved_argv

        rpc_port = engine_config.server_config.rpc_server_port
        self.rpc_client = ModelRpcClient(
            addresses=[f"127.0.0.1:{rpc_port}"],
            client_config={},
        )

        await self._wait_engine_ready()

        self.tokenizer = self.model.tokenizer
        self._device = "cuda"
        self.is_model_in_gpu = True
        logger.info(
            f"RtpStrategy initialized with engine on device={self._device}, rpc_port={rpc_port}"
        )

    async def _wait_engine_ready(self, timeout_s: int = 120):
        from rtp_llm.config.generate_config import GenerateConfig
        from rtp_llm.utils.base_model_datatypes import GenerateInput

        eos_id = self.model.tokenizer.eos_token_id
        if eos_id is None:
            eos_id = 0

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                test_input = GenerateInput(
                    request_id=self._next_request_id(),
                    token_ids=torch.tensor([eos_id], dtype=torch.int32),
                    mm_inputs=[],
                    generate_config=GenerateConfig(
                        max_new_tokens=1,
                        num_return_sequences=1,
                        return_output_ids=True,
                        stop_words_list=[[eos_id]],
                    ),
                )
                await self.rpc_client.batch_enqueue([test_input])
                logger.info("Engine gRPC server is ready")
                return
            except Exception as e:
                logger.debug(f"Engine not ready, retrying: {e}")
                await asyncio.sleep(0.5)
        raise RuntimeError(f"Engine did not become ready within {timeout_s}s")

    def _build_generate_config(
        self,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        stop_token_ids: list,
    ):
        from rtp_llm.config.generate_config import GenerateConfig

        stop_words_list = [[tid] for tid in stop_token_ids] if stop_token_ids else []
        do_sample = temperature > 0
        if top_k is not None and top_k < 0:
            top_k = 0

        return GenerateConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else 1.0,
            top_p=top_p,
            top_k=top_k,
            num_return_sequences=1,
            do_sample=do_sample,
            stop_words_list=stop_words_list,
            return_output_ids=True,
            return_incremental=False,
            is_streaming=False,
        )

    def _build_generate_input(self, prompt_ids: list, gen_config) -> "GenerateInput":
        from rtp_llm.utils.base_model_datatypes import GenerateInput

        token_ids = torch.tensor(prompt_ids, dtype=torch.int32)
        return GenerateInput(
            request_id=self._next_request_id(),
            token_ids=token_ids,
            mm_inputs=[],
            generate_config=gen_config,
        )

    def _extract_output_ids(self, result) -> list:
        if not result.generate_outputs:
            return []
        output = result.generate_outputs[0]
        output_ids = output.output_ids
        if output_ids is None:
            return []
        if isinstance(output_ids, torch.Tensor):
            output_ids = output_ids.cpu().tolist()
        elif not isinstance(output_ids, list):
            import numpy as np
            if isinstance(output_ids, np.ndarray):
                output_ids = output_ids.reshape(-1).tolist()
            else:
                output_ids = list(output_ids)
        if output_ids and isinstance(output_ids[0], list):
            output_ids = output_ids[0]
        return [int(x) for x in output_ids]

    async def generate(self, batch: DataProto, generation_config: Dict) -> torch.Tensor:
        input_ids = batch.batch["input_ids"]
        attention_mask = batch.batch["attention_mask"]
        prompts = gather_unpadded_input_ids(input_ids=input_ids, attention_mask=attention_mask)

        max_new_tokens = generation_config["max_new_tokens"]
        n = generation_config.get("num_return_sequences", 1)
        stop_token_ids = generation_config.get("eos_token_id", [])
        if isinstance(stop_token_ids, int):
            stop_token_ids = [stop_token_ids]

        temperature = generation_config.get("temperature", 1.0)
        top_p = generation_config.get("top_p", 1.0)
        top_k = generation_config.get("top_k", 0)

        gen_config = self._build_generate_config(
            max_new_tokens, temperature, top_p, top_k, stop_token_ids
        )

        inputs = []
        for prompt_ids in prompts:
            for _ in range(n):
                inputs.append(self._build_generate_input(prompt_ids, gen_config))

        try:
            results = await self.rpc_client.batch_enqueue(inputs)
        except Exception as e:
            msg = f"batch_enqueue failed: {type(e).__name__}: {e}"
            if hasattr(e, "exception_type"):
                msg += f" (exception_type={e.exception_type})"
            logger.error(msg)
            raise RuntimeError(msg) from None

        all_output_ids = []
        for result in results:
            output_ids = self._extract_output_ids(result)
            all_output_ids.append(torch.tensor(output_ids, device=input_ids.device))

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id or 0

        output_ids_tensor = pad_sequence(
            all_output_ids, batch_first=True, padding_value=pad_token_id
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
        n = sp.get("n", 1)

        temperature = sp.get("temperature", 1.0)
        top_p = sp.get("top_p", 1.0)
        top_k = sp.get("top_k", 0)

        gen_config = self._build_generate_config(
            max_new_tokens, temperature, top_p, top_k, stop_token_ids
        )

        inputs = []
        for _ in range(n):
            inputs.append(self._build_generate_input(input_ids, gen_config))

        try:
            results = await self.rpc_client.batch_enqueue(inputs)
        except Exception as e:
            msg = f"batch_enqueue failed: {type(e).__name__}: {e}"
            if hasattr(e, "exception_type"):
                msg += f" (exception_type={e.exception_type})"
            logger.error(msg)
            raise RuntimeError(msg) from None

        all_output_ids = []
        finish_reasons = []
        for result in results:
            output_ids = self._extract_output_ids(result)
            all_output_ids.append(output_ids)
            finished = bool(result.generate_outputs and result.generate_outputs[0].finished)
            finish_reasons.append("stop" if finished else "length")

        return {
            "output_token_ids": all_output_ids,
            "finish_reasons": finish_reasons,
            "output_logprobs": None,
        }

    async def abort_requests(self, request_ids=None):
        pass

    async def load_states(self, *args, **kwargs):
        if not self.is_model_in_gpu:
            self.is_model_in_gpu = True
            logger.info("RtpStrategy load_states (engine resident, no transfer needed)")

    async def offload_states(self, include=None, non_blocking=False):
        if include is None or OffloadStateType.model_params in include:
            if self.is_model_in_gpu and self.worker.pipeline_config.is_actor_infer_colocated:
                logger.info(
                    "RtpStrategy offload_states: engine stays resident on GPU "
                    "(pause/restart not yet exposed via pybind)"
                )
        clear_memory()

    async def setup_collective_group(
        self, master_address, master_port, rank_offset, world_size, group_name, backend=None
    ):
        logger.info(f"setup_collective_group {group_name=} rank={rank_offset} world_size={world_size}")
        backend = backend if backend is not None else current_platform.communication_backend
        collective.init_collective_group(
            world_size, rank_offset, backend=backend, group_name=group_name,
            master_addr=master_address, master_port=master_port,
        )

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
        self.model.weight_manager.load_weights(self._weight_buffer)
        logger.info("In-place weight update succeeded")
        self._weight_buffer.clear()
        clear_memory()

    def get_metrics(self, metric_names: Optional[List[str]] = None) -> Dict[str, float]:
        return {}
