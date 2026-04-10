from __future__ import annotations

import datetime
import os
from typing import Optional

from omegaconf import DictConfig

from verl import DataProto
from verl.single_controller.base import Worker
from verl.utils.device import get_device_name
from verl.utils.distributed import get_nccl_backend
from verl.utils.fs import copy_to_local
from verl.utils.import_utils import import_external_libs
from verl.utils.model import update_model_config
from verl.utils.ray_utils import get_event_loop
from verl.utils.profiler import DistProfiler, DistProfilerExtension, ProfilerConfig, log_gpu_memory_usage, simple_timer
from verl.workers.rollout import get_rollout_class

from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register

from verl.utils.config import omega_conf_to_dataclass

logger = __import__("logging").getLogger(__name__)


class TeacherRolloutWorker(Worker, DistProfilerExtension):
    """
    A lightweight rollout-only worker for teacher model generation.

    This worker:
    - only builds rollout engine
    - only serves generate_sequences
    - does not build actor / ref / optimizer
    - does not switch trainer/rollout mode
    - does not update weights
    """

    def __init__(self, config: DictConfig, role: str = "teacher_rollout", **kwargs):
        Worker.__init__(self)

        self.config = config
        self.role = role

        import torch
        import torch.distributed

        if not torch.distributed.is_initialized():
            rank = int(os.environ.get("RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            torch.distributed.init_process_group(
                backend=f"cpu:gloo,{get_device_name()}:{get_nccl_backend()}",
                rank=rank,
                world_size=world_size,
                timeout=datetime.timedelta(seconds=self.config.get("nccl_timeout", 600)),
                init_method=os.environ.get("DIST_INIT_METHOD", None),
            )

        omega_profiler_config = self.config.teacher_rollout.rollout.get("profiler", {})
        profiler_config = omega_conf_to_dataclass(omega_profiler_config, dataclass_type=ProfilerConfig)
        if omega_profiler_config.get("tool", None) in ["npu", "nsys", "torch", "torch_memory"]:
            tool_config = omega_conf_to_dataclass(
                omega_profiler_config.get("tool_config", {}).get(omega_profiler_config.get("tool"))
            )
        else:
            tool_config = None

        DistProfilerExtension.__init__(
            self, DistProfiler(rank=self.rank, config=profiler_config, tool_config=tool_config)
        )

        self.rollout = None
        self.tokenizer = None
        self.processor = None
        self.generation_config = None
        self.model_config = None

    def _build_rollout(self, trust_remote_code: bool = False):
        from torch.distributed.device_mesh import init_device_mesh

        from verl.utils import hf_processor, hf_tokenizer
        from verl.workers.config import HFModelConfig, RolloutConfig

        rollout_config: RolloutConfig = omega_conf_to_dataclass(self.config.teacher_rollout.rollout)
        model_config: HFModelConfig = omega_conf_to_dataclass(
            self.config.teacher_rollout.model, dataclass_type=HFModelConfig
        )
        self.model_config = model_config

        local_path = copy_to_local(
            self.config.teacher_rollout.model.path,
            use_shm=self.config.teacher_rollout.model.get("use_shm", False),
        )
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        self.processor = hf_processor(local_path, trust_remote_code=trust_remote_code)

        if self.config.teacher_rollout.model.get("custom_chat_template", None) is not None:
            if self.processor is not None:
                self.processor.chat_template = self.config.teacher_rollout.model.custom_chat_template
            else:
                self.tokenizer.chat_template = self.config.teacher_rollout.model.custom_chat_template

        infer_tp = (
            self.config.teacher_rollout.rollout.tensor_model_parallel_size
            * self.config.teacher_rollout.rollout.data_parallel_size
        )
        infer_pp = self.config.teacher_rollout.rollout.pipeline_model_parallel_size
        infer_world_size = infer_tp * infer_pp
        dp = self.world_size // infer_world_size

        assert self.world_size % infer_world_size == 0, (
            f"teacher rollout world_size: {self.world_size} is not divisible by "
            f"infer_world_size: {infer_world_size}"
        )

        device_name = get_device_name()
        rollout_device_mesh = init_device_mesh(
            device_name,
            mesh_shape=(dp, infer_tp, infer_pp),
            mesh_dim_names=["dp", "infer_tp", "infer_pp"],
        )

        rollout_name = self.config.teacher_rollout.rollout.name
        if rollout_name == "hf":
            self._register_dispatch_collect_info("rollout", dp_rank=self.rank, is_collect=True)
        else:
            is_collect = (
                rollout_device_mesh["infer_tp"].get_local_rank() == 0
                and rollout_device_mesh["infer_pp"].get_local_rank() == 0
            )
            self._register_dispatch_collect_info(
                "rollout",
                dp_rank=rollout_device_mesh["dp"].get_local_rank(),
                is_collect=is_collect,
            )

        import torch

        self.torch_random_states = torch.cuda.get_rng_state()
        gen_dp_rank = rollout_device_mesh["dp"].get_local_rank()
        torch.cuda.manual_seed(gen_dp_rank + 1000)
        self.gen_random_states = torch.cuda.get_rng_state()
        torch.cuda.set_rng_state(self.torch_random_states)

        log_gpu_memory_usage(f"Before building teacher {rollout_name} rollout", logger=logger)
        self.rollout = get_rollout_class(rollout_config.name, rollout_config.mode)(
            config=rollout_config,
            model_config=model_config,
            device_mesh=rollout_device_mesh,
        )
        log_gpu_memory_usage(f"After building teacher {rollout_name} rollout", logger=logger)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        import_external_libs(self.config.teacher_rollout.model.get("external_lib", None))
        self._build_rollout(
            trust_remote_code=self.config.teacher_rollout.model.get("trust_remote_code", False)
        )

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="rollout"))
    @DistProfiler.annotate(color="magenta", role="teacher_rollout_generate")
    def generate_sequences(self, prompts: DataProto):
        assert self.rollout is not None, "Teacher rollout engine is not initialized."

        from verl.utils.memory_buffer import aggressive_empty_cache
        from verl.utils.torch_functional import reduce_timing, topk_reduce_ratio_min_max
        from verl.utils.tracking import simple_timer

        prompts = prompts.to(get_device_name())

        meta_info = {
            "eos_token_id": self.generation_config.eos_token_id
            if self.generation_config is not None
            else self.tokenizer.eos_token_id,
            "pad_token_id": self.generation_config.pad_token_id
            if self.generation_config is not None
            else self.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)

        timing_generate = {}
        with simple_timer("generate_sequences", timing_generate):
            output = self.rollout.generate_sequences(prompts=prompts)

        timing_generate_topk_ratio, timing_generate_min, timing_generate_max = topk_reduce_ratio_min_max(
            timing_generate["generate_sequences"]
        )
        timing_generate = reduce_timing(timing_generate)
        timing_generate.update(
            {
                "generation_timing/max": timing_generate_max,
                "generation_timing/min": timing_generate_min,
                "generation_timing/topk_ratio": timing_generate_topk_ratio,
            }
        )
        output.meta_info["timing"] = timing_generate
        output = output.to("cpu")

        aggressive_empty_cache(force_sync=True)
        return output