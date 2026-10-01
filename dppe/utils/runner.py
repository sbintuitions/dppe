"""Training and testing launcher for distributed PyTorch experiments."""

import glob
import itertools
import os
import random
import shutil
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
import tyro
import yaml
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter


def set_random_seed(seed):
    """Set random seed for reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def nested_to_device(data: dict, device) -> dict:
    """Recursively move tensors in a nested dict/list structure to a device."""
    if isinstance(data, dict):
        return {k: nested_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [nested_to_device(v, device) for v in data]
    elif isinstance(data, Tensor):
        return data.to(device)
    else:
        return data


@dataclass
class LauncherConfig:
    """Configuration for the training/testing launcher."""

    # Dump everything to this directory.
    output_dir: str = "results/dbg"

    # Maximum number of steps to train.
    max_steps: int = 100

    # Resume training from this checkpoint.
    auto_resume: bool = False
    resume: str | None = None
    only_model: bool = False

    # Save checkpoint every this many steps. (On only rank 0.)
    ckpt_every: int = 10
    # Keep this many checkpoints.
    ckpt_keeps: int = 3

    # Gradient scaling for mixed precision training.
    amp: bool = False
    amp_dtype: Literal["bf16", "fp16"] = "fp16"

    # Check for NaN in gradients.
    check_nan_in_params: bool = False

    # Gradient Accumulation.
    acc: int = 1

    # Random seed.
    fixed_seed: bool = True
    seed: int = 42

    # Gradient clipping.
    grad_clip: float = 1.0

    # test related
    test_every: int = -1
    test_only: bool = False

    # subdirs
    ckpt_subdir: str = "ckpts"
    stats_subdir: str = "stats"
    visual_subdir: str = "visuals"
    test_subdir: str = "tests"


class Launcher:
    """Distributed training and testing launcher with checkpointing and logging."""

    def __init__(self, config: LauncherConfig) -> None:
        self.config = config
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.world_rank = int(os.environ.get("RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))

        self.device = torch.device(f"cuda:{self.local_rank}")

        if self.config.test_only and self.config.resume is not None:
            resume_file = Path(self.config.resume)
            # Expected structure: <any_folder>/ckpts/step-xxxx.pt
            if resume_file.parent.name == self.config.ckpt_subdir:
                # Get the parent of the ckpts folder (experiment root directory)
                inferred_dir = str(resume_file.parent.parent)
                if self.world_rank == 0 and inferred_dir != self.config.output_dir:
                    print(
                        f"[Info] Test mode detected. Overriding output_dir:\n"
                        f"  From : {self.config.output_dir}\n"
                        f"  To   : {inferred_dir}"
                    )
                self.config.output_dir = inferred_dir

        exist_ok = is_pure_resume = (
            self.config.resume is not None
            and not self.config.only_model
            and not self.config.test_only
        )

        exist_ok = is_pure_resume or self.config.test_only

        if is_pure_resume and self.world_rank == 0:
            resume_file = Path(self.config.resume)
            if resume_file.parent.name == self.config.ckpt_subdir:
                old_base_dir = resume_file.parent.parent
                new_base_dir = Path(self.config.output_dir)

                if old_base_dir.exists() and old_base_dir.resolve() != new_base_dir.resolve():
                    print(
                        f"Pure resume detected. Copying previous run directory from {old_base_dir} to {new_base_dir}..."
                    )
                    shutil.copytree(old_base_dir, new_base_dir, dirs_exist_ok=False)
                elif not old_base_dir.exists():
                    raise ValueError(f"{old_base_dir} doesn't exist.")
                else:
                    raise ValueError(
                        "resuming dir == new dir, this is not allowed. Please make new experiment directory unique."
                    )

            else:
                raise ValueError(
                    f"The resume path {resume_file} does not match the expected structure (*/{self.config.ckpt_subdir}/*.pt). Bypassing directory copy."
                )

        # Setup output directories.
        self.output_dir = self.config.output_dir
        self.stats_dir = f"{self.output_dir}/{self.config.stats_subdir}"
        self.ckpt_dir = f"{self.output_dir}/{self.config.ckpt_subdir}"
        self.visual_dir = f"{self.output_dir}/{self.config.visual_subdir}"
        self.test_dir = f"{self.output_dir}/{self.config.test_subdir}"
        self.amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[self.config.amp_dtype]
        self.use_grad_scaler = self.config.amp and self.config.amp_dtype == "fp16"

        if self.world_rank == 0:
            os.makedirs(self.ckpt_dir, exist_ok=exist_ok)
            os.makedirs(self.stats_dir, exist_ok=exist_ok)
            os.makedirs(self.visual_dir, exist_ok=exist_ok)
            os.makedirs(self.test_dir, exist_ok=exist_ok)
            self.writer = SummaryWriter(log_dir=f"{self.config.output_dir}/tb")
            if not self.config.test_only:
                (Path(self.output_dir) / "config.yaml").write_text(yaml.dump(config))
                print(f"Wrote config to {self.output_dir}/config.yaml")

        self.shared_rng = random.Random(self.config.seed)

    def run(self):
        """Run training or testing based on configuration."""
        if self.config.test_only:
            self.test()
        else:
            self.train()

    def train_initialize(self) -> dict[str, Any]:
        """Initialize model, optimizer, scheduler, and data iterator for training. Override to customize."""
        state = {}
        state["model"] = torch.nn.Linear(1, 1)
        state["optimizer"] = torch.optim.Adam(state["model"].parameters(), lr=1e-3)
        state["scheduler"] = None
        state["train_dataiter"] = itertools.repeat(
            {"x": torch.randn(1, 1), "y": torch.randn(1, 1)}
        )
        return state

    def test_initialize(self, model: torch.nn.Module | None = None) -> dict[str, Any]:
        """Initialize model and data iterator for testing. Override to customize."""
        state = {}
        if model is not None:
            state["model"] = model
        else:
            state["model"] = torch.nn.Linear(1, 1)
        state["test_dataiter"] = itertools.repeat(
            {"x": torch.randn(1, 1), "y": torch.randn(1, 1)},
            times=3 if self.world_rank == 0 else 2,  # mimic multi-gpu split.
        )
        return state

    def train_iteration(self, step: int, state: Any, acc_step: int = 0) -> Tensor:
        """Run a single training iteration and return the loss. Override to customize."""
        model = state["model"]
        # optimizer = state["optimizer"]
        train_dataiter = state["train_dataiter"]

        model.train()
        data = next(train_dataiter)
        data = nested_to_device(data, self.device)

        with torch.amp.autocast("cuda", enabled=self.config.amp, dtype=self.amp_dtype):
            output = model(data["x"])
            loss = F.mse_loss(output, data["y"])
        return loss

    @torch.inference_mode()
    def test_iteration(self, step: int, state: Any, acc_step: int = 0) -> Any:
        """Run a single test iteration with all-gather across ranks. Override to customize."""
        model = state["model"]
        test_dataiter = state["test_dataiter"]

        model.eval()
        losses = []
        for data in test_dataiter:
            data = nested_to_device(data, self.device)
            with torch.amp.autocast("cuda", enabled=self.config.amp, dtype=self.amp_dtype):
                output = model(data["x"])
                loss = F.mse_loss(output, data["y"])
                losses.append(loss.item())

        # collect losses from all ranks
        collected_sizes = [None] * self.world_size
        torch.distributed.all_gather_object(collected_sizes, len(losses))

        collected_metrics = [torch.empty(size, device=self.device) for size in collected_sizes]
        torch.distributed.all_gather(collected_metrics, torch.tensor(losses, device=self.device))
        collected_metrics = torch.cat(collected_metrics)

        avg_loss = collected_metrics.mean().item()
        self.print_on_master(f"Average loss: {avg_loss}")
        if self.world_rank == 0:
            self.writer.add_scalar("test/loss", avg_loss, step)
        return avg_loss

    def load_state_dict_to_model(self, state_dict: dict, model: torch.nn.Module) -> None:
        """Load a state dict into the model. Override to customize loading behavior."""
        self._loosely_load_state_dict_to_model(state_dict, model)

    def load_state_dict_to_optimizer(
        self, state_dict: dict, optimizer: torch.optim.Optimizer
    ) -> None:
        """Load a state dict into the optimizer."""
        optimizer.load_state_dict(state_dict)

    def load_state_dict_to_scheduler(
        self, state_dict: dict, scheduler: torch.optim.lr_scheduler._LRScheduler | None
    ) -> None:
        """Restore scheduler state by replaying steps up to the saved epoch."""
        if scheduler is not None:
            saved_last_epoch = state_dict["last_epoch"]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for _ in range(saved_last_epoch):
                    scheduler.step()

    def save_checkpoint(self, step: int, state: Any) -> None:
        """Save a training checkpoint on rank 0."""
        if self.world_rank != 0:
            return
        if step == 0:
            return
        if step % self.config.ckpt_every == 0 or step == self.config.max_steps:
            state_dict = {"step": step}
            if self.world_size > 1:
                model = state["model"].module
            else:
                model = state["model"]
            # torch.compile adds this prefix to the model
            # https://github.com/pytorch/pytorch/issues/101107#issuecomment-1869839379
            state_dict["model"] = getattr(model, "_orig_mod", model).state_dict()
            state_dict["optimizer"] = state["optimizer"].state_dict()
            if state["scheduler"] is not None:
                state_dict["scheduler"] = state["scheduler"].state_dict()
            torch.save(state_dict, f"{self.ckpt_dir}/step-{step:09d}.pt")

    def maybe_resume(self, state: Any) -> int:
        """Load checkpoint if resuming, and return the starting step."""
        step = 0
        # From scratch
        if (not self.config.auto_resume) and (self.config.resume is None):
            return step

        ckpt_candidates = []
        if self.config.resume:
            assert os.path.exists(self.config.resume), (
                f"Checkpoint {self.config.resume} not found."
            )
            ckpt_candidates = [self.config.resume]
        elif self.config.auto_resume:
            # sort to put latest checkpoint first
            ckpt_candidates = sorted(glob.glob(f"{self.ckpt_dir}/*"), reverse=True)

        for i, ckpt_path in enumerate(ckpt_candidates, start=-len(ckpt_candidates) + 1):
            try:
                # It is possible that the checkpoint is corrupted due to brutal shutdown.
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            except Exception as e:  # noqa: BLE001 - checkpoint loading may raise any error
                self.print_on_master(
                    f"Error loading checkpoint {ckpt_path}: {e}. Try next candidate."
                )
                if i == 0:
                    raise ValueError(f"No available checkpoint was found in {self.ckpt_dir}.")
                continue

            self.load_state_dict_to_model(ckpt["model"], state["model"])
            if not self.config.test_only:
                step = ckpt.get("step", 0) + 1
                if not self.config.only_model:
                    self.load_state_dict_to_optimizer(ckpt["optimizer"], state["optimizer"])
                    self.load_state_dict_to_scheduler(
                        ckpt["scheduler"], state.get("scheduler", None)
                    )
            elif self.config.test_only:
                step = ckpt.get("step", 0)
            self.print_on_master(f"Resuming from ckpt: {ckpt_path}. set step to: {step}")
            break
        return step

    def _loosely_load_state_dict_to_model(self, state_dict: dict, model: torch.nn.Module) -> None:
        """Load state dict with mismatched keys/shapes filtered out."""
        # torch.compile might introduces this prefix
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        model = getattr(model, "_orig_mod", model)
        # Filter out parameters that do not match with the model (via key and shape check).
        state_dict_filtered = {}
        for k, v in state_dict.items():
            if k not in model.state_dict():
                print(f"Warning: {k} in ckpt but not in model state_dict.")
                continue
            if model.state_dict()[k].shape != v.shape:
                print(f"Warning: {k} shape mismatch: {model.state_dict()[k].shape} vs {v.shape}")
                continue
            state_dict_filtered[k] = v
        model.load_state_dict(state_dict_filtered, strict=False)
        self.print_on_master(
            f"Loosly loaded ckpt to model: "
            f"{len(state_dict)} keys in ckpt, "
            f"{len(state_dict_filtered)} keys loaded, "
            f"{len(model.state_dict())} keys in model."
        )

    def train(self):
        """Execute the distributed training loop."""
        print(f"Distributed worker: {self.world_rank + 1} / {self.world_size}")

        if self.config.fixed_seed:
            set_random_seed(self.config.seed + self.world_rank)
        torch.cuda.set_device(self.local_rank)

        # Initialize model, dataset, optimizer, scheduler ... and load checkpoint if needed
        state = self.train_initialize()
        init_step = self.maybe_resume(state)
        if self.config.test_every > 0:
            test_state = self.test_initialize(model=state["model"])

        for key in ["model", "optimizer"]:
            assert key in state, f"{key} is not in state."
        self.logging_on_master(
            f"Total trainable parameters: "
            f"{sum(p.numel() for p in state['model'].parameters() if p.requires_grad)}"
        )

        # torch.distributed.run ensures that this will work
        # by exporting all the env vars needed to initialize the process group
        torch.distributed.init_process_group(backend="nccl")

        # To device. Use DDP if needed.
        for k, v in state.items():
            if isinstance(v, torch.nn.Module):
                v = v.to(self.device)
                if self.world_size > 1 and k == "model":
                    v = DDP(v, device_ids=[self.local_rank])
                state[k] = v

        if self.use_grad_scaler:
            grad_scaler = torch.amp.GradScaler(device="cuda")

        # First test
        _ = self.test_iteration(init_step, test_state)
        # Training loop.
        for step in range(init_step, self.config.max_steps + 1):
            for acc_step in range(self.config.acc):
                # Train iteration.
                loss = self.train_iteration(step, state, acc_step)
                loss = loss / self.config.acc

                if loss.isnan():
                    if self.use_grad_scaler:
                        # with grad_scaler we could safely skip this step
                        print(f"Warning: [step={step}] rank={self.world_rank} | loss is NaN.")
                    else:
                        # without grad_scaler, we should exit.
                        print(f"Fatal: [step={step}] rank={self.world_rank} | loss is NaN.")
                        # exit()

                # Backward.
                if self.use_grad_scaler:
                    grad_scaler.scale(loss).backward()
                else:
                    loss.backward()

                # Clean grads from nan contamination
                with torch.no_grad():
                    for p in state["model"].parameters():
                        if p.requires_grad and (p.grad is not None):
                            p.grad.nan_to_num_(nan=0.0, posinf=1e-6, neginf=-1e-6)

                # For debugging.
                if self.config.check_nan_in_params:
                    for name, param in state["model"].named_parameters():
                        if torch.isnan(param).any():
                            print(f"[step={step}] rank={self.world_rank} | {name} has NaN.")
                        if param.grad is not None and torch.isnan(param.grad).any():
                            print(f"[step={step}] rank={self.world_rank} | {name} grad has NaN.")

            # Update model.
            model = state["model"]
            optimizer = state["optimizer"]
            scheduler = state.get("scheduler", None)

            if self.use_grad_scaler:
                grad_scaler.unscale_(optimizer)
                if self.config.grad_clip > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.config.grad_clip
                    )
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                if self.config.grad_clip > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.config.grad_clip
                    )
                optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()

            if step % self.config.print_every == 0 and self.world_rank == 0:
                self.writer.add_scalar("train/grad_norm", grad_norm.item(), step)

            # Save checkpoint
            self.save_checkpoint(step, state)

            # Test
            if self.config.test_every > 0 and step % self.config.test_every == 0 and step > 0:
                _ = self.test_iteration(step, test_state)

        # Exit.
        torch.distributed.destroy_process_group()

    def test(self):
        """Execute the distributed testing loop."""
        assert self.config.resume is not None or self.config.auto_resume, (
            "Resume checkpoint or auto_resume must be provided for testing."
        )
        print(f"Distributed worker: {self.world_rank + 1} / {self.world_size}")

        if self.config.fixed_seed:
            set_random_seed(self.config.seed + self.world_rank)
        torch.cuda.set_device(self.local_rank)

        # Initialize model, dataset, ... and load checkpoint if needed
        state = self.test_initialize()
        init_step = self.maybe_resume(state)

        for key in ["model"]:
            assert key in state, f"{key} is not in state."

        # torch.distributed.run ensures that this will work
        # by exporting all the env vars needed to initialize the process group
        torch.distributed.init_process_group(backend="nccl")

        # To device. Test does not need gradient accumulation so no need for DDP.
        for k, v in state.items():
            if isinstance(v, torch.nn.Module):
                v = v.to(self.device)
                state[k] = v

        # Run test.
        _ = self.test_iteration(init_step, state)

        # Exit.
        torch.distributed.destroy_process_group()

    def print_on_master(self, msg: str) -> None:
        """Print a message only on rank 0."""
        if self.world_rank == 0:
            print(msg)

    def logging_on_master(self, msg: str) -> None:
        """Log a timestamped message to file and stdout on rank 0."""
        if self.world_rank == 0:
            msg = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
            with open(f"{self.output_dir}/log.txt", "a") as logger:
                logger.write(msg + "\n")
            print(msg)


if __name__ == "__main__":
    """Example usage:

    Note: somali needs `NCCL_P2P_DISABLE=1`

    # 2 GPUs training
    OMP_NUM_THREADS=1 torchrun --standalone --nnodes=1 --nproc-per-node=2 \
        <THIS_SCRIPT.py> (args) 

    # 2 GPUs testing
    OMP_NUM_THREADS=1 torchrun --standalone --nnodes=1 --nproc-per-node=2 \
        <THIS_SCRIPT.py> --resume results/dbg/ckpts/step-000000100.pt --test_only

    # 2 GPUs x 2 nodes training
    NCCL_DEBUG=INFO OMP_NUM_THREADS=1 torchrun --nnodes=2 --nproc-per-node=2 --node_rank=0 \
        --rdzv-backend=c10d --rdzv-endpoint=10.55.10.177:29603 lvsm/runner.py
    NCCL_DEBUG=INFO OMP_NUM_THREADS=1 torchrun --nnodes=2 --nproc-per-node=2 --node_rank=1 \
        --rdzv-backend=c10d --rdzv-endpoint=10.55.10.177:29603 lvsm/runner.py
    """
    # Flash attention only supports gpu architectures in the range [sm80, sm90]
    # So we use Memory Efficient Attention.
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)

    cfg = tyro.cli(LauncherConfig)
    launcher = Launcher(cfg)
    launcher.run()
