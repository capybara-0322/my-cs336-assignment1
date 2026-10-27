import argparse
import wandb
import torch
import os
import time
from cs336_basics.grad_clip import gradient_clipping
from cs336_basics.checkpointing import save_checkpoint, load_checkpoint
from cs336_basics.cross_entropy import cross_entropy
from cs336_basics.cosine_lr_schedule import learning_rate_schedule
from cs336_basics.adamw import AdamW
from cs336_basics.transformer import TransformerLM
from cs336_basics.data_loading import data_loading
from cs336_basics.config import Config
import numpy as np


def parse_args() -> Config:
    """Use Config defaults, allowing command-line overrides."""
    defaults = Config()
    parser = argparse.ArgumentParser(description="Train the CS336 Transformer")
    for name, field in type(defaults).model_fields.items():
        default = getattr(defaults, name)
        flag = "--" + name.replace("_", "-")
        if name == "betas":
            parser.add_argument(flag, type=float, nargs=2, metavar=("BETA1", "BETA2"), default=None)
        elif name in {"device", "dtype", "wandb_mode"}:
            choices = {
                "device": ["cpu", "cuda"],
                "dtype": ["float32", "bfloat16", "float16"],
                "wandb_mode": ["online", "offline", "disabled"],
            }[name]
            parser.add_argument(flag, type=str, choices=choices, default=None)
        else:
            arg_type = str if default is None else type(default)
            parser.add_argument(flag, type=arg_type, default=None)
    args = vars(parser.parse_args())
    overrides = {name: value for name, value in args.items() if value is not None}
    if "betas" in overrides:
        overrides["betas"] = tuple(overrides["betas"])
    config = Config(**overrides)
    if config.eval_interval < 1 or config.save_interval < 1 or config.log_interval < 1:
        parser.error("eval-interval, save-interval, log-interval must be positive")
    if config.eval_iters < 1 or config.num_steps < 1 or config.batch_size < 1:
        parser.error("eval-iters, num-steps, batch-size must be positive")
    if not (0 <= config.warmup_iters < config.cosine_cycle_iters):
        parser.error("require 0 <= warmup-iters < cosine-cycle-iters")
    if config.min_lr > config.max_lr:
        parser.error("min-lr cannot exceed max-lr")
    if config.d_model % config.num_heads != 0:
        parser.error("d-model must be divisible by num-heads")
    return config


dtype_map = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}

def evaluate(model, valid_dataset, config: Config):
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(config.eval_iters):
            x, y = data_loading(
                valid_dataset,
                config.batch_size,
                config.context_length,
                config.device,
            )

            logits = model(x)
            loss = cross_entropy(
                logits.reshape(-1, config.vocab_size),
                y.reshape(-1),
            )
            losses.append(loss.item())

    model.train()
    return sum(losses) / len(losses)


def train(config: Config):
    # Ensure logging/checkpoint destinations exist.
    os.makedirs(config.wandb_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    run = wandb.init(
        project=config.wandb_project,
        name=config.wandb_run_name,
        config=config.model_dump(),
        mode=config.wandb_mode,
        dir=config.wandb_dir,
    )
    run.define_metric("train/step")
    run.define_metric("train/*", step_metric="train/step")
    run.define_metric("valid/*", step_metric="train/step")
    run.define_metric("time/*", step_metric="train/step")

    try:
        # 加载训练数据集
        train_dataset = np.load(config.train_data_path, mmap_mode="r")
        valid_dataset = np.load(config.valid_data_path, mmap_mode="r")

        # 初始化各组件
        model = TransformerLM(
            vocab_size=config.vocab_size, 
            max_seq_len=config.context_length,
            d_model=config.d_model,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            d_ff=config.d_ff,
            theta=config.rope_theta,
            device=torch.device(config.device),
            dtype=dtype_map[config.dtype]
        )

        optimizer = AdamW(
            params=model.parameters(),
            lr=config.max_lr,
            weight_decay=config.weight_decay,
            betas=config.betas,
            eps=config.eps
        )
        start_step = 0
        # 恢复checkpoint
        if config.resume_path is not None:
            last_step = load_checkpoint(config.resume_path, model, optimizer)
            start_step = last_step + 1
        
        train_start_time = time.perf_counter()
        # CUDA kernels are asynchronous; use synchronized timing for meaningful throughput.
        if config.device == "cuda":
            torch.cuda.synchronize()
        last_log_time = time.perf_counter()
        best_valid_loss = float("inf")
        tokens_per_step = config.batch_size * config.context_length

        if start_step >= config.num_steps:
            print(f"Checkpoint already reached {start_step} steps (target {config.num_steps}); nothing to train.")
            return

        for step in range(start_step, config.num_steps):
            # 更新学习率
            lr = learning_rate_schedule(
                t = step,
                a_max=config.max_lr,
                a_min=config.min_lr,
                t_w=config.warmup_iters,
                t_c=config.cosine_cycle_iters
            )
            for group in optimizer.param_groups:
                group["lr"] = lr

            # 将模型切换到训练模式
            model.train()
            # 随机采样训练数据
            x, y = data_loading(train_dataset, config.batch_size, config.context_length, config.device)
            # 前向传播
            logits = model(x)
            # 计算损失值
            loss = cross_entropy(logits.reshape(-1, config.vocab_size), y.reshape(-1))
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            gradient_clipping(model.parameters(), config.max_l2_norm)  # ty:ignore[unresolved-attribute]
            optimizer.step()

            # 实验记录
            if config.device == "cuda":
                torch.cuda.synchronize()
            now = time.perf_counter()
            step_time_sec = now - last_log_time
            elapsed_sec = now - train_start_time
            last_log_time = now
            tokens_seen = (step + 1) * tokens_per_step # 处理的总token数
            tokens_per_sec = tokens_per_step / step_time_sec # 每秒处理多少个 token
            log_dict = {
                "train/loss": loss.item(),
                "train/lr": lr,
                "train/step": step,
                "train/tokens_seen": tokens_seen,
                "time/step_time_sec": step_time_sec,
                "time/elapsed_sec": elapsed_sec,
                "time/tokens_per_sec": tokens_per_sec,
            }

            # 验证
            if (step + 1) % config.eval_interval == 0 or step == config.num_steps - 1:
                valid_loss = evaluate(model, valid_dataset, config)
                best_valid_loss = min(best_valid_loss, valid_loss)

                log_dict["valid/loss"] = valid_loss
                log_dict["valid/best_loss"] = best_valid_loss
            
            run.log(log_dict, step=step)
            if (step + 1) % config.log_interval == 0 or step == start_step:
                msg = (
                    f"step={step}/{config.num_steps} "
                    f"train_loss={loss.item():.4f} "
                    f"lr={lr:.2e} "
                    f"tokens_per_sec={tokens_per_sec:.0f} "
                    f"elapsed={elapsed_sec / 60:.1f}min"
                )

                if "valid/loss" in log_dict:
                    msg += f" valid_loss={log_dict['valid/loss']:.4f}"

                print(msg, flush=True)

            # 保存checkpoint
            if (step + 1) % config.save_interval == 0:
                os.makedirs(config.checkpoint_dir, exist_ok=True)
                save_checkpoint(model, optimizer, step, config.latest_checkpoint_path)
                run.log(
                    {
                        "checkpoint/step": step,
                    },
                    step=step,
                )
        # 保存训练好的模型
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        save_checkpoint(model, optimizer, step, config.latest_checkpoint_path)
        run.log({"checkpoint/step": step}, step=step)
        print(f"saved final checkpoint at step={step}", flush=True)

        # 实验总结记录  
        run.summary["final_step"] = step
        run.summary["best_valid_loss"] = best_valid_loss
        run.summary["total_elapsed_sec"] = time.perf_counter() - train_start_time
        run.summary["total_tokens_seen"] = (step + 1) * tokens_per_step
    finally: 
        run.finish()

if __name__ == "__main__":
    train(parse_args())
