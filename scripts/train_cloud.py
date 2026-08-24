"""Single-GPU CADmium QLoRA launcher with a local configuration dry-run."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import yaml


def load_config(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("配置文件顶层必须是对象")
    for section in ("data", "model", "training"):
        if not isinstance(value.get(section), dict):
            raise ValueError(f"配置缺少对象：{section}")
    return value


def resolve_paths(config: dict, project_root: Path) -> dict[str, Path]:
    return {
        "train": project_root / config["data"]["train_qwen_tokenized_parquet_path"],
        "validation": project_root / config["data"]["validation_qwen_tokenized_parquet_path"],
    }


def audit_parquet(path: Path) -> dict:
    from datasets import load_dataset

    dataset = load_dataset("parquet", data_files={"data": str(path)})["data"]
    required = {"uid", "input_ids", "attention_mask", "labels"}
    missing = required - set(dataset.column_names)
    if missing:
        raise ValueError(f"{path} 缺少列：{sorted(missing)}")
    zero_supervised = 0
    padding_labels_not_masked = 0
    length_mismatch = 0
    for row in dataset:
        if not (len(row["input_ids"]) == len(row["attention_mask"]) == len(row["labels"])):
            length_mismatch += 1
        if not any(label != -100 for label in row["labels"]):
            zero_supervised += 1
        padding_labels_not_masked += sum(
            not attended and label != -100
            for attended, label in zip(row["attention_mask"], row["labels"])
        )
    return {
        "path": str(path.resolve()),
        "rows": len(dataset),
        "columns": dataset.column_names,
        "zero_supervised": zero_supervised,
        "padding_labels_not_masked": padding_labels_not_masked,
        "length_mismatch": length_mismatch,
    }


def validate(config: dict, project_root: Path) -> dict:
    paths = resolve_paths(config, project_root)
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{name} Parquet 不存在：{path}")
    reports = {name: audit_parquet(path) for name, path in paths.items()}
    invalid = {
        name: report
        for name, report in reports.items()
        if report["zero_supervised"]
        or report["padding_labels_not_masked"]
        or report["length_mismatch"]
    }
    if invalid:
        raise ValueError(f"训练数据标签审计失败：{invalid}")
    if "fsdp" in config["training"]:
        raise ValueError("单卡 QLoRA 配置不得包含 FSDP")
    if int(config["training"].get("per_device_train_batch_size", 0)) != 1:
        raise ValueError("24GB smoke test 的 per_device_train_batch_size 必须为 1")
    quantize = config["model"].get("quantize", {})
    if not quantize.get("load_in_4bit"):
        raise ValueError("24GB 资源版必须显式启用 load_in_4bit")
    return {"data": reports, "config_valid": True}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs_repro/cloud_qlora_1.5b_smoke.yaml")
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    config = load_config(args.config)
    validation = validate(config, project_root)

    if args.dry_run:
        validation["result"] = "配置、Parquet、padding labels 和监督 token 验证通过；未加载模型。"
        print(json.dumps(validation, indent=2, ensure_ascii=False))
        return 0

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, set_seed
    from transformers.trainer_utils import get_last_checkpoint
    from trl import SFTTrainer

    if not torch.cuda.is_available():
        raise RuntimeError(f"训练需要 CUDA；当前 torch={torch.__version__}, CUDA={torch.version.cuda}")
    if torch.cuda.get_device_properties(0).total_memory < 20 * 1024**3:
        raise RuntimeError("训练设备显存不足 20 GiB；本配置面向标称至少 24GB 的 GPU")

    set_seed(int(config.get("seed", 42)))
    dtype_names = {"float16": torch.float16, "bfloat16": torch.bfloat16}
    quantize = dict(config["model"]["quantize"])
    dtype_value = quantize.get("bnb_4bit_compute_dtype", "bfloat16")
    if isinstance(dtype_value, str):
        if dtype_value not in dtype_names:
            raise ValueError(f"不支持的 bnb_4bit_compute_dtype：{dtype_value}")
        quantize["bnb_4bit_compute_dtype"] = dtype_names[dtype_value]
    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["model_name"],
        quantization_config=BitsAndBytesConfig(**quantize),
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=bool(config["training"].get("gradient_checkpointing", True)),
    )

    lora = config["model"]["lora"]
    peft_config = LoraConfig(
        r=int(lora["r"]),
        lora_alpha=int(lora["lora_alpha"]),
        target_modules=lora["target_modules"],
        lora_dropout=float(lora["lora_dropout"]),
        bias=lora["bias"],
        task_type=lora["task_type"],
    )
    paths = resolve_paths(config, project_root)
    train_dataset = load_dataset("parquet", data_files={"train": str(paths["train"])})["train"]
    validation_dataset = load_dataset(
        "parquet", data_files={"validation": str(paths["validation"])}
    )["validation"]

    training_values = dict(config["training"])
    output_dir = project_root / training_values["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    training_values["output_dir"] = str(output_dir)
    supported_args = set(inspect.signature(TrainingArguments).parameters)
    unknown = set(training_values) - supported_args
    if unknown:
        raise ValueError(f"当前 transformers 不支持训练参数：{sorted(unknown)}")
    (output_dir / "resolved_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    trainer = SFTTrainer(
        model=model,
        args=TrainingArguments(**training_values),
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        peft_config=peft_config,
    )
    checkpoint = get_last_checkpoint(str(output_dir))
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model(str(output_dir / "final_adapter"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
