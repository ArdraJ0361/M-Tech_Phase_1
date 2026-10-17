"""Patch build_notebook.py: fix OOM in Cell 6B by reducing memory footprint."""
path = '/home/ardraj/ARC/Mtech/notebooks/build_notebook.py'
content = open(path).read()

# ── 1. Reduce squad_qa config memory usage ─────────────────────────────────
old_squad = (
    '  max_source_length: 512\n'
    '  max_target_length: 64\n'
    '  train_samples: 5000\n'
    '  val_samples: 500\n'
    '  test_samples: 500\n'
    'training:\n'
    '  output_dir: "experiments/squad_qa"\n'
    '  num_train_epochs: 3\n'
    '  per_device_train_batch_size: 4\n'
    '  per_device_eval_batch_size: 8\n'
    '  gradient_accumulation_steps: 4\n'
    '  learning_rate: 2.0e-4\n'
    '  weight_decay: 0.01\n'
    '  warmup_ratio: 0.03\n'
    '  lr_scheduler_type: "cosine"\n'
    '  fp16: false\n'
    '  bf16: true\n'
)
new_squad = (
    '  max_source_length: 384\n'   # 512→384: cuts sequence memory ~25%
    '  max_target_length: 48\n'
    '  train_samples: 5000\n'
    '  val_samples: 500\n'
    '  test_samples: 500\n'
    'training:\n'
    '  output_dir: "experiments/squad_qa"\n'
    '  num_train_epochs: 3\n'
    '  per_device_train_batch_size: 2\n'   # 4→2: halves batch memory
    '  per_device_eval_batch_size: 4\n'    # 8→4
    '  gradient_accumulation_steps: 8\n'   # 4→8: keeps effective batch=16
    '  learning_rate: 2.0e-4\n'
    '  weight_decay: 0.01\n'
    '  warmup_ratio: 0.03\n'
    '  lr_scheduler_type: "cosine"\n'
    '  fp16: true\n'    # fp16 uses less memory than bf16 on T4
    '  bf16: false\n'
)
assert old_squad in content, "squad config pattern not found"
content = content.replace(old_squad, new_squad, 1)
print("✅ Patched squad_qa config (reduced batch/seq size, fp16)")

# ── 2. Add gradient_checkpointing to TrainingArguments kwargs ───────────────
old_ta = (
    "    \"        'dataloader_pin_memory': False,\\n\"\n"
    "    \"    }\\n\"\n"
)
new_ta = (
    "    \"        'dataloader_pin_memory': False,\\n\"\n"
    "    \"        'gradient_checkpointing': True,\\n\"\n"  # saves ~30% VRAM
    "    \"    }\\n\"\n"
)
assert old_ta in content, "dataloader_pin_memory pattern not found"
content = content.replace(old_ta, new_ta, 1)
print("✅ Added gradient_checkpointing=True to TrainingArguments")

open(path, 'w').write(content)
print("✅ Saved build_notebook.py")
