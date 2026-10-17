"""Patch build_notebook.py: fix Trainer() tokenizer -> processing_class compat."""
path = '/home/ardraj/ARC/Mtech/notebooks/build_notebook.py'
content = open(path).read()

old = (
    '    "    trainer = Trainer(\\n"\n'
    '    "        model=model, args=args, train_dataset=ds.get(\'train\'),\\n"\n'
    '    "        eval_dataset=ds.get(\'validation\'), tokenizer=tok,\\n"\n'
    '    "        data_collator=get_data_collator(tok),\\n"\n'
    '    "        compute_metrics=make_compute_metrics_fn(cfg[\'task\'], tok),\\n"\n'
    '    "    )\\n"\n'
)

new = (
    '    "    # \'tokenizer\' was renamed to \'processing_class\' in transformers>=4.47\\n"\n'
    '    "    _TR_PARAMS = set(inspect.signature(Trainer.__init__).parameters)\\n"\n'
    '    "    tok_kw = \'processing_class\' if \'processing_class\' in _TR_PARAMS else \'tokenizer\'\\n"\n'
    '    "    trainer = Trainer(\\n"\n'
    '    "        model=model, args=args, train_dataset=ds.get(\'train\'),\\n"\n'
    '    "        eval_dataset=ds.get(\'validation\'), **{tok_kw: tok},\\n"\n'
    '    "        data_collator=get_data_collator(tok),\\n"\n'
    '    "        compute_metrics=make_compute_metrics_fn(cfg[\'task\'], tok),\\n"\n'
    '    "    )\\n"\n'
)

assert old in content, "Pattern not found! Check build_notebook.py manually."
content = content.replace(old, new, 1)
open(path, 'w').write(content)
print("✅ Patched Trainer() tokenizer kwarg")
