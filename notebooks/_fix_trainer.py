"""One-shot patch: remove logging_dir and change report_to in build_notebook.py"""
path = '/home/ardraj/ARC/Mtech/notebooks/build_notebook.py'
content = open(path).read()

# Remove the logging_dir line
old1 = "    \"        'logging_dir': str(out / 'logs'),\\n\"\n"
assert old1 in content, f"logging_dir line not found!\nRepr: {repr(old1)}"
content = content.replace(old1, '', 1)
print("✅ Removed logging_dir line")

# Change report_to from tensorboard to none
old2 = "    \"        'report_to': 'tensorboard',\\n\"\n"
new2 = "    \"        'report_to': 'none',\\n\"\n"
assert old2 in content, f"report_to line not found!\nRepr: {repr(old2)}"
content = content.replace(old2, new2, 1)
print("✅ Changed report_to to 'none'")

open(path, 'w').write(content)
print("✅ Saved build_notebook.py")
