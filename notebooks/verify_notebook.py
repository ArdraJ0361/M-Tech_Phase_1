import json, ast, pathlib, tempfile, sys

nb = json.loads(pathlib.Path('/home/ardraj/ARC/Mtech/notebooks/05_colab_ready.ipynb').read_text())
cell2_src = next(c['source'] for c in nb['cells']
                 if c.get('cell_type') == 'code' and '_w(' in c['source'])

with tempfile.TemporaryDirectory() as tmp:
    src = cell2_src.replace('/content/Mtech', tmp)
    exec(compile(src, '<cell2>', 'exec'), {'__builtins__': __builtins__})
    errors = []
    for pyf in sorted(pathlib.Path(tmp).rglob('*.py')):
        try:
            ast.parse(pyf.read_text(encoding='utf-8'))
        except SyntaxError as e:
            errors.append(f'{pyf.relative_to(tmp)}: {e}')
    if errors:
        print('SYNTAX ERRORS:')
        for e in errors:
            print(e)
        sys.exit(1)
    else:
        files = sorted(str(p.relative_to(tmp)) for p in pathlib.Path(tmp).rglob('*.py'))
        print(f'All {len(files)} .py files parse correctly:')
        for f in files:
            print(f'  OK  {f}')
