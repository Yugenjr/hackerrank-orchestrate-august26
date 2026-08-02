import os

scripts = ['evaluate.py', 'generalization_audit.py', 'calibration_audit.py', 'chaos_test.py', 'performance_benchmark.py']

for s in scripts:
    if os.path.exists(f'code/{s}'):
        print(f'--- {s} ---')
        with open(f'code/{s}', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if '.md' in line or 'open(' in line:
                    print(f'{i}: {line.strip()}')
