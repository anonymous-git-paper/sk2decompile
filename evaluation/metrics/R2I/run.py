import os
import sys
import subprocess
import shutil
from pathlib import Path
from metrics.R2I.src.aggregator import aggregate_features
from metrics.R2I.src.evaluator import evaluate_targets
from metrics.R2I.src.mean import compute_mean_rank
from metrics.R2I.src.extractor import run_extraction
import pandas as pd
import json

current_dir = os.path.dirname(os.path.abspath(__file__))

def run(decompilers, func_names):
    experiment = "test"
    # default_decompilers = ["angr", "bn", "ghidra", "ida", "radare2", "retdec"]
    # decompilers = ["predictions", "sources"]
    weight_version = "paper"

    BASE = Path(__file__).resolve().parent

    all_scores = []
    stripped = True

    for d in decompilers:
        dataset_dir = BASE / "dataset" / experiment / d
        # extract_dir = BASE / "extract" / experiment / d
        # extract_dir.mkdir(parents=True, exist_ok=True)

        c_dir = dataset_dir / "c"
        if not c_dir.exists():
            continue

        # Count .c files
        count = sum(1 for f in c_dir.glob("*") if f.suffix == ".c")

        for file in c_dir.glob("*"):
            if file.suffix != ".c":
                continue

            relative_file = file.name
            index = int(relative_file.replace('file','').replace('.c',''))
            df = run_extraction(filename=relative_file, path=f"{experiment}/{d}", debug=False, save=False, func_name = func_names[index])
            if stripped == False:
                df.insert(0, '_decompiler', d)
                df.insert(1, '_binary', relative_file.split('-')[0])
                all_scores.append(df)
            else:
                funcs = []
                for i, func in enumerate(df.values):
                    if func[0].startswith('fts_') or func[0].startswith('obstack_'):
                        func[0] = func[0].replace('_', '')
                    try:
                        funcname = '0x' + func[0].split('_')[1].lower().lstrip('0').zfill(6)
                        funcs.append(funcname)
                    except:
                        funcs.append(func[0])  
                df['_func_name'] = funcs
                df.insert(0, '_decompiler', d)
                df.insert(1, '_binary', relative_file[:-2])
                all_scores.append(df)          

        # Clean yacc/lex generated files
        for f in ["yacctab.py", "lextab.py"]:
            try:
                os.remove(f)
            except FileNotFoundError:
                pass
       
    agg_df = pd.concat(all_scores)        
    agg_df.drop_duplicates(inplace=True)
    agg_df.sort_values(by=['_binary', '_func_name'], inplace=True)

    eval_dir = BASE / "eval" / experiment
    eval_dir.mkdir(parents=True, exist_ok=True)

    # 1. 进入 BASE/extract 目录并运行 aggregator.py
    # os.chdir(BASE / "extract")
    # agg_df = aggregate_features(experiment, decompilers, stripped=True)

    # 2. 切换目录到 BASE/eval/$experiment 并准备 bin 目录
    os.chdir(BASE / "eval" / experiment)
    os.makedirs("bin", exist_ok=True)

    # 3. 运行 evaluator.py（此时工作目录正确）
    evaluate_targets(decompilers, weight_version, agg_df)

    # 4. 运行 mean.py（工作路径仍是 BASE/eval/$experiment，符合原始行为）
    compute_mean_rank(experiment, decompilers)

    shutil.rmtree(BASE / "extract", ignore_errors=True)
    shutil.rmtree(BASE / "eval" / experiment / "bin", ignore_errors=True)


def run_r2i(decompilers, func_names):
    os.chdir(current_dir)
    run(decompilers, func_names)
    data = pd.read_csv(f'{current_dir}/eval/test/r2i.csv')
    scores = {}
    counts = {}
    total = 0
    for index, row in data.iterrows():
        if row['decompiler'] in scores:
            scores[row['decompiler']] += row['r2i']
            counts[row['decompiler']] += 1
        else:
            scores[row['decompiler']] = row['r2i']
            counts[row['decompiler']] = 1  
    for key in scores:
        scores[key] = scores[key] / counts[key]

    root_folder = f'{current_dir}/dataset'
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            os.remove(file_path)
    return scores