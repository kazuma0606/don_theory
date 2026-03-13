"""全図を一括生成するスクリプト"""
import subprocess, sys

scripts = ["noncommutativity.py", "mollifier.py"]

for script in scripts:
    print(f"\n{'='*40}")
    print(f"実行: {script}")
    print('='*40)
    subprocess.run([sys.executable, script], check=True)

print("\n✅ 全図の生成完了。visualization/img/ を確認してください。")
