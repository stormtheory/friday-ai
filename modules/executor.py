# modules/executor.py

import subprocess

DANGEROUS_PYTHON = ["import os", "import sys", "open(", "eval(", "exec(", "__import__"]

def run_python_code(code):
    if any(term in code for term in DANGEROUS_PYTHON):
        return "⚠️ This code is restricted for security reasons."

    local_scope = {}
    try:
        exec(code, {}, local_scope)
        output = local_scope.get('output', '✅ Code executed successfully.')
        return str(output)
    except Exception as e:
        return f"❌ Python error: {e}"

def run_bash_command(command):
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=3)
        if result.stderr:
            return f"⚠️ Bash error: {result.stderr.strip()}"
        return result.stdout.strip() or "✅ Command executed successfully."
    except Exception as e:
        return f"❌ Bash execution failed: {e}"

def handle_execution(user_input):
    user_input = user_input.lower()

    if "run python code:" in user_input:
        code = user_input.split("run python code:")[1].strip()
        return run_python_code(code)

    elif "run bash command:" in user_input:
        cmd = user_input.split("run bash command:")[1].strip()
        return run_bash_command(cmd)

    return "⚠️ Execution format not recognized. Try 'run python code: ...' or 'run bash command: ...'"
