"""Native MiniZinc compilation checks."""
import os
import subprocess
import tempfile

from backends import run_tool, ToolError, check_record


def mzn_check(model_text, minizinc_cmd, solver="highs", include_dirs=(), *, instance_id):
    """Compile self-contained model text without solving."""
    with tempfile.TemporaryDirectory(prefix="mzn_check_") as directory:
        path = os.path.join(directory, "solution.mzn")
        with open(path, "w", encoding="utf-8") as stream:
            stream.write(model_text)
        command = [minizinc_cmd, "-c", "--solver", solver]
        for parent in dict.fromkeys(include_dirs):
            command += ["-I", os.path.abspath(parent)]
        command.append(path)
        try:
            proc = run_tool(command, cwd=directory, capture_output=True,
                                  text=True, timeout=300)
            out = (proc.stdout or "") + (proc.stderr or "")
            return check_record(instance_id, 'passed' if proc.returncode == 0 else 'failed',
                None if proc.returncode == 0 else out.strip() or 'MiniZinc compilation failed (no error output)')
        except (ToolError, subprocess.TimeoutExpired, OSError) as error:
            return check_record(instance_id, 'failed', failure=error)
