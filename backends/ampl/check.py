"""Original-model AMPL loading and expansion checks."""
import os
import subprocess
import tempfile

from backends import run_tool, ToolError, ToolConfigurationError, check_record
from . import contracts as mod_contracts


def check_original(model_path, instance_id, ampl_cmd):
    """Load and expand a self-contained original model without invoking a solver."""
    try:
        solve_source = mod_contracts.solve_command_source([model_path], allow_model_trailer=True)
        if solve_source:
            return check_record(instance_id, 'failed',
                'Embedded solve command is not supported by original-model checking: ' + solve_source)
        if not ampl_cmd:
            raise ToolConfigurationError('AMPL executable is unavailable')
        with tempfile.TemporaryDirectory(prefix='ampl_preflight_') as directory:
            script = os.path.join(directory, 'check.run')
            expansion = os.path.join(directory, 'expanded.txt')
            with open(script, 'w', encoding='utf-8') as stream:
                stream.write('option eexit 1;\n')
                stream.write(mod_contracts.load_commands(model_path, [], directory, evaluation=True))
                stream.write('display _nvars, _ncons, _nobjs;\n')
                stream.write('expand > ' + mod_contracts.quote(expansion) + ';\n')
                stream.write('print "AMPL_PREFLIGHT_OK";\n')
            proc = run_tool([ampl_cmd, script], cwd=os.path.dirname(model_path),
                            capture_output=True, text=True, timeout=300)
            output = (proc.stdout or '') + (proc.stderr or '')
            ok = proc.returncode == 0 and 'AMPL_PREFLIGHT_OK' in (proc.stdout or '').splitlines()
            return check_record(instance_id, 'passed' if ok else 'failed',
                None if ok else output.strip()[-2000:] or 'AMPL preflight did not complete')
    except ToolError as error:
        return check_record(instance_id, 'failed', failure=error)
    except subprocess.TimeoutExpired as error:
        return check_record(instance_id, 'failed', 'AMPL preflight timed out (>300s)', failure=error)
    except OSError as error:
        return check_record(instance_id, 'failed', failure=error)
    except ValueError as error:
        return check_record(instance_id, 'failed', str(error))


def ampl_check(model_text, ampl_cmd, source_dir=None, *, instance_id):
    """Actually load self-contained model text with AMPL to validate syntax."""
    with tempfile.TemporaryDirectory(prefix='ampl_check_') as directory:
        mod_path = os.path.join(directory, 'solution.mod')
        with open(mod_path, 'w', encoding='utf-8') as stream:
            stream.write(model_text)
        script = os.path.join(directory, 'check.run')
        try:
            commands = mod_contracts.load_commands(mod_path, [], directory, evaluation=True)
        except ValueError as error:
            return check_record(instance_id, 'failed', str(error))
        with open(script, 'w', encoding='utf-8') as stream:
            stream.write('option eexit 1;\n')
            stream.write(commands)
            stream.write('expand > ' + mod_contracts.quote(os.path.join(directory, 'expanded.txt')) + ';\n')
            stream.write('print "AMPL_CHECK_OK";\n')
        try:
            proc = run_tool([ampl_cmd, script], capture_output=True,
                                  text=True, timeout=300, cwd=source_dir)
            out = (proc.stdout or '') + (proc.stderr or '')
            if proc.returncode == 0 and 'AMPL_CHECK_OK' in (proc.stdout or '').splitlines():
                return check_record(instance_id, 'passed')
        except (ToolError, subprocess.TimeoutExpired, OSError) as error:
            return check_record(instance_id, 'failed', failure=error)
        return check_record(instance_id, 'failed', out.strip()[:2000] or 'AMPL loading failed (no error output)')


def ampl_dump_vars(model_text, ampl_cmd, timeout=300, source_dir=None):
    """Load a self-contained model and export all variable names."""
    with tempfile.TemporaryDirectory(prefix='ampl_dump_') as directory:
        mod_path = os.path.join(directory, 'solution.mod')
        var_path = os.path.join(directory, 'variables.txt')
        script = os.path.join(directory, 'dump.run')
        try:
            with open(mod_path, 'w', encoding='utf-8') as stream:
                stream.write(model_text)
            with open(script, 'w', encoding='utf-8') as stream:
                stream.write('option eexit 1;\n')
                stream.write(mod_contracts.load_commands(mod_path, [], directory, evaluation=True))
                stream.write('printf {i in 1.._nvars}: "%s\\n", _varname[i] > '
                             + mod_contracts.quote(var_path) + ';\nprint "AMPL_DUMP_OK";\n')
            proc = run_tool([ampl_cmd, script], capture_output=True,
                                  text=True, timeout=timeout, cwd=source_dir)
            if proc.returncode != 0 or 'AMPL_DUMP_OK' not in (proc.stdout or '').splitlines():
                return None
            with open(var_path, 'r', encoding='utf-8') as stream:
                return [line.rstrip('\n') for line in stream if line.strip()]
        except ToolError:
            raise
        except (subprocess.TimeoutExpired, OSError, ValueError):
            return None
