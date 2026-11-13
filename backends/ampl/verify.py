"""Evaluate a mapped solution in its original AMPL model, without solving."""
import math
from pathlib import Path
import re
import subprocess
from backends import run_tool
import tempfile
import time

from . import contracts as contracts
from .solution import validate_values

OBJECTIVE_REL_TOL = 1e-4


def context(original, data_files, args):
    """Export the original model's parameter context for one verification."""
    from backends.verification import CONTEXT_TIMEOUT

    started = time.monotonic()

    def remaining():
        seconds = CONTEXT_TIMEOUT - (time.monotonic() - started)
        if seconds <= 0:
            raise subprocess.TimeoutExpired('AMPL verification context', CONTEXT_TIMEOUT)
        return seconds

    return contracts.instance_context(original, data_files, args.ampl_cmd, timeout=remaining)


def _set_domain(definition):
    """Recognize membership attributes without mistaking indexed sums for domains."""
    depth = 0
    for match in re.finditer(r'''"(?:[^"\\]|\\.|"")*"|'(?:[^'\\]|\\.|'')*'|:=|[A-Za-z_]\w*|<=|>=|!=|==|.''', definition, re.S):
        token = match[0]
        if token in ('(', '[', '{'):
            depth += 1
        elif token in (')', ']', '}'):
            depth -= 1
        elif depth == 0:
            if token in ('=', ':='):
                break
            if token == 'in':
                return True
    return False


def verify(original, assignments, args, data_files=()):
    """Check coverage, original domains, constraints, and the mapped objective.

    Native synonyms supply instantiated names and domain flags, including unused
    variables. Presolve and automatic equality substitution are disabled so that
    independent variables cannot disappear. Defined variables are evaluated by AMPL.
    """
    from backends.verification import feasibility_tolerance

    started = time.monotonic()
    result = dict(map_status='error', map_error='', map_objective=None,
                  map_objective_status='undetermined', map_objective_match=None,
                  map_checks_complete=False, verify_method='ampl_injection',
                  verify_feas_tol=None, verify_integer_tol=0,
                  verify_objective_rel_tol=OBJECTIVE_REL_TOL)
    stage = 'preparation'
    try:
        tolerance = feasibility_tolerance(args)
        result['verify_feas_tol'] = tolerance
        validate_values(assignments)
        source = contracts.solve_command_source([original, *data_files], allow_model_trailer=True)
        if source:
            raise ValueError('Embedded solve command is not supported by verification: ' + source)
        # Set-membership domains can have holes not represented by lb0/ub0 or
        # the integer suffix. Do not claim full domain coverage for these forms.
        unsupported = [row['name'] for path in contracts.sources(original)
                       for row in contracts.inspect_model(contracts.read_text(path))['declarations']
                       if row['kind'] == 'var' and _set_domain(row['definition'])]
        if unsupported:
            result.update(map_status='unknown', map_error='Unsupported set-membership variable domains: ' + ', '.join(unsupported),
                          map_error_stage='domains')
            return dict(result, verify_time_sec=round(time.monotonic()-started, 3))
        with tempfile.TemporaryDirectory(prefix='ampl_verify_') as directory:
            directory = Path(directory)
            script = directory / 'verify.run'
            # Instantiate every declared variable, even if no objective or constraint
            # references it. AMPL writes these commands from its own entity names.
            instantiate = directory / 'instantiate.run'
            instantiate.touch()
            warmup = directory / 'initial.txt'
            prefix = ('option eexit 1;\noption presolve 0;\noption substout 0;\n' +
                      contracts.load_commands(original, data_files, directory, evaluation=True))
            prefix += ('''printf {v in _VARS}: 'display %s > "%s";\\n', v, ''' +
                       contracts.quote(warmup) + ' > ' + contracts.quote(instantiate) + ';\n' +
                       'close ' + contracts.quote(instantiate) + ';\n' +
                       'include ' + contracts.quote(instantiate) + ';\n')

            def run(commands):
                remaining = args.verify_timeout - (time.monotonic() - started)
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(args.ampl_cmd, args.verify_timeout)
                script.write_text(prefix + commands + '\nprint "AMPL_VERIFY_OK";\n', encoding='utf-8')
                proc = run_tool([args.ampl_cmd, str(script)], capture_output=True,
                    text=True, timeout=remaining, cwd=str(Path(original).resolve().parent))
                if proc.returncode or 'AMPL_VERIFY_OK' not in proc.stdout.splitlines():
                    raise ValueError('AMPL solution evaluation failed: ' + (proc.stderr or proc.stdout)[-1500:])

            stage = 'coverage'
            variables = directory / 'variables.tsv'
            variables.touch()
            run('printf {i in 1.._nvars}: "%d\\t%s\\t%d\\t%d\\t%.17g\\t%.17g\\n", i, _varname[i], '
                '_var[i].defeqn, _var[i].int, _var[i].lb, _var[i].ub > ' + contracts.quote(variables) + ';')
            rows = []
            for line in variables.read_text(encoding='utf-8').splitlines():
                index, name, defined, integer, lower, upper = line.split('\t')
                rows.append(dict(index=int(index), name=name, defined=int(defined) != 0,
                                 integer=int(integer) != 0, lower=float(lower), upper=float(upper)))
            required = {row['name'] for row in rows if not row['defined']}
            known = {row['name'] for row in rows}
            missing, unknown = required - assignments.keys(), assignments.keys() - known
            result.update(map_required_variables=sorted(required), map_assigned_variables=sorted(assignments),
                          map_missing_variables=sorted(missing), map_unknown_variables=sorted(unknown),
                          map_skipped=sum(row['defined'] and row['name'] in assignments for row in rows))
            if missing or unknown:
                raise ValueError(f'Mapped variable coverage error: missing {sorted(missing)}, non-original variables {sorted(unknown)}')
            stage = 'evaluation'
            values = directory / 'values.tsv'
            constraints = directory / 'constraints.tsv'
            objective = directory / 'objectives.tsv'
            for path in (values, constraints, objective):
                path.touch()
            commands = ''
            for row in rows:
                if not row['defined']:
                    # Initialize the native value explicitly, then release it and
                    # reassign: fix alone projects out-of-bound solutions; let alone
                    # can retain a nonzero := initializer when assigning zero through
                    # defined expressions. Check the resulting values below as well.
                    name, value = row['name'], assignments[row['name']]
                    commands += f"fix {name} := {value:.17g};\nunfix {name};\nlet {name} := {value:.17g};\n"
            commands += ('printf {i in 1.._nvars}: "%s\\t%.17g\\n", '
                         '_varname[i], _var[i] > ' + contracts.quote(values) + ';\n')
            commands += ('printf {i in 1.._ncons}: "%.17g\\t%.17g\\t%.17g\\n", '
                         '_con[i].lb, _con[i].body, _con[i].ub > ' + contracts.quote(constraints) + ';\n')
            commands += ('printf {i in 1.._nobjs}: "%.17g\\n", _obj[i] > ' + contracts.quote(objective) + ';\n')
            run(commands)
            actual = {}
            bound_violations, integer_violations = [], []
            for line in values.read_text(encoding='utf-8').splitlines():
                name, value = line.split('\t')
                actual[name] = float(value)
            if set(actual) != known:
                raise ValueError('Original variable inventory changed during evaluation')
            for row in rows:
                value = actual[row['name']]
                lower, upper = row['lower'], row['upper']
                if not math.isfinite(value) or math.isnan(lower) or math.isnan(upper):
                    raise ValueError('Invalid original variable value or bounds: ' + row['name'])
                bound_violations.append(max(0, lower-value, value-upper))
                if not row['defined'] and value != assignments[row['name']]:
                    raise ValueError('AMPL did not retain the mapped value: ' + row['name'])
                integer_violations.append(bool(row['integer'] and not value.is_integer()))
            constraint_violations = []
            for line in constraints.read_text(encoding='utf-8').splitlines():
                lower, body, upper = map(float, line.split('\t'))
                if not math.isfinite(body) or math.isnan(lower) or math.isnan(upper):
                    raise ValueError('Invalid original constraint value')
                constraint_violations.append(max(0, lower-body, body-upper))
            n_bounds = sum(v > tolerance for v in bound_violations)
            n_constraints = sum(v > tolerance for v in constraint_violations)
            n_integer = sum(integer_violations)
            result.update(map_status='infeasible' if n_bounds or n_constraints or n_integer else 'feasible',
                          map_n_bound_viol=n_bounds, map_n_integer_viol=n_integer,
                          map_n_constraint_viol=n_constraints, map_n_viol=n_bounds+n_constraints+n_integer,
                          map_max_viol=max([0, *bound_violations, *constraint_violations]),
                          map_checks_complete=True)
            objectives = [float(line) for line in objective.read_text(encoding='utf-8').splitlines()]
            result['map_objective_count'] = len(objectives)
            if not objectives:
                result['map_objective_status'] = 'not_applicable'
            elif len(objectives) == 1 and math.isfinite(objectives[0]):
                result['map_objective'] = objectives[0]
            else:
                result['map_objective_error'] = 'Multiple or invalid original objectives are not supported'
    except subprocess.TimeoutExpired:
        result.update(map_status='unknown', map_error='Original-model verification timed out', map_error_stage=stage)
    except Exception as error:
        result.update(map_status='error', map_error=str(error)[:2000], map_error_stage=stage)
    result['verify_time_sec'] = round(time.monotonic()-started, 3)
    return result


def compare_objective(record, solution):
    """Compare original and solution objectives using AMPL's relative tolerance."""
    record['solution_objective'] = solution
    if record.get('map_objective_status') == 'not_applicable' or record.get('map_status') != 'feasible':
        return
    original = record.get('map_objective')
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in (original, solution)):
        record.update(map_objective_status='undetermined',
                      map_objective_error=record.get('map_objective_error') or 'Missing or invalid solution objective')
        return
    difference = abs(original-solution) / max(abs(solution), 1e-10)
    matched = difference <= OBJECTIVE_REL_TOL
    record.update(map_vs_llm_rel=difference, map_objective_match=matched,
                  map_objective_status='matched' if matched else 'mismatch')
