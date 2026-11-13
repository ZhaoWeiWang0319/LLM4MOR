"""Shared solution, mapping, and verification protocols."""
import json
import math
from pathlib import Path
import sys
import tempfile

import backends


REQUIRED_FIELDS = {'instance_id', 'variables', 'objective', 'parameters'}
OPTIONAL_FIELDS = {'sets', 'indices', 'context_omissions'}
INPUT_FIELDS = REQUIRED_FIELDS | OPTIONAL_FIELDS


def solution_payload(variables, objective, validate_variables):
    """Validate the common saved-solution envelope before publication."""
    if not isinstance(variables, dict):
        raise ValueError('Solution variables must be a JSON object')
    if objective is not None and (type(objective) not in (int, float) or not math.isfinite(objective)):
        raise ValueError('Solution objective must be a finite number or null')
    validate_variables(variables)
    solution = dict(variables=variables, objective=objective)
    json.dumps(solution, allow_nan=False)
    return solution


def validate_input(payload):
    if not isinstance(payload, dict):
        raise ValueError('Mapping input must be a JSON object')
    missing, unknown = REQUIRED_FIELDS - payload.keys(), payload.keys() - INPUT_FIELDS
    if missing or unknown:
        raise ValueError(f'Invalid mapping input fields: missing={sorted(missing)}, unknown={sorted(unknown)}')
    if not isinstance(payload['instance_id'], str) or not payload['instance_id']:
        raise ValueError('Mapping input requires a nonempty instance_id')
    objective = payload['objective']
    if objective is not None and (type(objective) not in (int, float) or not math.isfinite(objective)):
        raise ValueError('Mapping objective must be a finite number or null')
    for name in ('variables', 'parameters', 'sets', 'indices'):
        if name in payload and (not isinstance(payload[name], dict)
                                or any(not isinstance(key, str) for key in payload[name])):
            raise ValueError('Mapping ' + name + ' must be an object with string keys')
    omissions = payload.get('context_omissions', [])
    if not isinstance(omissions, list) or any(not isinstance(name, str) for name in omissions):
        raise ValueError('Mapping context_omissions must be a list of names')
    json.dumps(payload, allow_nan=False)


def make_input(variables, objective, *, instance_id, parameters,
               sets=None, indices=None, context_omissions=None):
    payload = dict(instance_id=instance_id, variables=variables,
                   objective=objective, parameters=parameters)
    for name, value in (('sets', sets), ('indices', indices), ('context_omissions', context_omissions)):
        if value:
            payload[name] = value
    validate_input(payload)
    return payload


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate mapping output key: ' + key)
        result[key] = value
    return result


def _execute(script, payload, directory, timeout, validate_variables):
    validate_input(payload)
    source, target = directory / 'input.json', directory / 'output.json'
    source.write_text(json.dumps(payload, allow_nan=False), encoding='utf-8')
    proc = backends.run_tool([sys.executable, str(script), str(source), str(target)],
                          native=False, before_attempt=lambda: target.unlink(missing_ok=True),
                          cwd=directory, capture_output=True, text=True, timeout=timeout)
    if proc.returncode:
        raise backends.MappingExecutionError('map.py execution failed: ' + (proc.stderr or proc.stdout)[-1500:])
    try:
        output = target.read_text(encoding='utf-8')
    except FileNotFoundError as error:
        raise backends.MappingExecutionError('map.py did not produce output.json') from error
    except UnicodeError as error:
        raise backends.MappingExecutionError('Mapping output must be UTF-8') from error
    try:
        result = json.loads(output, object_pairs_hook=_object)
        if not isinstance(result, dict) or set(result) != {'variables'} or not isinstance(result['variables'], dict):
            raise ValueError('Mapping output must contain only a variables object')
        json.dumps(result, allow_nan=False)
        validate_variables(result['variables'])
    except ValueError as error:
        raise backends.MappingExecutionError('Invalid mapping output: ' + str(error)) from error
    return result['variables']


def run_map(path, payload, timeout, validate_variables):
    script = Path(path).resolve()
    with tempfile.TemporaryDirectory(prefix='mapping_run_') as directory:
        return _execute(script, payload, Path(directory), timeout, validate_variables)


def check_map(code, payload, timeout, validate_variables):
    with tempfile.TemporaryDirectory(prefix='mapping_check_') as directory:
        directory = Path(directory)
        script = directory / 'map.py'
        script.write_text(code, encoding='utf-8')
        return _execute(script, payload, directory, timeout, validate_variables)


def number(value):
    return value if type(value) in (int, float) and math.isfinite(value) else None


def verification_result(native, *, model_id, instance_id, language,
                        solution_objective, seconds, feasibility_tolerance):
    checks = {}
    native_status = native.get('map_status')
    message = native.get('map_error') or native.get('error')
    stage = native.get('map_error_stage')
    if native.get('_mapping_passed'):
        checks['mapping'] = dict(status='passed')
    elif stage == 'map':
        checks['mapping'] = dict(status='failed' if native_status == 'error' else 'unknown',
                                 message=message or 'Mapping did not complete')
    complete = native.get('map_checks_complete') is True
    if native_status in ('feasible', 'infeasible'):
        checks['feasibility'] = dict(status=(
            'passed' if native_status == 'feasible' else 'failed') if complete else 'unknown')
        if message:
            checks['feasibility']['message'] = message
    elif native.get('_mapping_passed'):
        checks['feasibility'] = dict(status='unknown', message=message or 'Original-model checking did not complete')
    if complete and native_status == 'feasible':
        objective_status = native.get('map_objective_status')
        objective_status = ('failed' if native.get('map_objective_match') is False else
            {'matched': 'passed', 'mismatch': 'failed', 'not_applicable': 'not_applicable'}.get(objective_status, 'unknown'))
        checks['objective'] = dict(status=objective_status)
        if native.get('map_objective_error'):
            checks['objective']['message'] = native['map_objective_error']
        elif objective_status == 'unknown':
            checks['objective']['message'] = 'Objective comparison did not complete'
        if (number(native.get('map_max_viol')) is not None
                and native['map_max_viol'] > feasibility_tolerance):
            checks['feasibility'] = dict(status='failed', message='Maximum violation exceeds feasibility tolerance')
    solution, mapped = number(solution_objective), number(native.get('map_objective'))
    if checks.get('objective', {}).get('status') == 'passed':
        if solution is None or mapped is None:
            checks['objective'] = dict(status='unknown', message='Solution or mapped objective is unavailable')
        elif abs(mapped-solution)/max(abs(solution), 1e-10) > 1e-4:
            checks['objective'] = dict(status='failed', message='Mapped and solution objectives differ')
    result = dict(model_id=model_id, instance_id=instance_id, language=language,
        status='no_solution' if native_status == 'no_solution' else 'error' if native_status == 'error' else 'unknown',
        checks=checks, solution_objective=solution, mapped_objective=mapped,
        objective_relative_difference=(abs(mapped-solution)/max(abs(solution), 1e-10)
                                       if mapped is not None and solution is not None else None),
        max_violation=number(native.get('map_max_viol')),
        feasibility_tolerance=feasibility_tolerance, seconds=seconds)
    if native_status not in ('no_solution', 'error'):
        if checks.get('feasibility', {}).get('status') == 'failed' or checks.get('objective', {}).get('status') == 'failed':
            result['status'] = 'failed'
        elif (checks.get('mapping', {}).get('status') == 'passed'
              and checks.get('feasibility', {}).get('status') == 'passed'
              and checks.get('objective', {}).get('status') in ('passed', 'not_applicable')):
            result['status'] = 'passed'
    if stage:
        result['stage'] = stage
    if message:
        result['message'] = message
    elif result['status'] == 'failed':
        result['message'] = ('Mapped assignment violates the original model'
            if checks.get('feasibility', {}).get('status') == 'failed' else 'Mapped and solution objectives differ')
    return result
