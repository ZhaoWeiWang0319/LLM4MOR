#!/usr/bin/env python3
"""Verify a reformulated solution against its original model without re-solving."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import backends
from backends import write_text, resolve_data_file, stem
from backends import verification
import solution_protocol as protocol
from backends.ampl import solution as ampl_solution
from backends.ampl import verify as ampl_verify
from backends.minizinc import solution as mzn_solution
from backends.minizinc import verify as mzn_verify


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('model', help='Original AMPL or MiniZinc model file')
    parser.add_argument('--sol', dest='solution', required=True,
                        help='Solution .sol.json file with variables and objective')
    parser.add_argument('--map', required=True, dest='map_file',
                        help='Python mapping script generated for this model')
    parser.add_argument('--data', default=None, metavar='PATH',
                        help='One original data file; omit for a self-contained model')
    parser.add_argument('--instance-id', default=None,
                        help='Original instance identifier (default: model filename stem)')
    parser.add_argument('--backend', default='highs', choices=('highs', 'gurobi'),
                        help='MiniZinc compilation backend only; no optimization solve is allowed')
    parser.add_argument('--map-timeout', type=int, default=300,
                        help='Mapping script timeout in seconds')
    parser.add_argument('--verify-timeout', type=int, default=900,
                        help='Original-model verification timeout in seconds')
    parser.add_argument('--verify-feas-tol', type=float, default=verification.DEFAULT_FEAS_TOL,
                        help='Absolute numerical feasibility tolerance')
    parser.add_argument('--output', default=None,
                        help='Write the verification record to this JSON file instead of stdout')
    return parser


def load_solution(path, language):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(value, dict) or set(value) != {'variables', 'objective'}:
        raise ValueError('Solution must contain exactly variables and objective')
    validate = ampl_solution.validate_values if language == 'ampl' else mzn_solution.validate_values
    return protocol.solution_payload(value['variables'], value['objective'], validate)


def run_verification(args):
    model = Path(args.model).expanduser().resolve()
    if not model.is_file():
        raise ValueError('Original model file does not exist: ' + str(model))
    backend = backends.for_path(model)
    args.instance_id = args.instance_id or stem(model.name)
    args.solver = args.backend
    args.mzn_check_solver = args.backend
    args.allow_residual_solve = False
    backend.reconcile(args)
    data_files = resolve_data_file(args.data, backend.data_suffixes)
    solution = load_solution(args.solution, backend.language)
    map_file = Path(args.map_file).expanduser().resolve()
    if not map_file.is_file():
        raise ValueError('Mapping script does not exist: ' + str(map_file))
    started = time.monotonic()
    if backend.language == 'ampl':
        context = ampl_verify.context(str(model), data_files, args)
        payload = protocol.make_input(solution['variables'], solution['objective'],
            instance_id=args.instance_id, **context)
        assignments = backend.run_map_script(str(map_file), payload, args)
        native = backend.ampl_verify(str(model), assignments, args, data_files=data_files)
        ampl_verify.compare_objective(native, solution['objective'])
        native['_mapping_passed'] = True
    else:
        include_dirs = list(dict.fromkeys(os.path.dirname(path) for path in backend.sources(model)))
        solution = dict(solution, instance_id=args.instance_id)
        native = mzn_verify.verify(str(model), data_files, include_dirs,
                                   str(map_file), solution, args)
    return protocol.verification_result(native, model_id=stem(model.name),
        instance_id=args.instance_id, language=backend.language,
        solution_objective=solution['objective'], seconds=round(time.monotonic()-started, 3),
        feasibility_tolerance=verification.feasibility_tolerance(args))


def main():
    args = build_parser().parse_args()
    if args.map_timeout <= 0 or args.verify_timeout <= 0:
        build_parser().error('--map-timeout and --verify-timeout must be positive')
    try:
        record = run_verification(args)
    except Exception as error:
        print('# Verification failed: ' + str(error), file=sys.stderr)
        return 1
    text = json.dumps(record, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        write_text(args.output, text)
    else:
        print(text, end='')
    return 0 if record['status'] == 'passed' else 1


if __name__ == '__main__':
    sys.exit(main())
