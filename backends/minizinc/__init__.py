"""MiniZinc backend: generation support and independent solution verification."""
import os
from pathlib import Path

from backends import check_record, solver_environment
from backends import resolve_tool, read_text, ToolConfigurationError, InputPreparationError, MappingProbeUnavailable
from . import check, contracts, solution

language = 'minizinc'
model_suffixes = ('.mzn',)
data_suffixes = ('.dzn',)
from prompt_loader import minizinc as prompts


def sources(path):
    found, visited = [], set()
    def visit(path):
        path = Path(path).resolve()
        if path in visited:
            return
        visited.add(path)
        found.append(str(path))
        for name in contracts.inspect_model(read_text(path))['includes']:
            child = path.parent / name
            if child.is_file():
                visit(child)
    visit(path)
    return found


def configure(args, purpose):
    args.minizinc_cmd = resolve_tool(os.environ.get('LLM4MOR_MINIZINC') or 'minizinc', 'minizinc',
        required=purpose == 'generation' and args.fix_rounds > 0)
    if purpose == 'generation':
        args.mzn_check_solver = getattr(args, 'mzn_check_solver', None) or 'highs'
        if args.fix_rounds > 0:
            with solver_environment(args.mzn_check_solver):
                result = check.mzn_check('solve satisfy;', args.minizinc_cmd, args.mzn_check_solver,
                                         instance_id='configuration')
            if result['status'] != 'passed':
                if result.get('_failure') is not None:
                    raise result['_failure']
                raise ToolConfigurationError('MiniZinc check backend is unavailable: ' + result['message'])


def prepare_input(args):
    try:
        source = read_text(args.model_path)
    except UnicodeDecodeError as error:
        raise InputPreparationError('MiniZinc input encoding unsupported: model must be UTF-8; explicitly normalize a copy') from error
    requires_data = any(contracts.requires_data(path) for path in sources(args.model_path))
    return source, contracts.build_contract_context(args.model_path), requires_data


def original_checks(args):
    return check_model(read_text(args.model_path), args)


def skipped_original(args):
    return check_record(args.instance_id, 'skipped', 'fix_rounds=0')


def check_model(model, args):
    if args.has_external_data:
        return check_record(args.instance_id, 'syntax_only',
                            'External data is checked only by verify_solution.py')
    with solver_environment(args.mzn_check_solver):
        return check.mzn_check(model, args.minizinc_cmd, args.mzn_check_solver,
            [os.path.dirname(p) for p in sources(args.model_path)], instance_id=args.instance_id)


def mapping_input(model, args):
    if args.has_external_data:
        return MappingProbeUnavailable('External data is checked only by verify_solution.py')
    with solver_environment(args.mzn_check_solver):
        return solution.mapping_input(model, args.model_path, sources(args.model_path), args.instance_id,
                                      args.minizinc_cmd, args.mzn_check_solver)


def check_mapping_run(code, payload):
    return solution.check_map_run(code, payload)


def add_arguments(parser, command):
    if command == 'generation':
        parser.add_argument('--mzn-check-solver', default=None, choices=['highs', 'gurobi'],
            help="MiniZinc compilation backend; defaults to highs")


def reconcile(args):
    args.mzn_check_solver = args.mzn_check_solver or 'highs'
    args.minizinc_cmd = resolve_tool(os.environ.get('LLM4MOR_MINIZINC') or 'minizinc', 'minizinc')


def prompt_options(args):
    return dict(solver=args.mzn_check_solver)
