"""AMPL backend: generation support and independent solution verification."""
import os

import solution_protocol as protocol
from backends import check_record, solver_environment
from backends import resolve_tool, ToolError, MappingProbeUnavailable
from . import check, contracts, solution, verify

mod_verify = verify
language = 'ampl'
model_suffixes = ('.mod',)
data_suffixes = ('.dat',)
from prompt_loader import ampl as prompts


def sources(path):
    return contracts.sources(path)


def configure(args, purpose):
    args.ampl_cmd = resolve_tool(os.environ.get('LLM4MOR_AMPL') or 'ampl', 'ampl',
        required=purpose == 'generation' and args.fix_rounds > 0)


def prepare_input(args):
    return (contracts.read_text(args.model_path), contracts.build_contract_context(args.model_path),
            contracts.requires_data(args.model_path))


def original_checks(args):
    if args.has_external_data:
        return check_record(args.instance_id, 'syntax_only',
                            'External data is checked only by verify_solution.py')
    with solver_environment():
        return check.check_original(args.model_path, args.instance_id, args.ampl_cmd)


def skipped_original(args):
    return check_record(args.instance_id, 'skipped', 'fix_rounds=0')


def check_model(model, args):
    if args.has_external_data:
        return check_record(args.instance_id, 'syntax_only',
                            'External data is checked only by verify_solution.py')
    with solver_environment():
        try:
            return check.ampl_check(model, args.ampl_cmd, os.path.dirname(args.model_path),
                                    instance_id=args.instance_id)
        except ToolError as failure:
            return check_record(args.instance_id, 'failed', failure=failure)
        except OSError as error:
            return check_record(args.instance_id, 'failed', failure=error)
        except ValueError as error:
            return check_record(args.instance_id, 'failed', str(error))


def mapping_input(model, args):
    if args.has_external_data:
        return MappingProbeUnavailable('External data is checked only by verify_solution.py')
    with solver_environment():
        return solution.mapping_input(model, args.model_path, args.instance_id,
                                      args.ampl_cmd, check.ampl_dump_vars)


def check_mapping_run(code, payload):
    return solution.run_map(code, payload)


def run_map_script(map_py, sol, args):
    return protocol.run_map(map_py, sol, args.map_timeout, solution.validate_values)


def ampl_verify(orig_src, assignments, args, data_files=()):
    """Evaluate a mapped solution with the native AMPL verifier."""
    return mod_verify.verify(orig_src, assignments, args, data_files)


def reconcile(args):
    args.ampl_cmd = resolve_tool(os.environ.get('LLM4MOR_AMPL') or 'ampl', 'ampl')


def prompt_options(args):
    return {}
