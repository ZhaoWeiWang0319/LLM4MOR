"""Generation-stage AMPL mapping inputs and script execution."""
import solution_protocol as protocol
import math
from pathlib import Path
from backends import InputPreparationError, MappingProbeInput, MappingProbeUnavailable

from . import contracts as mod_contracts


def mapping_input(model_text, model_path, instance_id, ampl_cmd, dump_vars):
    """Prepare an artificial numeric solution using the actual instance shape."""
    names = dump_vars(model_text, ampl_cmd, source_dir=str(Path(model_path).parent))
    if names is None:
        raise InputPreparationError('AMPL solution variable export failed')
    context = mod_contracts.instance_context(model_path, [], ampl_cmd)
    if context['context_omissions']:
        return MappingProbeUnavailable('Original set context cannot be fully represented: '
                                       + ', '.join(context['context_omissions']))
    return MappingProbeInput(protocol.make_input({name: 0.0 for name in names}, 0.0,
        instance_id=instance_id, **context))


def validate_values(mapped):
    if not isinstance(mapped, dict) or any(
            not isinstance(name, str) or type(value) not in (int, float)
            or not math.isfinite(value) for name, value in mapped.items()):
        raise ValueError('AMPL mapping must contain variable names and finite numeric values')


def run_map(code, payload, timeout=120):
    return protocol.check_map(code, payload, timeout, validate_values)
