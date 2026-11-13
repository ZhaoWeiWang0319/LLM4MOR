"""Native parameter-only checkers with a conservative fixed-model fallback.

Python transforms source boundaries, never evaluates MiniZinc expressions. A direct
checker has no decision variables or constraints: all checks are parameter booleans
evaluated through an OZN. MiniZinc decides which expressions are computable.
"""
import solution_protocol as protocol
import json
import math
import re
import subprocess
import time
from pathlib import Path
import tempfile

from . import solution as native
from .contracts import read_text, tokens

PREFIX = '_llm4_v_'
OBJECTIVE_REL_TOL = 1e-4


def objective_matches(value, reference):
    """Use the same relative objective comparison and zero floor as AMPL."""
    return abs(value-reference) / max(abs(reference), 1e-10) <= OBJECTIVE_REL_TOL


class DirectUnsupported(ValueError):
    pass


def context(original, data_files, include_dirs, args, remaining):
    """Prepare the original model's parameter context for one verification."""
    info = native.model_info(native.inline_source(original, include_dirs),
                             [read_text(p) for p in data_files])
    with tempfile.TemporaryDirectory(prefix='mzn_verify_context_') as directory:
        command = [args.minizinc_cmd, '--solver', args.solver]
        for parent in dict.fromkeys([str(Path(original).resolve().parent), *include_dirs]):
            command += ['-I', str(parent)]
        parameters, indices = native.instance_context(info, directory,
            native.stage_data(directory, data_files), command, remaining)
    return dict(info=info, parameters=parameters, indices=indices)


def validate_assignment(info, indices, mapped):
    rows = info['rows']
    unknown = set(mapped) - {n for n, row in rows.items() if row['decision']}
    missing = set(info['required']) - set(mapped)
    if unknown or missing:
        raise ValueError(f'Mapped variable coverage error: missing {sorted(missing)}, non-original variables {sorted(unknown)}')
    for name, value in mapped.items():
        dim = rows[name]['dim']
        if dim:
            if not isinstance(value, dict) or set(value) != {'array', 'index_sets'} or len(value['index_sets']) != dim:
                raise ValueError('Mapped arrays must provide the correct dimensions and actual index domains: ' + name)
            if name not in indices or not all(native.same_index(a, b) for a, b in zip(value['index_sets'], indices[name])):
                raise ValueError('Mapped array index domains do not match the original model: ' + name)
        native.literal(value)  # Validates finite values, array lengths, and ground types.


# Common float comparisons use absolute tolerance; discrete checks stay exact.
_number = re.compile(r'(?:\d+\.(?!\.)\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?\Z')
_lex = re.compile(r'''\s+|%[^\r\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|(?:\d+\.(?!\.)\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?|[A-Za-z_]\w*|'[^'\\]+'|<=|>=|==|!=|/\\|\\/|\.\.|->|.''', re.S)


def _top(text):
    """Source tokens at depth zero, coalescing multi-character operators."""
    ts, out, depth, i = list(tokens(text)), [], 0, 0
    while i < len(ts):
        value, start, end = ts[i]
        if value in ('(', '[', '{'):
            depth += 1
        elif value in (')', ']', '}'):
            depth -= 1
        elif depth == 0:
            if i+1 < len(ts) and end == ts[i+1][1] and value+ts[i+1][0] in ('<=', '>=', '==', '!=', '/\\', '\\/', '->', '..'):
                value += ts[i+1][0]
                end = ts[i+1][2]
                i += 1
            out.append((value, start, end))
        i += 1
    return out


def _unwrap(text):
    text = text.strip()
    while text.startswith('(') and text.endswith(')'):
        ts, depth, closes = list(tokens(text)), 0, None
        for value, _, end in ts:
            depth += (value == '(') - (value == ')')
            if depth == 0:
                closes = end
                break
        if closes != len(text):
            break
        text = text[1:-1].strip()
    return text


def _parts(text, separator):
    cuts = [(a, b) for value, a, b in _top(text) if value == separator]
    start, out = 0, []
    for a, b in cuts:
        out.append(text[start:a])
        start = b
    return [*out, text[start:]]


def _generator_types(generator, types):
    # Local generator names shadow model declarations. Do not guess the type
    # of their index domain (which may be an enum rather than an integer).
    local = dict(types)
    pending = []
    for part in _parts(generator, ','):
        separator = next(((a, b) for op, a, b in _top(part) if op in ('in', '=')), None)
        before = part[:separator[0]] if separator else part
        pending.extend(t[0] for t in tokens(before))
        if separator:
            for name in pending:
                operator = part[separator[0]:separator[1]]
                local[name] = ('index' if operator == 'in' else 'unknown', 0)
            pending = []
    return local


class _Numeric:
    """Small type recogniser, not an evaluator; unsupported syntax stays strict."""
    def __init__(self, text, types):
        self.text = text
        matches = [m for m in _lex.finditer(text)
                   if not m[0].isspace() and not m[0].startswith(('%', '/*'))]
        self.ts = [m[0] for m in matches]
        self.spans = [m.span() for m in matches]
        self.pos, self.types = 0, types

    def take(self, expected=None):
        value = self.ts[self.pos] if self.pos < len(self.ts) else ''
        if expected is not None and value != expected:
            raise ValueError('unsupported numeric syntax')
        self.pos += 1
        return value

    def group(self, opening, closing):
        self.take(opening)
        start, stack = self.spans[self.pos-1][1], [closing]
        while stack:
            value = self.take()
            if not value:
                raise ValueError('unclosed group')
            if value in ('(', '[', '{'):
                stack.append({'(': ')', '[': ']', '{': '}'}[value])
            elif value in (')', ']', '}'):
                if value != stack.pop():
                    raise ValueError('unbalanced group')
        return self.text[start:self.spans[self.pos-1][0]]

    def atom(self):
        value = self.ts[self.pos] if self.pos < len(self.ts) else ''
        if value in ('+', '-'):
            self.take()
            return self.atom()
        if value == '(':
            return numeric_kind(self.group('(', ')'), self.types)
        if _number.fullmatch(value):
            self.take()
            return ('float' if any(c in value for c in '.eE') else 'int', 0)
        if value == '[':
            body = self.group('[', ']')
            comprehension = _parts(body, '|')
            body = comprehension[0]
            types = _generator_types(comprehension[1], self.types) if len(comprehension) == 2 else self.types
            kinds = [numeric_kind(part, types) for part in _parts(body, ',')]
            if kinds and all(k and k[1] == 0 for k in kinds):
                return ('float' if any(k[0] == 'float' for k in kinds) else 'int', 1)
            return None
        self.take()
        if self.pos < len(self.ts) and self.ts[self.pos] == '(':
            if value not in ('sum', 'min', 'max', 'abs', 'int2float', 'sqrt', 'exp', 'ln', 'log2', 'log10', 'floor', 'ceil', 'round'):
                return None
            argument = self.group('(', ')')
            argument_types = self.types
            generator_form = False
            if self.pos < len(self.ts) and self.ts[self.pos] == '(':
                # Generator forms such as sum(i in I)(x[i]); generator names
                # are not guessed to be integers (the index set may be an enum).
                if value not in ('sum', 'min', 'max'):
                    return None
                argument_types = _generator_types(argument, self.types)
                argument = self.group('(', ')')
                generator_form = True
            kinds = [numeric_kind(part, argument_types) for part in _parts(argument, ',')]
            if not kinds or any(k is None for k in kinds):
                return None
            if value in ('sum', 'min', 'max'):
                if not ((len(kinds) == 1 and (kinds[0][1] > 0 or generator_form)) or
                        (value != 'sum' and len(kinds) == 2 and all(k[1] == 0 for k in kinds))):
                    return None
            elif len(kinds) != 1 or kinds[0][1] != 0:
                return None
            kind = 'float' if any(k[0] == 'float' for k in kinds) else 'int'
            if value in ('int2float', 'sqrt', 'exp', 'ln', 'log2', 'log10'):
                kind = 'float'
            elif value in ('floor', 'ceil', 'round'):
                kind = 'int'
            return kind, 0
        kind = self.types.get(value)
        if self.pos < len(self.ts) and self.ts[self.pos] == '[':
            indices = _parts(self.group('[', ']'), ',')
            if not kind or kind[1] != len(indices):
                return None
            index_types = {name: ('int', 0) if entry == ('index', 0) else entry
                           for name, entry in self.types.items()}
            for index in indices:
                index = _unwrap(index)
                index_kind = numeric_kind(index, index_types)
                if index_kind != ('int', 0) and not (native._identifier.fullmatch(index) and index not in self.types):
                    return None  # Set/range slices and unknown functions may return arrays.
            return kind[0], 0
        return kind

    def expression(self):
        kind = self.atom()
        while self.pos < len(self.ts) and self.ts[self.pos] in ('+', '-', '*', '/', 'div', 'mod', '^'):
            operator = self.take()
            right = self.atom()
            if not kind or not right or kind[1] or right[1] or kind[0] not in ('int', 'float') or right[0] not in ('int', 'float'):
                kind = None
            else:
                kind = ('float' if operator == '/' or 'float' in (kind[0], right[0]) else 'int', 0)
        return kind


def numeric_kind(text, types):
    parser = _Numeric(text, types)
    try:
        kind = parser.expression()
        return kind if kind and kind[0] in ('int', 'float') and parser.pos == len(parser.ts) else None
    except (ValueError, IndexError):
        return None


def _declarations(info):
    for item in native.items(info['body']):
        ts = list(tokens(item))
        if not ts or ts[0][0] in ('constraint', 'include', 'function', 'predicate', 'test', 'annotation', 'ann', 'enum') or (len(ts) > 2 and ts[1][0] == '='):
            continue
        _, name = native.declaration_header(ts)
        depth = 0
        for colon, (value, _, _) in enumerate(ts):
            if value == ':' and depth == 0:
                break
            depth += (value in ('[', '(')) - (value in (']', ')'))
        header = ts[:colon]
        var = next((i for i, t in enumerate(header) if t[0] == 'var'), None)
        if var is not None:
            start = var+1
        elif header[0][0] in ('array', 'list'):
            start = next(i for i, t in enumerate(header) if t[0] == 'of')+1
        else:
            start = 1 if header[0][0] == 'par' else 0
        core = item[ts[start][1]:ts[colon][1]].strip()
        yield item, ts, colon, name, core, start


def numeric_types(info):
    declarations = list(_declarations(info))
    types = {}
    # Domains may refer to parameters declared later in the source.
    for _ in range(len(declarations)+1):
        changed = False
        for _, _, _, name, core, _ in declarations:
            if name in types:
                continue
            kind = (core, 0) if core in ('int', 'float') else None
            bounds = _parts(_unwrap(core), '..')
            if len(bounds) == 2:
                ends = [numeric_kind(bound, types) for bound in bounds]
                if all(end and end[1] == 0 for end in ends):
                    kind = ('float' if any(end[0] == 'float' for end in ends) else 'int', 0)
            if kind:
                types[name] = (kind[0], info['rows'][name]['dim'])
                changed = True
        if not changed:
            break
    for _, _, _, name, core, _ in declarations:
        if name not in types:
            types[name] = ('other', info['rows'][name]['dim'])
    return types




def relax_numeric_check(expression, types, tol):
    """Return (boolean check, nonnegative violation or None)."""
    text = _unwrap(expression)
    top = _top(text)
    # Relaxing beneath negation, implication or disjunction can strengthen or
    # otherwise change the meaning of the enclosing formula. Leave it intact.
    if any(t in ('\\/', '->', 'if', 'let') for t, _, _ in top):
        return expression, None
    conjunction = _parts(text, '/\\')
    if len(conjunction) > 1:
        checks = [relax_numeric_check(part, types, tol) for part in conjunction]
        violations = [v for _, v in checks if v is not None]
        return ' /\\ '.join('(' + c + ')' for c, _ in checks), native.max_violation_expression(violations) if violations else None
    if any(t in ('<', '>', 'not') for t, _, _ in top):
        return expression, None
    if text.startswith('forall'):
        parser = _Numeric(text[len('forall'):], types)
        try:
            generator = parser.group('(', ')')
            body = parser.group('(', ')')
            if parser.pos != len(parser.ts):
                return expression, None
            check, violation = relax_numeric_check(body, _generator_types(generator, types), tol)
            if violation is not None:
                return f'forall({generator})({check})', f'max([0.0] ++ [({violation}) | {generator}])'
        except (ValueError, IndexError):
            pass
        return expression, None
    comparisons = [(op, a, b) for op, a, b in top if op in ('=', '==', '<=', '>=', '!=')]
    if len(comparisons) != 1 or comparisons[0][0] == '!=':
        return expression, None
    operator, a, b = comparisons[0]
    left, right = text[:a].strip(), text[b:].strip()
    kinds = [numeric_kind(side, types) for side in (left, right)]
    if any(not k or k[1] for k in kinds) or all(k[0] == 'int' for k in kinds):
        return expression, None
    difference = f'(({left}) - ({right}))'
    violation = (f'abs({difference})' if operator in ('=', '==') else
                 f'max(0.0, {difference})' if operator == '<=' else f'max(0.0, -{difference})')
    check = f'({violation}) <= {tol!r}' if tol else expression
    return check, violation


def prepare_tolerant_model(info, mapped, tol):
    """Use the same transformed model, domains and bindings in both check paths."""
    types = numeric_types(info)
    domains = dict(native.decision_domains(info))
    reformulated, violations, widened = [], [], {}
    identifiers = {t[0] for t in tokens(info['body'])}
    declarations = {name: (item, ts, colon, core, start)
                    for item, ts, colon, name, core, start in _declarations(info)}
    for name, (item, ts, colon, core, start) in declarations.items():
        bounds = _parts(_unwrap(core), '..')
        if (tol and info['rows'][name]['decision'] and types.get(name, ('',))[0] == 'float'
                and len(bounds) == 2):
            lower, upper = bounds
            value = '_llm4_tol_value' if info['rows'][name]['dim'] else name
            if value in identifiers and value != name:
                raise ValueError('Original model uses a name reserved by the verifier: ' + value)
            kind = dict(types, **{value: ('float', 0)})
            check, violation = relax_numeric_check(f'({value} >= ({lower})) /\\ ({value} <= ({upper}))', kind, tol)
            if info['rows'][name]['dim']:
                check = f'forall({value} in array1d({name}))({check})'
                violation = f'max([0.0] ++ [({violation}) | {value} in array1d({name})])'
            domains['domain_'+name] = check
            violations.append(violation)
            # Preserve the var/par marker, array index sets, definitions and annotations.
            replacement = item[:ts[start][1]] + 'float' + item[ts[colon][1]:]
            widened[item] = replacement
    for item in native.items(info['body']):
        ts = list(tokens(item))
        if ts and ts[0][0] == 'constraint':
            expr = item[ts[0][2]:item.rfind(';')]
            check, violation = relax_numeric_check(expr, types, tol)
            reformulated.append('constraint ' + check + ';')
            if violation is not None:
                violations.append(violation)
        else:
            reformulated.append(widened.get(item, item))
    bindings = {}
    for name, value in mapped.items():
        if name not in info['required']:
            check, violation = relax_numeric_check(f'{name} = ({native.literal(value)})', types, tol)
            bindings[name] = check
            if violation is not None:
                violations.append(violation)
    return dict(info, body='\n'.join(reformulated), _verification_domains=list(domains.items()),
                _verification_bindings=bindings, _verification_violations=violations,
                _verification_tolerance=tol)


def checker_source(info, mapped):
    """Parameterise top-level decisions; let MiniZinc evaluate the original code."""
    declarations, checks = [], native.decision_domains(info)
    enum_names = {ts[1][0] for item in native.items(info['body'])
                  if (ts := list(tokens(item))) and ts[0][0] == 'enum'}
    for item in native.items(info['body']):
        ts = list(tokens(item))
        if not ts:
            continue
        first = ts[0][0]
        if any(t[0].startswith(PREFIX) for t in ts):
            raise DirectUnsupported('Original model uses a name reserved by the checker')
        if first == 'constraint':
            checks.append(('constraint_'+str(len(checks)), item[ts[0][2]:item.rfind(';')].strip()))
            continue
        if first in ('include', 'function', 'predicate', 'test', 'annotation', 'ann', 'enum') or (len(ts) > 2 and ts[1][0] == '='):
            declarations.append(item)
            continue
        _, name = native.declaration_header(ts)
        row = info['rows'][name]
        if not row['decision']:
            declarations.append(item)
            continue
        # Locate the top-level colon and the scalar type following any array header.
        depth = 0
        for colon, (value, _, _) in enumerate(ts):
            if value == ':' and depth == 0:
                break
            depth += (value in ('[', '(', '{')) - (value in (']', ')', '}'))
        var_index = next((i for i, t in enumerate(ts[:colon]) if t[0] == 'var'), None)
        if var_index is None:
            raise DirectUnsupported('Cannot determine decision variable declaration type')
        core = item[ts[var_index][2]:ts[colon][1]].strip()
        if 'opt' in [t[0] for t in ts[:colon]]:
            # Preserve optional types; explicit domain checks also cover arrays.
            plain = core
        elif core in ('int', 'float', 'bool', 'string') or core in enum_names:
            plain = core
        elif core.startswith('set of '):
            domain = core[len('set of '):].strip()
            if domain in enum_names:
                plain = core
            else:
                plain = 'set of int'
        else:
            # Widen known integer solutions for detailed domain reports. Other
            # domains retain their exact type and are checked by MiniZinc itself.
            values = mapped.get(name)
            if row['dim'] and values is not None:
                values = values['array']
            elif name in mapped:
                values = [values]
            if values is not None and values and not all(type(v) is int for v in values):
                plain = core
            else:
                declarations.append(f'set of int: {PREFIX}domain_{len(declarations)} = ({core});')
                plain = 'int'
        if name in mapped and plain in ('int', 'float', 'bool', 'string'):
            values = mapped[name]['array'] if row['dim'] else [mapped[name]]
            for value in values:
                valid = (type(value) is int if plain == 'int' else
                         type(value) in (int, float) if plain == 'float' else
                         type(value) is bool if plain == 'bool' else isinstance(value, str))
                if not valid:
                    raise ValueError('Mapped variable type mismatch: ' + name)
        reformulated = item[:ts[var_index][1]] + plain + item[ts[colon][1]:]
        if name in info['required']:
            reformulated = reformulated[:reformulated.rfind(';')] + ' = ' + native.literal(mapped[name]) + ';'
        elif name in mapped:
            checks.append(('binding_'+name, info.get('_verification_bindings', {}).get(name, f'{name} = ({native.literal(mapped[name])})')))
        declarations.append(reformulated)
    expressions = [expr for _, expr in checks]
    outputs = []
    for i, expr in enumerate(expressions):
        outputs += [json.dumps(f'check_{i} = '), f'show(({expr}))', json.dumps(';\n')]
    if info['objective']:
        outputs += [json.dumps('_objective = '), f'show(({info["objective"]}))', json.dumps(';\n')]
    else:
        outputs += [json.dumps('checked = true;\n')]
    if info.get('_verification_violations'):
        outputs += [json.dumps('_max_violation = '),
                    f'show(({native.max_violation_expression(info["_verification_violations"])}))', json.dumps(';\n')]
    text = '\n'.join(declarations) + '\nsolve satisfy;\noutput [' + ','.join(outputs) + '];'
    return text, [label for label, _ in checks]


def direct_check(info, mapped, data, directory, command, remaining, metrics=None):
    source, labels = checker_source(info, mapped)
    model = Path(directory)/'checker.mzn'
    model.write_text(source, encoding='utf-8')
    try:
        fzn, ozn, status = native.compile_verification(model, data, command, remaining, metrics)
        if status == 'UNSATISFIABLE' or native.residual_problem(fzn) or not native.ground_output(ozn):
            raise DirectUnsupported('Checker contains variables, constraints, or contradictions after compilation; use original-model verification')
        values, _ = native.evaluate_verification(ozn, command, remaining, metrics)
        if values is None:
            raise DirectUnsupported('Checker returned no check results')
    except ValueError as error:
        raise DirectUnsupported('Parameter checker could not be evaluated directly: ' + str(error)[-1000:]) from error
    flags = [values.get(f'check_{i}') for i in range(len(labels))]
    if any(type(flag) is not bool for flag in flags) or (not labels and values.get('checked') is not True and '_objective' not in values):
        raise DirectUnsupported('Checker did not return a complete set of check results')
    failed = [label for label, flag in zip(labels, flags) if not flag]
    return dict(map_status='infeasible' if failed else 'feasible',
                map_objective=values.get('_objective') if not failed else None,
                map_max_viol=values.get('_max_violation'),
                map_n_viol=len(failed), violated_checks=failed[:50], verify_check_count=len(labels))


def verify(original, data_files, include_dirs, map_path, solution, args):
    from backends.verification import CONTEXT_TIMEOUT, feasibility_tolerance

    started = time.monotonic()
    result = dict(map_status='error', map_objective=None, map_objective_match=None,
        map_max_viol=None, map_error='', map_error_stage='', verify_method='none', fallback_reason='',
        map_objective_status='not_checked', map_objective_error='',
        verify_objective_rel_tol=OBJECTIVE_REL_TOL,
        context_time_sec=0, map_time_sec=0, direct_time_sec=0, fallback_time_sec=0,
        verify_compile_sec=0, verify_solve_sec=0, verify_output_sec=0,
        verify_compile_count=0, verify_solver_calls=0)

    def context_remaining():
        seconds = CONTEXT_TIMEOUT-(time.monotonic()-started)
        if seconds <= 0:
            raise subprocess.TimeoutExpired('MiniZinc verification context', CONTEXT_TIMEOUT)
        return seconds

    def timed(field, function):
        before = time.monotonic()
        try:
            return function()
        finally:
            result[field] += time.monotonic()-before

    stage, payload = 'context', None
    try:
        ctx = timed('context_time_sec', lambda: context(original, data_files, include_dirs, args, context_remaining))
        info = ctx['info']
        payload = protocol.make_input(solution['variables'], solution['objective'],
            instance_id=solution['instance_id'], parameters=ctx['parameters'], indices=ctx['indices'])
        stage = 'map'
        mapped = timed('map_time_sec', lambda: native.run_map(map_path, payload, args.map_timeout))
        result['_mapping_passed'] = True
        verification_started = time.monotonic()

        def remaining():
            seconds = args.verify_timeout-(time.monotonic()-verification_started)
            if seconds <= 0:
                raise subprocess.TimeoutExpired('MiniZinc verification', args.verify_timeout)
            return seconds

        stage = 'assignment'
        validate_assignment(info, ctx['indices'], mapped)
        tol = feasibility_tolerance(args)
        result['verify_feas_tol'] = tol
        info = prepare_tolerant_model(info, mapped, tol)
        with tempfile.TemporaryDirectory(prefix='mzn_verify_') as directory:
            data = native.stage_data(directory, data_files)
            command = [args.minizinc_cmd, '--solver', args.solver]
            for parent in dict.fromkeys([str(Path(original).resolve().parent), *include_dirs]):
                command += ['-I', str(parent)]
            mode = getattr(args, 'verify_mode', 'auto')
            direct = None
            if mode != 'fixed_model':
                stage = 'direct_check'
                result['verify_method'] = 'direct'
                try:
                    direct = timed('direct_time_sec', lambda: direct_check(info, mapped, data, directory, command, remaining, result))
                except DirectUnsupported as error:
                    result['fallback_reason'] = str(error)
            if (direct is None or mode == 'crosscheck') and getattr(args, 'allow_residual_solve', True):
                stage = 'fixed_model'
                result['verify_method'] = 'fixed_model' if direct is None else 'direct'
                fixed = timed('fallback_time_sec', lambda: native.fixed_check(info, data, command, directory, mapped, remaining, result))
                if direct is not None:
                    result['verify_crosschecked'] = fixed['map_status'] != 'unknown'
                    a, b = direct.get('map_objective'), fixed.get('map_objective')
                    if fixed['map_status'] == 'unknown':
                        direct = dict(map_status='unknown', map_objective=None,
                                      map_error='Fixed-model cross-check did not complete')
                    elif direct['map_status'] != fixed['map_status']:
                        raise ValueError('Direct check and fixed-model verification results are inconsistent')
                    elif info['objective'] and direct['map_status'] == 'feasible':
                        if fixed.get('map_objective_status') == 'undetermined':
                            direct.update(map_objective_status='undetermined',
                                          map_objective_error='Fixed-model cross-check objective is undetermined')
                        elif (any(type(v) not in (int, float) or not math.isfinite(v) for v in (a, b))
                                or not objective_matches(a, b)):
                            raise ValueError('Direct check and fixed-model verification objective values are inconsistent')
                else:
                    direct = fixed
            elif direct is None:
                direct = dict(map_status='unknown', map_objective=None,
                    map_error='Complete verification requires residual solving, which is disabled',
                    map_objective_status='undetermined')
            result.update(direct)
            if result['map_status'] in ('error', 'unknown'):
                result['map_error_stage'] = stage
        stage = 'objective'
        if not info['objective']:
            result['map_objective_status'] = 'not_applicable'
        elif result['map_status'] == 'feasible' and result['map_objective_status'] != 'undetermined':
            a, b = result['map_objective'], solution.get('objective')
            if any(type(v) not in (int,float) or not math.isfinite(v) for v in (a,b)):
                result.update(map_objective_status='undetermined', map_objective_error='Invalid original objective value or reformulated solution objective value')
            else:
                result['map_objective_match'] = objective_matches(a, b)
                result['map_objective_status'] = 'matched' if result['map_objective_match'] else 'mismatch'
        result['map_required_variables'] = info['required']
        result['map_assigned_variables'] = sorted(mapped)
        result['map_checks_complete'] = result['map_status'] in ('feasible', 'infeasible')
    except subprocess.TimeoutExpired:
        result.update(map_status='unknown', map_error='Mapping or original-model verification timed out', map_error_stage=stage)
    except Exception as error:
        result.update(map_status='error', map_error=str(error)[:2000], map_error_stage=stage)
    if result['map_status'] in ('error', 'unknown') and payload is not None and stage in ('map', 'assignment'):
        result['map_input_keys'] = native.mapping_input_keys(payload)
        if stage == 'map':
            result['map_error'] += ('\nFields available in mapping input (names only; lists may be truncated): ' +
                json.dumps(result['map_input_keys'], ensure_ascii=False) +
                '\nparameters/indices come from the original model; custom output and parameters added to the reformulated model are not automatically passed to map.py.')
    for field in ('context_time_sec', 'map_time_sec', 'direct_time_sec', 'fallback_time_sec',
                  'verify_compile_sec', 'verify_solve_sec', 'verify_output_sec'):
        result[field] = round(result[field], 6)
    result['verify_time_sec'] = round(time.monotonic()-started, 3)
    return result
