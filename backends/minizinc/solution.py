"""MiniZinc solution codec and original-model verification.

MiniZinc evaluates models/data, types, domains and constraints. Python only scans
source boundaries and handles a deliberately bounded grammar of ground values.
Unsupported syntax fails explicitly; it never becomes an empty assignment.
"""
import solution_protocol as protocol
from backends import run_tool, InputPreparationError, MappingProbeInput, MappingProbeUnavailable
import json
import math
import re
import subprocess
import tempfile
import time
from pathlib import Path

from .contracts import read_text, tokens


class UnsupportedStructure(ValueError):
    """A known limitation of conservative MiniZinc structure recognition."""


def mapping_input_keys(payload, limit=30):
    """Bounded interface diagnostics; never include instance values in errors."""
    result = {}
    for namespace in ('variables', 'parameters', 'indices'):
        values = payload.get(namespace, {})
        keys = sorted(values) if isinstance(values, dict) else []
        result[namespace] = dict(count=len(keys), keys=keys[:limit], truncated=len(keys) > limit)
    return result


def items(text):
    """Split top-level items, including strings with nested interpolation."""
    spans, stack, start, pos = [], [], 0, 0

    def comment(i):
        if text.startswith('%', i):
            end = text.find('\n', i)
            return len(text) if end < 0 else end
        if text.startswith('/*', i):
            end = text.find('*/', i + 2)
            if end < 0:
                raise ValueError('Unterminated MiniZinc block comment')
            if '/*' in text[i + 2:end]:
                raise UnsupportedStructure('Nested MiniZinc block comments are not supported')
            return end + 2
        return i

    def quoted(i, quote):
        if text.startswith('"""', i):
            raise UnsupportedStructure('Verification conversion of MiniZinc triple-quoted strings is not supported')
        i += 1
        while i < len(text):
            if text[i] == quote:
                return i + 1
            if text.startswith('\\(', i) and quote == '"':
                i, depth = i + 2, 1
                while i < len(text) and depth:
                    end = comment(i)
                    if end != i:
                        i = end
                    elif text[i] in ('"', "'"):
                        i = quoted(i, text[i])
                    else:
                        depth += (text[i] == '(') - (text[i] == ')')
                        i += 1
                if depth:
                    raise ValueError('Unterminated MiniZinc string interpolation')
            elif text[i] == '\\':
                i += 2
            else:
                i += 1
        raise ValueError('Unterminated MiniZinc string')

    while pos < len(text):
        end = comment(pos)
        if end != pos:
            pos = end
            continue
        char = text[pos]
        if char in ('"', "'"):
            pos = quoted(pos, char)
            continue
        if char in '([{':
            stack.append(char)
        elif char in ')]}':
            if not stack or stack.pop() != {')': '(', ']': '[', '}': '{'}[char]:
                raise ValueError('Mismatched MiniZinc delimiters')
        elif char == ';' and not stack:
            spans.append((start, pos + 1))
            start = pos + 1
        pos += 1
    if stack:
        raise ValueError('Unclosed MiniZinc delimiters')
    output = [text[a:b] for a, b in spans]
    if next(tokens(text[start:]), None):
        # MiniZinc permits the final item to omit its semicolon.
        output.append(text[start:] + '\n;')
    return output


def declaration_header(ts):
    depth = 0
    for index, (value, _, _) in enumerate(ts):
        if value == ':' and depth == 0:
            if index + 2 >= len(ts) or not _identifier.fullmatch(ts[index+1][0]) or ts[index+2][0] not in ('=', '::', ';'):
                break
            return [t[0] for t in ts[:index]], ts[index+1][0]
        depth += (value in ('(', '[', '{')) - (value in (')', ']', '}'))
    raise UnsupportedStructure('Cannot reliably extract MiniZinc declaration: ' + ' '.join(t[0] for t in ts[:40]))


def inline_source(path, include_dirs=(), active=()):
    path = Path(path).resolve()
    if path in active:
        raise ValueError('MiniZinc include cycle: ' + str(path))
    try:
        text = read_text(path)
    except UnicodeDecodeError as error:
        raise ValueError('MiniZinc verification requires UTF-8 input') from error
    output = []
    for item in items(text.lstrip('\ufeff')):
        ts = list(tokens(item))
        if ts and ts[0][0] == 'include':
            name = json.loads(ts[1][0])
            child = next((p / name for p in [path.parent, *map(Path, include_dirs)]
                          if (p / name).is_file()), None)
            if child:
                item = inline_source(child, include_dirs, (*active, path))
        output.append(item)
    return '\n'.join(output)


def model_info(text, data_texts=()):
    """Conservative declaration coverage; unknown headers cannot silently pass."""
    rows, objective, solve_count, clean, source_bound = {}, None, 0, [], set()
    domain_names = set()
    annotation_names = set()
    for item in items(text):
        ts = list(tokens(item))
        if ts and ts[0][0] == 'enum':
            domain_names.add(ts[1][0])
        elif (len(ts) == 3 and ts[0][0] == 'annotation' and ts[2][0] == ';'
              and _identifier.fullmatch(ts[1][0])):
            # Bare annotation declarations may have a separate assignment.
            # They are neither solution variables nor serializable parameters.
            annotation_names.add(ts[1][0])
        elif ts and ts[0][0] == 'ann':
            _, name = declaration_header(ts)
            annotation_names.add(name)
        elif ts and ts[0][0] in ('set', 'par'):
            header, name = declaration_header(ts)
            if 'set' in header and 'var' not in header:
                domain_names.add(name)
    for item in items(text):
        ts = list(tokens(item))
        if not ts:
            continue
        first = ts[0][0]
        if first == 'solve':
            solve_count += 1
            depth = 0
            for value, a, b in ts[1:]:
                if depth == 0 and value in ('satisfy', 'minimize', 'maximize'):
                    objective = item[b:item.rfind(';')].strip() if value != 'satisfy' else None
                    break
                depth += (value in ('(', '[', '{')) - (value in (')', ']', '}'))
            else:
                raise UnsupportedStructure('Cannot reliably extract MiniZinc solve item')
            continue
        if first == 'output':
            continue
        if len(ts) >= 3 and ts[1][0] == '=':
            if any(t[0] == '_' for t in ts[2:]):
                raise UnsupportedStructure('Top-level bindings containing anonymous decision variables are not supported')
            source_bound.add(first)
            clean.append(item)
            continue
        if first == 'type':
            raise UnsupportedStructure('MiniZinc mapping verification does not currently support type aliases; variable coverage cannot be confirmed')
        if first not in ('include', 'constraint', 'function', 'predicate', 'test', 'annotation', 'ann', 'enum'):
            header, name = declaration_header(ts)
            core = header
            is_list = core[:2] == ['list', 'of']
            if is_list:
                # MiniZinc list is shorthand for array[int] of the same element
                # type. Keep the source intact; share array metadata and codecs.
                core = core[2:]
            elif core[0] == 'array':
                level = 0
                for index, value in enumerate(core[1:], 1):
                    level += (value == '[') - (value == ']')
                    if level == 0 and value == ']':
                        core = core[index+2:] if core[index+1:index+2] == ['of'] else []
                        break
            if not core:
                raise UnsupportedStructure('Cannot determine MiniZinc declaration type: ' + name)
            if 'var' in core:
                decision = True
            elif (core[0] in ('par', 'int', 'bool', 'float', 'string', 'opt', 'set', '{') or
                  core[0] in domain_names or any(core[i:i+2] == ['.', '.'] for i in range(len(core)-1))):
                decision = False
            else:
                raise UnsupportedStructure('Cannot determine MiniZinc declaration type: ' + name)
            row = dict(name=name, decision=decision)
            if row['name'] in rows:
                raise ValueError('Duplicate MiniZinc declaration: ' + row['name'])
            depth, equal, dim, array_depth, in_header = 0, False, 0, 0, True
            if is_list:
                dim = 1
            for value, _, _ in ts:
                if value == ':' and depth == 0:
                    in_header = False
                if value == '=' and depth == 0:
                    equal = True
                if in_header and ts[0][0] == 'array':
                    if value == '[' and depth == 0:
                        array_depth, dim = 1, 1
                    elif value == ',' and array_depth and depth == 1:
                        dim += 1
                    elif value == ']' and depth == 1:
                        array_depth = 0
                depth += (value in ('(', '[', '{')) - (value in (')', ']', '}'))
            rows[row['name']] = dict(row, defined=equal, dim=dim,
                                     string='string' in header)
            if equal and any(t[0] == '_' for t in ts):
                raise UnsupportedStructure('Top-level definitions containing anonymous decision variables are not supported')
        clean.append(item)
    if solve_count != 1:
        raise ValueError('MiniZinc verification requires exactly one solve item')
    for name in source_bound:
        if name not in rows and name not in domain_names and name not in annotation_names:
            raise ValueError('Cannot locate the declaration for top-level assignment target: ' + name)
        if name in rows:
            rows[name]['defined'] = True
    bound = set()
    for data in data_texts:
        for item in items(data):
            ts = list(tokens(item))
            if len(ts) >= 3 and ts[1][0] == '=':
                if ts[0][0] in rows and rows[ts[0][0]]['decision']:
                    if any(t[0] == '_' for t in ts[2:]):
                        raise UnsupportedStructure('Data bindings containing anonymous decision variables are not supported')
                    bound.add(ts[0][0])
            else:
                raise UnsupportedStructure('Unsupported data statement: ' + item.strip()[:120])
    required = sorted(n for n, r in rows.items() if r['decision'] and not r['defined'] and n not in bound)
    return dict(rows=rows, required=required, body='\n'.join(clean), objective=objective)


_value_token = re.compile(r'\s+|%[^\r\n]*|"(?:\\.|[^"\\])*"|\'[^\'\\]*\'|'
                          r'[+-]?(?:\d+\.\d+(?:[eE][+-]?\d+)?|\d+[eE][+-]?\d+|\d+)|'
                          r'[A-Za-z_][A-Za-z_0-9]*|\.\.|<>|[\[\]{}(),=;:|]')
_identifier = re.compile(r"(?:[A-Za-z_][A-Za-z_0-9]*|'[^'\\]+')\Z")


def read_dzn(text):
    """Parse compiler-produced ground DZN, never arbitrary data expressions."""
    parts, pos = [], 0
    for match in _value_token.finditer(text):
        if match.start() != pos:
            raise ValueError('Unsupported MiniZinc solution value: ' + text[pos:pos + 80])
        pos = match.end()
        if not match[0].isspace() and not match[0].startswith('%'):
            parts.append(match[0])
    if text[pos:].strip():
        raise ValueError('Unsupported MiniZinc solution trailer: ' + text[pos:pos + 80])
    cursor = 0

    def take(expected=None):
        nonlocal cursor
        if cursor >= len(parts):
            raise ValueError('MiniZinc solution value ended unexpectedly')
        value = parts[cursor]
        cursor += 1
        if expected is not None and value != expected:
            raise ValueError('Expected ' + expected + ' in MiniZinc solution value, got ' + value)
        return value

    def sequence(end):
        result = []
        if cursor < len(parts) and parts[cursor] == end:
            take(end)
            return result
        while True:
            result.append(value())
            sep = take()
            if sep == end:
                return result
            if sep != ',':
                raise ValueError('MiniZinc set or array is missing a comma')

    def index_values(values):
        if values and all(isinstance(v, int) and not isinstance(v, bool) for v in values):
            if values == list(range(values[0], values[0] + len(values))):
                return {'range': [values[0], values[-1]]}
        return {'set': values}

    def array():
        if cursor < len(parts) and parts[cursor] == '|':
            take('|')
            columns, labels, rows = None, [], []
            while cursor < len(parts) and parts[cursor] != ']':
                # Matrix segments are delimited by |. A header ends in a colon.
                end, depth = cursor, 0
                while end < len(parts):
                    if parts[end] == '|' and depth == 0:
                        break
                    depth += (parts[end] in ('[', '(', '{')) - (parts[end] in (']', ')', '}'))
                    end += 1
                if end == len(parts):
                    raise ValueError('MiniZinc two-dimensional array is missing |')
                if end > cursor and parts[end-1] == ':':
                    if columns is not None or rows:
                        raise ValueError('Duplicate MiniZinc two-dimensional array column indices')
                    columns = []
                    while cursor < end:
                        columns.append(value()); take(':')
                else:
                    first = value()
                    if cursor < end and parts[cursor] == ':':
                        take(':'); labels.append(first)
                        first = value()
                    elif labels:
                        raise ValueError('Incomplete MiniZinc two-dimensional array row indices')
                    row = [first]
                    while cursor < end:
                        take(','); row.append(value())
                    rows.append(row)
                take('|')
            take(']')
            width = len(rows[0]) if rows else 0
            if any(len(row) != width for row in rows) or (columns is not None and len(columns) != width):
                raise ValueError('MiniZinc two-dimensional array indices and shape are inconsistent')
            return {'array': [v for row in rows for v in row], 'index_sets': [
                index_values(labels) if labels else {'range': [1, len(rows)]},
                index_values(columns) if columns is not None else {'range': [1, width]}]}
        if cursor < len(parts) and parts[cursor] == ']':
            take(']'); return []
        first = value()
        if cursor < len(parts) and parts[cursor] == ':':
            labels, values = [], []
            while True:
                labels.append(first); take(':'); values.append(value())
                sep = take()
                if sep == ']':
                    break
                if sep != ',':
                    raise ValueError('MiniZinc indexed array is missing a comma')
                first = value()
            return {'array': values, 'index_sets': [index_values(labels)]}
        result = [first]
        while True:
            sep = take()
            if sep == ']':
                return result
            if sep != ',':
                raise ValueError('MiniZinc array is missing a comma')
            result.append(value())

    def value():
        token = take()
        if token == '[':
            result = array()
        elif token == '{':
            result = {'set': sequence('}')}
        elif token == '<>':
            result = None
        elif token in ('true', 'false'):
            result = token == 'true'
        elif token.startswith('"'):
            result = json.loads(token)
        elif re.fullmatch(r'array[1-6]d', token):
            take('(')
            args = sequence(')')
            dimension = int(token[5])
            if len(args) != dimension + 1 or not isinstance(args[-1], list):
                raise ValueError('Invalid MiniZinc arrayNd solution value format')
            result = {'array': args[-1], 'index_sets': args[:-1]}
        elif re.fullmatch(r'[+-]?\d+', token):
            result = int(token)
        elif re.fullmatch(r'[+-]?(?:\d+\.\d+(?:[eE][+-]?\d+)?|\d+[eE][+-]?\d+)', token):
            result = float(token)
            if not math.isfinite(result):
                raise ValueError('MiniZinc solution contains a non-finite number')
        elif _identifier.fullmatch(token):
            result = {'enum': token}
        else:
            raise ValueError('Unsupported MiniZinc solution value: ' + token)
        if cursor < len(parts) and parts[cursor] == '..':
            take('..')
            result = {'range': [result, value()]}
        return result

    result = {}
    while cursor < len(parts):
        name = take()
        if not _identifier.fullmatch(name) or name in result:
            raise ValueError('Invalid or duplicate MiniZinc solution variable name: ' + name)
        take('=')
        parsed = value()
        if isinstance(parsed, list) and not name.startswith('_indices_'):
            parsed = {'array': parsed, 'index_sets': [{'range': [1, len(parsed)]}]}
        result[name] = parsed
        take(';')
    return result


def literal(value):
    """Serialize only ground values. Mapping code cannot inject model statements."""
    if value is None:
        return '<>'
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and math.isfinite(value):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return '[' + ','.join(map(literal, value)) + ']'
    if isinstance(value, dict):
        if set(value) == {'enum'} and isinstance(value['enum'], str) and _identifier.fullmatch(value['enum']):
            return value['enum']
        if set(value) == {'set'} and isinstance(value['set'], list):
            return '{' + ','.join(map(literal, value['set'])) + '}'
        if set(value) == {'range'} and isinstance(value['range'], list) and len(value['range']) == 2:
            return '..'.join(map(literal, value['range']))
        if set(value) == {'array', 'index_sets'} and isinstance(value['array'], list) and isinstance(value['index_sets'], list):
            n = len(value['index_sets'])
            if 1 <= n <= 6:
                return f'array{n}d(' + ','.join([*map(literal, value['index_sets']), literal(value['array'])]) + ')'
    raise ValueError('Invalid or unsupported MiniZinc mapped value: ' + repr(value)[:200])


def same_index(left, right):
    if left == right:
        return True
    def key(value):
        if isinstance(value, dict) and set(value) == {'set'}:
            return {json.dumps(v, sort_keys=True) for v in value['set']}
        return None
    for a, b in ((left, right), (right, left)):
        if isinstance(a, dict) and set(a) == {'range'} and isinstance(b, dict) and set(b) == {'set'}:
            lo, hi = a['range']
            elements = b['set']
            if all(isinstance(x, int) and not isinstance(x, bool) for x in (lo, hi, *elements)):
                return len(set(elements)) == max(0, hi-lo+1) and all(lo <= x <= hi for x in elements)
    return key(left) is not None and key(left) == key(right)


def run_map(path, solution, timeout):
    return protocol.run_map(path, solution, timeout, validate_values)


def validate_values(variables):
    for name, value in variables.items():
        if not isinstance(name, str) or not _identifier.fullmatch(name):
            raise ValueError('map.py output contains an invalid variable name: ' + str(name))
        literal(value)


def stage_data(directory, paths):
    output = []
    for index, path in enumerate(paths):
        try:
            text = read_text(path)
        except UnicodeDecodeError as error:
            raise ValueError('MiniZinc data must be UTF-8') from error
        target = Path(directory)/f'data{index}.dzn'
        target.write_text(text, encoding='utf-8')
        output.append(str(target))
    return output


def index_expr(name, dim, axis):
    return f'index_set({name})' if dim == 1 else f'index_set_{axis}of{dim}({name})'


def show_value(name, row):
    if row['dim']:
        n = row['dim']
        parts = [json.dumps(f'array{n}d(')]
        for axis in range(1, n + 1):
            parts += [f'show({index_expr(name, n, axis)})', '","']
        parts += [f'show(array1d({name}))', '")"']
        return ' ++ '.join(parts)
    return f'{"showJSON" if row["string"] else "show"}({name})'


def instance_context(info, directory, data, command, timeout):
    """Evaluate parameters/index sets with an .ozn, without solving a new instance."""
    outputs, parameter_keys, index_keys = [], {}, {}
    index_declarations = []
    identifiers = {t[0] for t in tokens(info['body'])}
    for number, (name, row) in enumerate(info['rows'].items()):
        if not row['decision']:
            key = f'param{number}'
            parameter_keys[name] = key
            outputs += [json.dumps(key + ' = '), show_value(name, row), json.dumps(';\n')]
        elif row['dim']:
            key = f'_indices_{number}'
            index_keys[name] = key
            exprs = []
            for axis in range(1, row['dim'] + 1):
                index_name = f'_llm4_context_index_{number}_{axis}'
                while index_name in identifiers:
                    index_name += '_'
                identifiers.add(index_name)
                # Ground the index string during compilation. Leaving index_set
                # in output can retain derived arrays with unbound solver values
                # in the OZN, even though only their shape is needed.
                index_declarations.append(
                    f'string: {index_name} = show({index_expr(name, row["dim"], axis)});')
                exprs.append(index_name)
            outputs += [json.dumps(key + ' = ['),
                        (' ++ "," ++ ').join(exprs), json.dumps('];\n')]
    if not outputs:
        return {}, {}
    model, ozn = Path(directory)/'context.mzn', Path(directory)/'context.ozn'
    # Parameter values and array index sets do not depend on asserted constraints.
    # Avoid flattening those constraints before the solution is even available.
    metadata_source = '\n'.join(item for item in items(info['body'])
                                if next(tokens(item), ('',))[0] != 'constraint')
    model.write_text(metadata_source + '\n' + '\n'.join(index_declarations) +
                     '\nsolve satisfy;\noutput [' + ','.join(outputs) + '];', encoding='utf-8')
    proc = run_tool([*command, '-c', '--output-ozn-to-file', str(ozn), str(model), *data],
                          cwd=directory, capture_output=True, text=True, timeout=timeout())
    if proc.returncode:
        raise ValueError('Original-model context compilation failed: ' + (proc.stderr or proc.stdout)[-1500:])
    proc = run_tool([command[0], '--ozn-file', str(ozn)], input='----------\n',
                          cwd=directory, capture_output=True, text=True, timeout=timeout())
    if proc.returncode:
        raise ValueError('Original-model context evaluation failed: ' + (proc.stderr or proc.stdout)[-1500:])
    values = read_dzn('\n'.join(line for line in proc.stdout.splitlines() if not line.startswith(('----------', '=========='))))
    return ({name: values[key] for name, key in parameter_keys.items()},
            {name: values[key] for name, key in index_keys.items()})


def verification_process(command, directory, remaining, metrics, phase, input=None):
    started = time.monotonic()
    try:
        return run_tool(command, input=input, cwd=directory, capture_output=True,
                              text=True, timeout=remaining())
    finally:
        if metrics is not None:
            field = 'verify_' + phase + '_sec'
            metrics[field] = metrics.get(field, 0) + time.monotonic()-started


class VerificationError(ValueError):
    def __init__(self, message, event=None):
        super().__init__(message)
        self.event = event or {}


def verification_output(proc):
    """Read native JSON events; warnings never count as proof of infeasibility."""
    status, solution, diagnostic, failure = None, None, proc.stderr or '', None
    for line in proc.stdout.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if not isinstance(event, dict):
            continue
        if event.get('type') == 'solution':
            solution = read_dzn(event.get('output', {}).get('default', ''))
        elif event.get('type') == 'status':
            status = event.get('status')
        elif event.get('type') == 'error':
            diagnostic += event.get('message', str(event))
            status = 'ERROR'
            failure = event
    if proc.returncode or failure is not None:
        raise VerificationError('MiniZinc original-model verification failed: ' + (diagnostic or proc.stdout)[-1500:], failure)
    return solution, status


def compile_verification(model, data, command, remaining, metrics=None):
    """Compile once and retain both files for evaluation or residual solving."""
    model = Path(model)
    fzn, ozn = model.with_suffix('.fzn'), model.with_suffix('.ozn')
    if metrics is not None:
        metrics['verify_compile_count'] = metrics.get('verify_compile_count', 0) + 1
    proc = verification_process([*command, '-c', '--json-stream',
        '--output-fzn-to-file', str(fzn), '--output-ozn-to-file', str(ozn),
        str(model), *data], model.parent, remaining, metrics, 'compile')
    _, status = verification_output(proc)
    if status != 'UNSATISFIABLE' and not (fzn.is_file() and ozn.is_file()):
        raise VerificationError('MiniZinc verification compilation did not generate complete fzn/ozn files')
    return fzn, ozn, status


def residual_problem(fzn):
    """Only a recognised, constraint-free satisfy model may skip the solver."""
    solve_seen = False
    for item in items(Path(fzn).read_text(encoding='utf-8')):
        ts = [t[0] for t in tokens(item)]
        if not ts:
            continue
        if ts[0] == 'predicate':
            continue  # FlatZinc signatures do not create decision variables.
        if ts[0] == 'solve':
            if ts != ['solve', 'satisfy', ';'] or solve_seen:
                return True
            solve_seen = True
        elif ts[0] not in ('int', 'float', 'bool', 'set', 'array') or 'var' in ts:
            return True  # Includes constant false constraints and unfamiliar items.
    return not solve_seen


def ground_output(ozn):
    """OZN input parameters need solver values, even if dummy evaluation succeeds."""
    for item in items(Path(ozn).read_text(encoding='utf-8')):
        ts = list(tokens(item))
        if not ts or ts[0][0] in ('output', 'function', 'test', 'enum'):
            continue
        if ts[0][0] not in ('int', 'float', 'bool', 'string', 'set', 'array'):
            return False
        depth, defined = 0, False
        for token, _, _ in ts:
            if token == '=' and depth == 0:
                defined = True
            depth += (token in ('(', '[', '{')) - (token in (')', ']', '}'))
        if not defined:
            return False
    return True


def decision_domains(info):
    """Keep declared domains explicit, including arrays of sets after substitution."""
    if '_verification_domains' in info:
        return info['_verification_domains']
    checks = []
    element_name = '_llm4_domain_value'
    identifiers = {t[0] for t in tokens(info['body'])}
    while element_name in identifiers:
        element_name += '_'
    for item in items(info['body']):
        ts = list(tokens(item))
        if not ts or ts[0][0] in ('include', 'constraint', 'function', 'predicate', 'test', 'annotation', 'ann', 'enum'):
            continue
        if len(ts) > 2 and ts[1][0] == '=':
            continue
        _, name = declaration_header(ts)
        row = info['rows'][name]
        if not row['decision']:
            continue
        depth = 0
        for colon, (token, _, _) in enumerate(ts):
            if token == ':' and depth == 0:
                break
            depth += (token in ('(', '[', '{')) - (token in (')', ']', '}'))
        start = next(i for i, t in enumerate(ts[:colon]) if t[0] == 'var') + 1
        optional = ts[start][0] == 'opt'
        start += optional
        domain = item[ts[start][1]:ts[colon][1]].strip()
        domain_tokens = [t[0] for t in ts[start:colon]]
        if len(domain_tokens) == 1 and domain_tokens[0] in ('int', 'float', 'bool', 'string'):
            continue
        operation = 'in'
        if [t[0] for t in ts[start:start+2]] == ['set', 'of']:
            domain = item[ts[start+2][1]:ts[colon][1]].strip()
            if domain_tokens == ['set', 'of', 'int']:
                continue
            operation = 'subset'
        value = element_name if row['dim'] else name
        expression = f'{value} {operation} ({domain})'
        if optional:
            expression = f'if occurs({value}) then deopt({value}) {operation} ({domain}) else true endif'
        if row['dim']:
            expression = f'forall({value} in array1d({name}))({expression})'
        checks.append(('domain_' + name, expression))
    return checks


def evaluate_verification(ozn, command, remaining, metrics=None, input='----------\n'):
    proc = verification_process([command[0], '--ozn-file', str(ozn), '--json-stream'],
        Path(ozn).parent, remaining, metrics, 'output', input=input)
    return verification_output(proc)


def solve_verification(fzn, ozn, command, remaining, metrics=None):
    if metrics is not None:
        metrics['verify_solver_calls'] = metrics.get('verify_solver_calls', 0) + 1
    proc = verification_process([*command, '--time-limit', str(max(1, int(remaining()*1000))),
        str(fzn)], Path(fzn).parent, remaining, metrics, 'solve')
    if proc.returncode:
        raise ValueError('MiniZinc residual model solving failed: ' + (proc.stderr or proc.stdout)[-1500:])
    return evaluate_verification(ozn, command, remaining, metrics, input=proc.stdout)


def fixed_check(info, data, command, directory, mapped, remaining, metrics=None):
    """Substitute independent values early; solve only the residual original model."""
    required = set(info['required'])
    bindings = [f'{name} = {literal(mapped[name])};' for name in sorted(required)]
    mapped_data = Path(directory)/'mapped.dzn'
    mapped_data.write_text('\n'.join(bindings), encoding='utf-8')
    # Source/data definitions must be recomputed, never overridden by map output.
    constraints = ['constraint ' + expression + ';' for _, expression in decision_domains(info)]
    constraints += ['constraint ' + info.get('_verification_bindings', {}).get(name, f'{name} = ({literal(value)})') + ';'
                    for name, value in mapped.items() if name not in required]
    objective = info['objective']
    fields = ['"_objective = "', f'show({objective})', '";\\n"'] if objective else ['"verified = true;\\n"']
    violations = info.get('_verification_violations', [])
    # Diagnostics must not introduce residual output dependencies and thereby
    # make an otherwise ground objective appear undetermined. Derived decision
    # values are conservatively omitted from fixed-model violation reporting.
    if violations and not any(info['rows'].get(name, {}).get('decision') and name not in required
                              for expression in violations for name, _, _ in tokens(expression)):
        fields += ['"_max_violation = "', f'show({max_violation_expression(violations)})', '";\\n"']
    output = 'output [' + ', '.join(fields) + '];'
    model = Path(directory)/'verify.mzn'
    model.write_text(info['body'] + '\n' + '\n'.join(constraints) + '\nsolve satisfy;\n' + output, encoding='utf-8')
    try:
        fzn, ozn, status = compile_verification(model, [*data, str(mapped_data)], command, remaining, metrics)
    except VerificationError as error:
        # Native substitution reports an out-of-domain decision value as a
        # parameter error. Only this explicit contradiction means infeasible;
        # type errors, invalid input parameters and other failures remain errors.
        match = re.match(r"parameter (?:value )?out of range: declared domain of [`']([^`']+)'", error.event.get('message', ''))
        if match and info['rows'].get(match[1], {}).get('decision'):
            return dict(map_status='infeasible', map_objective=None)
        raise
    if status == 'UNSATISFIABLE':
        return dict(map_status='infeasible', map_objective=None)
    residual = residual_problem(fzn)
    objective_ground = ground_output(ozn)
    if not residual and not objective_ground:
        return dict(map_status='unknown', map_objective=None, map_error='Output requires variable values that were not provided')
    solution, status = (solve_verification(fzn, ozn, command, remaining, metrics) if residual
                         else evaluate_verification(ozn, command, remaining, metrics))
    if status == 'UNSATISFIABLE':
        return dict(map_status='infeasible', map_objective=None)
    if solution is not None:
        violation = solution.get('_max_violation')
        if violation is not None and violation > info.get('_verification_tolerance', 0):
            return dict(map_status='unknown', map_objective=None, map_max_viol=violation,
                        map_error='The completion returned by fixed-model solving exceeds the specified numeric tolerance; feasibility cannot be confirmed')
        if residual and objective and not objective_ground:
            return dict(map_status='feasible', map_objective=None,
                        map_objective_status='undetermined',
                        map_objective_error='The original objective depends on residual variables; the objective value from any completion cannot be used for consistency checking')
        return dict(map_status='feasible', map_objective=solution.get('_objective'),
                    map_max_viol=solution.get('_max_violation'))
    return dict(map_status='unknown', map_objective=None, map_error='Verification found no solution and did not prove unsatisfiability')




def _zero_index_members(axis):
    if isinstance(axis, dict) and 'range' in axis:
        lo, hi = axis['range']
        return range(lo, hi + 1)
    if isinstance(axis, dict) and 'set' in axis:
        return axis['set']
    raise UnsupportedStructure('Unsupported MiniZinc index representation')


def mapping_input(model_text, model_path, sources, instance_id, command, solver):
    """Preserve native types/indices; decline artificial values for unsupported types."""
    include_dirs = list(dict.fromkeys(str(Path(p).parent) for p in sources))
    cmd = [command, '--solver', solver]
    for parent in include_dirs:
        cmd += ['-I', parent]
    try:
        original = inline_source(model_path, include_dirs)
        original_info = model_info(original, [])
        solution_info = model_info(model_text, [])
    except UnsupportedStructure as error:
        return MappingProbeUnavailable('Instance structure cannot be safely described: ' + str(error))
    types = {}
    for item in items(model_text):
        ts = list(tokens(item))
        if not ts or ts[0][0] in ('include', 'constraint', 'solve', 'output', 'function',
                                  'predicate', 'test', 'annotation', 'ann', 'enum', 'type'):
            continue
        if len(ts) > 1 and ts[1][0] == '=':
            continue
        header, name = declaration_header(ts)
        if name in solution_info['required']:
            # Only a visible numeric or boolean type is sufficient evidence here.
            if 'set' in header or 'opt' in header or 'string' in header:
                return MappingProbeUnavailable('Zero input is not defined for set, optional, or string decision values')
            at = header.index('var') if 'var' in header else -1
            tail = header[at + 1:] if at >= 0 else []
            if not tail or not (tail[0] in ('int', 'float', 'bool', '-', '+', '{')
                                or re.fullmatch(r'\d+(?:\.\d+)?', tail[0])):
                return MappingProbeUnavailable('Zero input is not defined for enum or unconfirmed decision types')
            types[name] = 'bool' if tail[0] == 'bool' else 'number'
    with tempfile.TemporaryDirectory(prefix='mzn_mapping_context_') as directory:
        data = stage_data(directory, [])
        parameters, indices = instance_context(original_info, directory, data, cmd, lambda: 300)
        _, solution_indices = instance_context(solution_info, directory, data, cmd, lambda: 300)
    variables = {}
    for name in solution_info['required']:
        row = solution_info['rows'][name]
        zero = False if types[name] == 'bool' else 0
        if row['dim']:
            axes = solution_indices.get(name)
            if axes is None or len(axes) != row['dim']:
                raise InputPreparationError('Missing solution array index context: ' + name)
            try:
                count = math.prod(len(_zero_index_members(axis)) for axis in axes)
            except UnsupportedStructure as error:
                return MappingProbeUnavailable('Array indices cannot be safely represented: ' + str(error))
            if count > 100000:
                return MappingProbeUnavailable('Artificial array would exceed the mapping-check size limit')
            variables[name] = dict(array=[zero] * count, index_sets=axes)
        else:
            variables[name] = zero
    return MappingProbeInput(protocol.make_input(variables, 0, parameters=parameters, indices=indices,
        instance_id=instance_id))


def check_map_run(code, payload, timeout=120):
    return protocol.check_map(code, payload, timeout, validate_values)


def max_violation_expression(expressions):
    return 'max([0.0' + ''.join(',(' + expr + ')' for expr in expressions) + '])'
