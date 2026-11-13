"""AMPL source reading, model dependencies, and model/data loading helpers."""
from backends import run_tool

import re
from pathlib import Path
from prompt_loader import declaration_context


def read_text(path):
    """Read an AMPL source or data file as text."""
    raw = Path(path).read_bytes()
    try:
        return raw.decode('utf-8')
    except UnicodeDecodeError:
        return raw.decode('latin-1')


def quote(path):
    """Quote a path as an AMPL string literal."""
    return '"' + str(path).replace('\\', '\\\\').replace('"', '\\"') + '"'


def load_commands(model, data_files, directory, *, evaluation=False):
    """Stage inputs, optionally removing the main model's trailing run commands."""
    if evaluation:
        source = solve_command_source([model, *data_files], allow_model_trailer=True)
        if source:
            raise ValueError('Embedded solve command is not supported by evaluation: ' + source)
    commands = []
    for index, path in enumerate([model, *data_files]):
        plain = Path(path).resolve()
        text = read_text(plain)
        cleaned = without_run_trailer(text) if evaluation and index == 0 else text
        if cleaned != text:
            staged = Path(directory) / ('input' + str(index) + ('.mod' if index == 0 else '.dat'))
            staged.write_text(cleaned, encoding='utf-8')
            plain = staged
        commands.append(('model ' if index == 0 else 'data ') + quote(plain) + ';\n')
    return ''.join(commands)


def _statement_spans(text):
    """Yield source spans and code, excluding quoted/commented semicolons."""
    tokens = re.finditer(r'''"(?:[^"\\]|\\.|"")*"|'(?:[^'\\]|\\.|'')*'|/\*.*?\*/|\#[^\r\n]*|;|[^;"'\#/]+|.''',
                         text, re.S)
    current, start = [], 0
    for match in tokens:
        token = match[0]
        if token.startswith(('#', '/*')):
            # Preserve a separator between tokens on either side of a comment.
            current.append(' ')
        elif token == ';':
            if ''.join(current).strip():
                yield start, match.end(), ''.join(current).strip()
            current = []
            start = match.end()
        else:
            current.append(token)
    if ''.join(current).strip():
        yield start, len(text), ''.join(current).strip()


def statements(text):
    """Split statements without treating quoted semicolons or comments as code."""
    return [code for _, _, code in _statement_spans(text)]


def without_run_trailer(text):
    """Blank a final sequence of bare run commands, preserving source line numbers.

    Commands embedded in control flow or followed by other statements remain
    unsupported. Assignments, fixes, and declarations are never removed.
    """
    start = len(text)
    for first, end, code in reversed(list(_statement_spans(text))):
        if code not in ('solve', 'end', 'exit', 'quit') or text[end-1] != ';':
            break
        start = first
    return text[:start] + re.sub(r'[^\r\n]', ' ', text[start:])


def references(text, commands=('include', 'model')):
    """Find literal references; leave dynamic expressions to AMPL."""
    references = []
    for statement in statements(text):
        match = re.fullmatch(r'(?:' + '|'.join(re.escape(c) for c in commands) +
                             r')\s+("(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'|[\w./\\:\-]+)',
                             statement, re.I)
        if match:
            name = match[1]
            references.append(name[1:-1] if name[:1] in ('"', "'") else name)
    return references


def sources(model, commands=('include', 'model')):
    """Resolve literal AMPL dependencies relative to the model working directory."""
    root = Path(model).resolve().parent
    found, visited, active = [], set(), set()

    def visit(path):
        path = Path(path).resolve()
        if path in active:
            raise ValueError('Ambiguous AMPL entry points (cyclic dependencies): ' + str(path))
        if path in visited:
            return
        visited.add(path)
        active.add(path)
        found.append(str(path))
        for name in references(read_text(path), commands):
            child = root / name
            if child.is_file():
                visit(child)
        active.remove(path)
    visit(model)
    return found


def solve_command_source(paths, *, allow_model_trailer=False):
    """Locate known solve commands without restricting other AMPL statements.

    Inspect explicit inputs and existing literal dependencies. This is a targeted
    check for unintended optimization, not a parser or syntax validator.
    """
    quoted = r'"(?:[^"\\]|\\.|"")*"|' r"'(?:[^'\\]|\\.|'')*'"
    for number, path in enumerate(paths):
        for source in sources(path, commands=('include', 'model', 'data')):
            text = read_text(source)
            if allow_model_trailer and number == 0 and source == str(Path(path).resolve()):
                text = without_run_trailer(text)
            for statement in statements(text):
                code = re.sub(quoted, ' ', statement, flags=re.S)
                if re.search(r'\bsolve\b', code, re.I):
                    return source
    return None


def inspect_model(text):
    """Describe source-backed AMPL declarations without evaluating expressions."""
    declarations, unsupported = [], []
    for start, statement_end, statement in _statement_spans(text):
        match = re.match(r'(set|param|var)\s+([A-Za-z_]\w*)\b', statement, re.I)
        if not match:
            if not re.match(r'(?:minimize|maximize|s\.t\.|subject\s+to|include|model|data|check)(?:\s|$)', statement, re.I):
                unsupported.append(statement)
            continue
        kind, name = match[1].lower(), match[2]
        tail = statement[match.end():].strip()
        indexing = ''
        if tail.startswith('{'):
            depth, end = 0, None
            for i, char in enumerate(tail):
                depth += (char == '{') - (char == '}')
                if depth == 0:
                    end = i + 1
                    break
            if end is None:
                unsupported.append(statement)
                continue
            indexing, tail = tail[:end], tail[end:].strip()
        prefix = re.match(r'(?:[\s\ufeff]+|\#[^\r\n]*|/\*.*?\*/)*', text[start:statement_end], re.S)
        start += prefix.end()
        declarations.append(dict(name=name, kind=kind, indexing=indexing, start=start, end=statement_end,
                                 definition=tail,
                                 defined=bool(re.match(r'^(?::=|=(?!=)|default\b)', tail)),
                                 source=statement + ';'))
    return dict(declarations=declarations, unsupported=unsupported)


def requires_data(model_path):
    """Recognize declarations that need values from an external data file."""
    for path in sources(model_path):
        text = read_text(path)
        if any(re.match(r'data(?:\s|$)', statement, re.I) for statement in statements(text)):
            return False
        if any(row['kind'] in ('set', 'param') and not row['defined']
               for row in inspect_model(text)['declarations']):
            return True
    return False


def build_contract_context(model_path):
    """Describe the single main model's interface without reading instance data.

    Custom model dependencies are outside the supported LLM input scope. This
    bounded source recognizer reports declarations, not a complete semantic proof.
    """
    text = read_text(model_path)
    kinds = {'set': 'set', 'param': 'parameter', 'var': 'variable'}
    return declaration_context([
        dict(name=row['name'], kind=kinds[row['kind']], text=text[row['start']:row['end']],
             line=len(re.findall(r'\r\n|\r|\n', text[:row['start']])) + 1)
        for row in inspect_model(text)['declarations']])


def instance_context(model_path, data_files, ampl_cmd, timeout=300):
    """Export original scalar/indexed parameters and unindexed one-dimensional sets.

    AMPL evaluates values; Python only reads its CSV output. Unsupported set
    structures are reported, never replaced by invented values.
    """
    import math
    import tempfile
    solve_source = solve_command_source([model_path, *data_files], allow_model_trailer=True)
    if solve_source:
        raise ValueError('Embedded solve command is not supported by context export: ' + solve_source)
    declarations = [r for p in sources(model_path) for r in inspect_model(read_text(p))['declarations']]
    unique_declarations = {}
    for row in declarations:
        key = (row['kind'], row['name'])
        current = unique_declarations.get(key)
        if current is None or (row['indexing'] and not current['indexing']):
            unique_declarations[key] = row
    declarations = list(unique_declarations.values())
    parameters, sets, omitted = {}, {}, []
    with tempfile.TemporaryDirectory(prefix='ampl_context_') as directory:
        script = Path(directory) / 'context.run'
        prefix = 'option eexit 1;\n' + load_commands(model_path, data_files, directory, evaluation=True)

        def run(commands):
            script.write_text(prefix + commands + '\nprint "AMPL_CONTEXT_OK";\n', encoding='utf-8')
            proc = run_tool([ampl_cmd, str(script)], cwd=str(Path(model_path).resolve().parent),
                                  capture_output=True, text=True,
                                  timeout=timeout() if callable(timeout) else timeout)
            if proc.returncode or 'AMPL_CONTEXT_OK' not in proc.stdout.splitlines():
                raise ValueError('AMPL context export failed: ' + (proc.stderr or proc.stdout)[-1500:])

        # AMPL infers dimensions from expressions such as Cartesian products;
        # a missing literal "dimen" attribute does not imply a scalar member.
        dimensions = Path(directory) / 'sets.tsv'
        dimensions.touch()
        run('printf {s in _SETS}: "%s\\t%d\\t%d\\n", s, arity(s), indexarity(s) > '
            + quote(dimensions) + ';')
        shapes = {}
        for line in dimensions.read_text(encoding='utf-8').splitlines():
            name, arity, indexarity = line.split('\t')
            shapes[name] = (int(arity), int(indexarity))
        commands = 'option csvdisplay_header 0;\noption csvdisplay_precision 17;\n'
        exports = []
        for number, row in enumerate(declarations):
            if row['kind'] not in ('set', 'param'):
                continue
            name = row['name']
            if row['kind'] == 'set':
                if shapes.get(name) != (1, 0):
                    omitted.append(name)
                    continue
                expr = '{_llm_context_index in ' + name + '} _llm_context_index'
            else:
                expr = name
            target = Path(directory) / f'value{number}.csv'
            target.touch()
            commands += 'csvdisplay ' + expr + ' > ' + quote(target) + ';\n'
            exports.append((row, target))
        run(commands)
        for row, target in exports:
            rows = [re.findall(r'''"(?:[^"\\]|\\.|"")*"|'(?:[^'\\]|\\.|'')*'|[^,]+''', line)
                    for line in target.read_text(encoding='utf-8').splitlines() if line.strip()]
            symbolic = bool(re.search(r'\bsymbolic\b', row['definition']))
            def value(text):
                text = text.strip()
                if text[:1] in ('"', "'"):
                    delimiter = text[0]
                    if text[-1:] != delimiter:
                        raise ValueError('Unsupported multiline AMPL context value')
                    return text[1:-1].replace(delimiter * 2, delimiter).replace('\\' + delimiter, delimiter)
                if symbolic:
                    return text
                try:
                    result = float(text)
                    if not math.isfinite(result):
                        raise ValueError('Non-finite AMPL context value')
                    return int(result) if result.is_integer() else result
                except ValueError:
                    if row['kind'] == 'set':
                        return text
                    raise
            if row['kind'] == 'set':
                sets[row['name']] = [value(r[-1]) for r in rows]
            elif not row['indexing']:
                if len(rows) != 1 or len(rows[0]) != 1:
                    raise ValueError('Unexpected scalar AMPL CSV shape: ' + row['name'])
                parameters[row['name']] = value(rows[0][0])
            else:
                for record in rows:
                    indices = []
                    for index in record[:-1]:
                        if index.strip()[:1] in ('"', "'"):
                            indices.append(quote(value(index)))
                            continue
                        try:
                            number = float(index)
                            indices.append(str(int(number)) if number.is_integer() else str(number))
                        except ValueError:
                            indices.append(quote(index))
                    parameters[row['name'] + '[' + ','.join(indices) + ']'] = value(record[-1])
    return dict(parameters=parameters, sets=sets, context_omissions=omitted)
