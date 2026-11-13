"""Source-backed MiniZinc declarations and dependency recognition."""
from backends import read_text
from prompt_loader import declaration_context

import json
import re




# The scanner only establishes source boundaries. It does not evaluate MiniZinc.
_chunks = re.compile(r'%[^\r\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\'[^\'\\]*\'|[^%/"\'\[\]{}();]+|.', re.S)
_tokens = re.compile(r'\s+|%[^\r\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\'[^\'\\]*\'|[A-Za-z_][A-Za-z_0-9]*|::|\S', re.S)
_ident = re.compile(r"(?:[A-Za-z_][A-Za-z_0-9]*|'[^'\\]+')\Z")


def tokens(text):
    for match in _tokens.finditer(text):
        value = match[0]
        if match.start() == 0 and value == '\ufeff':
            continue
        if value.isspace() or value.startswith(('%', '/*')):
            continue
        yield value, match.start(), match.end()


def source_items(text):
    """Return complete source spans. On unsupported syntax only the safe prefix survives."""
    stack, start, spans = [], 0, []
    for match in _chunks.finditer(text):
        value = match[0]
        if value.startswith('%'):
            continue
        if value.startswith('/*'):
            if '/*' in value[2:]:
                return spans, 'nested_block_comment_not_supported'
            continue
        if value.startswith(('"', "'")):
            if text.startswith('"""', match.start()):
                return spans, 'triple_quoted_string_not_supported'
            if len(value) == 1 or value[-1] != value[0]:
                return spans, 'unrecognized_quoted_syntax'
            if value.startswith('"') and '\\(' in value:
                return spans, 'interpolated_string_not_supported'
            continue
        if value == '/' and text.startswith('/*', match.start()):
            return spans, 'unterminated_block_comment'
        if value in ('(', '[', '{'):
            stack.append(value)
        elif value in (')', ']', '}'):
            if not stack or stack.pop() != {')': '(', ']': '[', '}': '{'}[value]:
                return spans, 'unbalanced_delimiters'
        elif value == ';' and not stack:
            spans.append((start, match.end()))
            start = match.end()
    if stack:
        return spans, 'unbalanced_delimiters'
    if start < len(text):
        spans.append((start, len(text)))
    return spans, None


def declaration(text):
    """Recognize a bounded top-level declaration header; never infer its value."""
    header, depth = [], 0
    for value, _, _ in tokens(text):
        if len(header) > 256:
            return None
        if not header and value in ('include', 'constraint', 'solve', 'output',
                                     'function', 'predicate', 'test', 'annotation', 'type'):
            return None
        if value == '{' or (value == '(' and 'var' not in header):
            return None  # Keep unsupported complex types as source, without interpretation.
        if value == '(':
            depth += 1
        if value == ')':
            depth -= 1
        if value in ('=', ';'):
            return None
        if value == '[':
            depth += 1
        if value == ']':
            depth -= 1
        if value == ':' and depth == 0:
            break
        header.append(value)
    else:
        return None
    iterator = tokens(text)
    depth = 0
    for value, _, _ in iterator:
        depth += (value in ('[', '(')) - (value in (']', ')'))
        if value == ':' and depth == 0:
            name = next(iterator, ('', 0, 0))[0]
            following = next(iterator, ('', 0, 0))[0]
            if _ident.fullmatch(name) and following in ('', ';', '=', '::'):
                core = header
                if header[:1] == ['array']:
                    level = 0
                    for index, token in enumerate(header):
                        level += (token == '[') - (token == ']')
                        if token == ']' and level == 0:
                            core = header[index + 2:] if header[index + 1:index + 2] == ['of'] else []
                            break
                known_par = bool(core and (core[0] in ('int', 'bool', 'float', 'string') or
                                          core[:3] == ['set', 'of', 'int'] or core[0] in '0123456789'))
                # A type alias may itself contain var; absence of a visible var proves nothing.
                decision = True if 'var' in header else False if known_par else None
                kind = ('variable' if decision is True else
                        'set' if decision is False and core[:2] == ['set', 'of'] else
                        'parameter' if decision is False else 'unknown')
                return dict(name=name, decision=decision, kind=kind)
            return None
    return None


def inspect_model(text):
    spans, _ = source_items(text)
    declarations, includes = [], []
    for start, end in spans:
        item = text[start:end]
        first = next(tokens(item), None)
        if first is None:
            continue
        if first[0] == 'include':
            ts = list(tokens(item))
            if len(ts) in (2, 3) and (len(ts) == 2 or ts[-1][0] == ';'):
                try:
                    name = json.loads(ts[1][0])
                    if isinstance(name, str):
                        includes.append(name)
                except (ValueError, IndexError):
                    pass
        info = declaration(item)
        if info:
            defined = any(value == '=' for value, _, _ in tokens(item))
            declarations.append(dict(info, defined=defined, start=start + first[1], end=end))
    return dict(declarations=declarations, includes=includes)


def requires_data(model_path):
    """Recognize uninitialized parameters that need an external .dzn file."""
    text = read_text(model_path)
    return any(row['decision'] is False and not row['defined']
               for row in inspect_model(text)['declarations'])


def build_contract_context(model_path):
    """Describe declarations from the main source without reading instance data."""
    text = read_text(model_path)
    return declaration_context([
        dict(name=row['name'], kind=row['kind'], text=text[row['start']:row['end']],
             line=len(re.findall(r'\r\n|\r|\n', text[:row['start']])) + 1)
        for row in inspect_model(text)['declarations']])
