"""Compose shared reformulation requirements and language-specific prompts."""
from pathlib import Path
from string import Template
from collections import Counter
import json


def declaration_context(declarations):
    """Retain source declarations with an unambiguous name in the main model."""
    counts = Counter(row['name'] for row in declarations)
    return {'declarations': [row for row in declarations if counts[row['name']] == 1]}


def render_context(context):
    """Serialize the declarations-only interface shared by all language prompts."""
    if not isinstance(context, dict) or set(context) != {'declarations'}:
        raise ValueError('Model interface context must contain only declarations')
    declarations = context['declarations']
    if not isinstance(declarations, list):
        raise ValueError('Model interface declarations must be a list')
    names = set()
    for row in declarations:
        if not isinstance(row, dict) or set(row) != {'name', 'kind', 'text', 'line'}:
            raise ValueError('Each declaration must contain name, kind, text, and line')
        if not isinstance(row['name'], str) or not row['name'] or row['name'] in names:
            raise ValueError('Declaration names must be nonempty and unambiguous')
        names.add(row['name'])
        if row['kind'] not in ('set', 'parameter', 'variable', 'unknown'):
            raise ValueError('Unknown declaration kind: ' + str(row['kind']))
        if not isinstance(row['text'], str) or not row['text'].strip():
            raise ValueError('Declaration text must preserve a nonempty source excerpt')
        if type(row['line']) is not int or row['line'] < 1:
            raise ValueError('Declaration line must be a positive integer')
    return json.dumps(context, ensure_ascii=False)


class PromptSet:
    """Load requirements and templates on demand for each request."""

    _FILES = {
        'SYSTEM_PROMPT': 'system.txt',
        'STANDALONE_TEMPLATE': 'reformulate_standalone.txt',
        'PARAMETERIZED_TEMPLATE': 'reformulate_parameterized.txt',
        'MAP_CONTRACT': 'mapping.txt',
        'REPAIR_TEMPLATE': 'repair.txt',
        'REPORT_MD_PROMPT': 'report_md.txt',
        'REPORT_TEX_PROMPT': 'report_tex.txt',
        'COMMON_REQUIREMENTS': '../common/reformulation.txt',
        'STANDALONE_REQUIREMENTS': '../common/standalone.txt',
        'PARAMETERIZED_REQUIREMENTS': '../common/parameterized.txt',
        'COMMON_MAPPING_CONTRACT': '../common/mapping.txt',
        'INTERFACE_CONTEXT': '../common/interface_context.txt',
    }

    def __init__(self, directory):
        self.directory = Path(directory)

    def __getattr__(self, name):
        if name not in self._FILES:
            raise AttributeError(name)
        text = (self.directory / self._FILES[name]).read_text(encoding='utf-8').removesuffix('\n')
        return self.COMMON_MAPPING_CONTRACT + '\n\n' + text if name == 'MAP_CONTRACT' else text

    def reformulation_request(self, filename, source, context, *, want_map=False,
                              has_external_data=False, language_variables=None):
        """Render shared and backend variables once in the selected outer template."""
        name = 'PARAMETERIZED_TEMPLATE' if has_external_data else 'STANDALONE_TEMPLATE'
        variables = dict(
            filename=filename, source=source, context=render_context(context),
            mapping=self.MAP_CONTRACT if want_map else '',
            common_requirements=self.COMMON_REQUIREMENTS,
            interface_requirements=self.INTERFACE_CONTEXT,
            instance_requirements=(self.PARAMETERIZED_REQUIREMENTS if has_external_data
                                   else self.STANDALONE_REQUIREMENTS))
        language_variables = {} if language_variables is None else language_variables
        if not isinstance(language_variables, dict):
            raise TypeError('Language prompt variables must be a dictionary')
        if any(not isinstance(key, str) or not key.isascii() or not key.isidentifier()
               for key in language_variables):
            raise ValueError('Language prompt variable names must be ASCII identifiers')
        conflicts = variables.keys() & language_variables.keys()
        if conflicts:
            raise ValueError('Language prompt variables cannot override shared variables: '
                             + ', '.join(sorted(conflicts)))
        variables.update(language_variables)
        return self._render_template(name, variables)

    def _render_template(self, name, variables):
        path = self.directory / self._FILES[name]
        try:
            return Template(getattr(self, name)).substitute(variables)
        except KeyError as error:
            raise ValueError(f'Prompt template {path} requires missing variable: {error.args[0]}') from error
        except ValueError as error:
            raise ValueError(f'Invalid prompt template {path}: {error}') from error

    def repair_instruction(self, stage, details):
        return self._render_template('REPAIR_TEMPLATE', dict(stage=stage, details=details))


_ROOT = Path(__file__).resolve().parent / 'prompts'
ampl = PromptSet(_ROOT / 'ampl')
minizinc = PromptSet(_ROOT / 'minizinc')
