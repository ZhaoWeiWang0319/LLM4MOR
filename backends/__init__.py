"""Shared source utilities and modeling-language backend discovery."""
import contextlib
import contextvars
from dataclasses import dataclass
import errno
import importlib
import json
import os
from pathlib import Path
import pkgutil
import re
import shutil
import subprocess
import sys
import time


def read_text(path):
    return Path(path).read_text(encoding='utf-8')


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')


def resolve_data_file(value, suffixes):
    """Resolve one optional data file for independent verification."""
    if value is None:
        return []
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise ValueError('Data path is not a file: ' + str(path))
    if not str(path).lower().endswith(tuple(suffixes)):
        raise ValueError('Data file has an unsupported extension: ' + str(path))
    return [path]


def stem(path):
    value = str(path)
    suffixes = {s for b in available().values() for s in (*b.model_suffixes, *b.data_suffixes)}
    for suffix in sorted(suffixes, key=len, reverse=True):
        if value.lower().endswith(suffix):
            return value[:-len(suffix)]
    return value


CHECK_STATUSES = frozenset(('passed', 'failed', 'skipped', 'syntax_only'))


def check_record(instance_id, status, message=None, *, failure=None):
    """Create the shared analysis check record; exceptions remain in memory."""
    if not isinstance(instance_id, str) or not instance_id:
        raise ValueError('Check instance_id must be a nonempty string')
    if status not in CHECK_STATUSES:
        raise ValueError('Unknown analysis check status: ' + str(status))
    if message is not None and not isinstance(message, str):
        raise TypeError('Check message must be text')
    if failure is not None and (status != 'failed' or not isinstance(failure, Exception)):
        raise ValueError('Only failed checks may carry an internal exception')
    if status == 'failed' and not message:
        message = (str(failure) if failure is not None else '') or 'Check failed without a diagnostic.'
    record = dict(instance_id=instance_id, status=status)
    if message:
        record['message'] = message
    if failure is not None:
        record['_failure'] = failure
    return record


class ToolError(RuntimeError):
    """An execution failure that cannot be repaired by changing generated code."""


class ToolConfigurationError(ToolError, ValueError):
    """A required executable, backend, or license is unavailable."""


class ToolRetriesExhausted(ToolError):
    """The local retry budget for an unchanged tool invocation is exhausted."""


class InputPreparationError(ValueError):
    """Required input cannot be prepared; generated mapping code is not at fault."""


class MappingExecutionError(ValueError):
    """The generated mapping script failed execution or its output protocol."""


@dataclass(frozen=True)
class MappingProbeInput:
    """An artificial mapping input ready for shared protocol validation."""
    payload: dict

    def __post_init__(self):
        if not isinstance(self.payload, dict):
            raise TypeError('Mapping probe payload must be a dictionary')


@dataclass(frozen=True)
class MappingProbeUnavailable:
    """An explicit limitation of artificial input construction, not a failure."""
    reason: str

    def __post_init__(self):
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError('Mapping probe limitation requires a nonempty reason')


_tool_policy = contextvars.ContextVar('analysis_tool_policy', default=None)
_tool_environment = contextvars.ContextVar('native_tool_environment', default=None)


@contextlib.contextmanager
def tool_environment(environment):
    """Scope native subprocess configuration without changing process globals."""
    token = _tool_environment.set(environment)
    try:
        yield
    finally:
        _tool_environment.reset(token)


def current_tool_environment():
    return _tool_environment.get()


def resolve_gurobi_home():
    """Resolve the configured Gurobi installation."""
    selected = os.environ.get('LLM4MOR_GUROBI')
    if not selected:
        return None
    home = Path(selected).expanduser().resolve()
    if not home.is_dir() or not any((home / part).is_dir() for part in ('bin', 'lib')):
        raise ToolConfigurationError(
            'Invalid Gurobi installation directory: ' + str(home) +
            '. Set LLM4MOR_GUROBI to a directory containing bin/ or lib/.')
    return str(home)


def gurobi_environment():
    """Preserve license configuration and extend native search paths in a copy."""
    env = dict(os.environ)
    resolved = resolve_gurobi_home()
    if resolved:
        env['GUROBI_HOME'] = resolved
        home = Path(resolved)
        entries = [('PATH', home / 'bin')]
        if sys.platform.startswith('linux'):
            entries.append(('LD_LIBRARY_PATH', home / 'lib'))
        elif sys.platform == 'darwin':
            entries.append(('DYLD_LIBRARY_PATH', home / 'lib'))
        for variable, directory in entries:
            if directory.is_dir():
                values = [str(directory), *env.get(variable, '').split(os.pathsep)]
                env[variable] = os.pathsep.join(dict.fromkeys(value for value in values if value))
    # Inherit GRB_LICENSE_FILE unchanged; the native solver owns default discovery.
    return env


@contextlib.contextmanager
def solver_environment(solver=None):
    """Scope native subprocess configuration for the selected solver."""
    env = gurobi_environment() if solver == 'gurobi' else None
    with tool_environment(env):
        yield


@contextlib.contextmanager
def tool_policy(retries=2, interval=1):
    """Apply analysis-only retries to the call stack."""
    token = _tool_policy.set((retries, interval))
    try:
        yield
    finally:
        _tool_policy.reset(token)


def run_tool(*args, before_attempt=None, native=True, **kwargs):
    """Retry only timeouts and explicit temporary execution failures.

    Outside analysis, preserve subprocess.run behavior. Native diagnostics are
    classified here, before adapters turn model diagnostics into check records.
    Mapping-script diagnostics remain solution errors.
    """
    if kwargs.get('env') is None and current_tool_environment() is not None:
        kwargs['env'] = current_tool_environment()
    policy = _tool_policy.get()
    if policy is None:
        if before_attempt:
            before_attempt()
        return subprocess.run(*args, **kwargs)
    retries, interval = policy
    command = args[0] if args else kwargs.get('args')
    transient_errno = {errno.EAGAIN, errno.ENOMEM, errno.EMFILE, errno.ENFILE, errno.EBUSY}
    for attempt in range(retries + 1):
        if before_attempt:
            before_attempt()
        try:
            result = subprocess.run(*args, **kwargs)
        except subprocess.TimeoutExpired as error:
            diagnostic = f'Tool timed out after {error.timeout}s: {command}'
        except OSError as error:
            if error.errno in (errno.ENOENT, errno.EACCES, errno.ENOEXEC):
                raise ToolConfigurationError(f'Tool is unavailable or not executable: {command}: {error}') from error
            if error.errno not in transient_errno:
                raise ToolError(f'Tool execution failed: {command}: {error}') from error
            diagnostic = f'Temporary tool execution failure: {command}: {error}'
        else:
            if result.returncode == 0:
                return result
            output = '\n'.join(str(v or '') for v in (result.stdout, result.stderr))
            if native and re.search(
                    r'(?:no|missing|invalid|expired)\s+(?:(?:a|valid|AMPL|solver)\s+)*licen[cs]e\b|'
                    r'(?:cannot|could not|failed to|unable to)\s+(?:find|open|read|obtain)\s+(?:(?:a|the|valid|AMPL|solver)\s+)*licen[cs]e\b|'
                    r'licen[cs]e[^\n]{0,100}(?:expired|not found|not available|invalid|denied|limit|too (?:many|large))|'
                    r'(?:model|problem)[^\n]{0,80}(?:exceeds|too large)[^\n]{0,80}(?:demo|size|limit)|'
                    r'(?:no solver with tag|solver[^\n]{0,100}(?:not found|not available|not installed|unknown))',
                    output, re.I):
                raise ToolConfigurationError('Native tool configuration failed: ' + output.strip()[-2000:])
            if not re.search(r'resource temporarily unavailable|cannot allocate memory|out of memory|'
                             r'too many open files|temporarily unable to (?:start|fork)|'
                             r'license server[^\n]{0,80}(?:temporarily unavailable|busy)', output, re.I):
                if result.returncode < 0:
                    raise ToolError(f'Tool terminated by signal {-result.returncode}: {command}: {output[-2000:]}')
                return result
            diagnostic = 'Temporary tool resource failure: ' + output.strip()[-2000:]
        if attempt == retries:
            raise ToolRetriesExhausted(f'{diagnostic} (exhausted {retries} tool retries)')
        if interval:
            time.sleep(interval)


def available():
    result = {}
    suffixes = {}
    required = ('sources', 'prepare_input', 'original_checks', 'check_model', 'mapping_input',
                'check_mapping_run', 'configure', 'reconcile', 'skipped_original', 'prompt_options')
    for entry in sorted(pkgutil.iter_modules(__path__), key=lambda e: e.name):
        if entry.name.startswith('_') or not entry.ispkg:
            continue
        backend = importlib.import_module(__name__ + '.' + entry.name)
        language = backend.language
        if language in result:
            raise ValueError('Duplicate backend language: ' + language)
        for name in required:
            if not callable(getattr(backend, name, None)):
                raise ValueError(f'Backend {language} is missing {name}')
        for suffix in backend.model_suffixes:
            if suffix in suffixes:
                raise ValueError('Conflicting model suffix: ' + suffix)
            suffixes[suffix] = language
        result[language] = backend
    return result


def get(language):
    try:
        return available()[language]
    except KeyError:
        raise ValueError('Unsupported modeling language: ' + str(language)) from None


def for_path(path):
    name = str(path).lower()
    for backend in available().values():
        if name.endswith(backend.model_suffixes):
            return backend
    raise ValueError('Unsupported model path: ' + str(path))


def add_arguments(parser, command):
    for backend in available().values():
        add = getattr(backend, 'add_arguments', None)
        if add is not None:
            add(parser, command)


def resolve_tool(given, command, *, required=False):
    """Resolve a configured tool path or a command name on PATH."""
    requested = os.path.expanduser(given)
    resolved = shutil.which(requested)
    if resolved:
        return os.path.abspath(resolved)
    if required:
        raise ToolConfigurationError(
            f'{command} executable is unavailable or not executable: {requested}. '
            f'Set LLM4MOR_{command.upper()} by sourcing init.sh, or add {command} to PATH.')
    return os.path.abspath(requested) if os.path.dirname(requested) else requested
