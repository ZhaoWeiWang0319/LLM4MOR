#!/usr/bin/env python3
"""Reformulate one AMPL or MiniZinc model with an LLM."""
import argparse
import ast
import json
import os
import re
import sys
import time
from pathlib import Path
import backends
from backends import write_json, write_text, check_record, stem
from solution_protocol import INPUT_FIELDS, validate_input
from llm_client import LLMClient, LLMRequestError


DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / 'outputs'


def map_path(model_path):
    return stem(model_path) + '.map.py'


def check_summary(record):
    """Validate the internal check record and project its public fields."""
    if not isinstance(record, dict) or set(record) - {'instance_id', 'status', 'message', '_failure'}:
        raise ValueError('Analysis checks must use the shared check record format')
    normalized = check_record(record['instance_id'], record['status'], record.get('message'),
                              failure=record.get('_failure'))
    result = dict(status=normalized['status'])
    if 'message' in normalized:
        result['message'] = normalized['message']
    return result


def record_check(metadata, stage, record):
    """Attach one native check result to the model metadata."""
    if record['instance_id'] != metadata['instance_id']:
        raise ValueError(stage.capitalize() + ' validation used a different instance identifier')
    metadata['checks'][stage] = check_summary(record)


def build_parser():
    p = argparse.ArgumentParser(
        description="Call an LLM to reformulate one modeling file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("input", help="Path to one AMPL or MiniZinc model file")
    p.add_argument("--output-dir", default=None,
                   help="Directory for generated artifacts and logs (default: outputs/ at the repository root)")
    p.add_argument("--retry-interval", type=int, default=10,
                   help="Seconds between request retries after temporary LLM API or network failures")
    p.add_argument("--max-rounds", type=int, default=5,
                   help="Maximum request attempts after temporary LLM failures (0 means unlimited)")
    p.add_argument("--fix-rounds", type=int, default=3,
                   help="Maximum repair rounds for each of the model and mapping stages "
                        "(0 skips native original/solution checks and repairs; modeling tools are not required)")
    p.add_argument('--tool-retries', type=int, default=2,
                   help='Additional attempts for each unchanged tool invocation after a timeout or temporary resource failure')
    p.add_argument('--tool-retry-interval', type=float, default=1,
                   help='Seconds between local tool retries; independent of LLM repair and file retry budgets')
    p.add_argument("--no-map", action="store_true",
                   help="Do not require the LLM to output a solution-mapping script (.llm.map.py). "
                        "By default, .mod/.mzn files require a mapping script "
                        "for solution-mapping validation by verify_solution.py after solving")
    p.add_argument("--no-report", action="store_true",
                   help="Do not ask the LLM to generate an instance report. By default, two conversation turns after successful reformulation "
                        "generate <instance>.llm.report.md and .report.tex")

    g = p.add_argument_group("LLM API")
    g.add_argument("--base-url", default=os.environ.get("LLM_BASE_URL",
                                                        "https://api.openai.com/v1"),
                   help="API URL; can also be set with the LLM_BASE_URL environment variable")
    g.add_argument("--api-key", default=os.environ.get("LLM_API_KEY",
                                                       os.environ.get("OPENAI_API_KEY", "")),
                   help="API key; the LLM_API_KEY / OPENAI_API_KEY environment variables are recommended")
    g.add_argument("--llm-model", default=os.environ.get("LLM_MODEL", "gpt-6-astra"),
                   help="Model name; can also be set with the LLM_MODEL environment variable")
    g.add_argument("--wire-api", choices=["chat", "responses"], default="chat",
                   help="chat=OpenAI-compatible /chat/completions; "
                        "responses=GPT /responses")
    g.add_argument("--temperature", type=float, default=1)
    g.add_argument("--reasoning-effort", default=os.environ.get("LLM_REASONING_EFFORT"),
                   help="Reasoning effort for supported models; can also be set with the LLM_REASONING_EFFORT environment variable")
    g.add_argument("--max-tokens", type=int, default=None)
    g.add_argument("--llm-timeout", type=int, default=600,
                   help="Maximum seconds to wait for one conversation; a timeout marks the file as failed")
    g.add_argument("--max-chars", type=int, default=1000000,
                   help="Maximum characters in the complete request, including prompts, interface context, and conversation history")
    g.add_argument("--report-timeout", type=int, default=600,
                   help="Maximum seconds to wait for each report HTTP request; independent of --llm-timeout")
    g.add_argument("--report-max-tokens", type=int, default=None,
                   help="Maximum output tokens for each Markdown or LaTeX report request; independent of --max-tokens")
    g.add_argument("--report-max-chars", type=int, default=1000000,
                   help="Maximum characters in each complete report request; independent of --max-chars")

    backends.add_arguments(p, 'generation')
    return p


class InvalidOriginalModel(ValueError):
    """The original model failed native checks and cannot enter generation."""


class InvalidSolution(ValueError):
    """The solution is incomplete or has exhausted its repair budget."""


class RequestTooLarge(ValueError):
    """The complete conversation exceeds the configured input limit."""


def _raise_check_failure(record):
    if record.get('_failure') is not None:
        raise record['_failure']

_fence_re = re.compile(r"^[ \t]*```([^\r\n]*)\r?\n(.*?)^[ \t]*```[ \t]*(?:\r?\n|\Z)",
                       re.DOTALL | re.MULTILINE)


def output_name_for(path):
    return stem(path) + '.llm' + backends.for_path(path).model_suffixes[0]


def resolve_model_path(value):
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise ValueError('Model file does not exist: ' + str(value))
    backends.for_path(path)
    return path


def generate_reports(messages, reply, reply_path, stem, args, llm, *, language='ampl'):
    """Prepare both report documents."""
    prompts = backends.get(language).prompts
    messages.append({'role': 'assistant', 'content': reply})
    messages.append({'role': 'user', 'content': prompts.REPORT_MD_PROMPT})
    report_md = _report_request(llm, args, messages, temperature=args.temperature, max_tokens=args.report_max_tokens)
    with open(reply_path, 'a', encoding='utf-8') as stream:
        stream.write('\n\n===== Instance Report (Markdown) =====\n\n' + report_md)
    md_body = report_md.strip()
    match = re.fullmatch(r'```(?:markdown|md)?\s*\n(.*?)\n?```', md_body, re.DOTALL)
    if match:
        md_body = match[1].strip()
    if not md_body:
        raise ValueError('Markdown report is empty')
    messages.append({'role': 'assistant', 'content': report_md})
    messages.append({'role': 'user', 'content': prompts.REPORT_TEX_PROMPT})
    report_tex = _report_request(llm, args, messages, temperature=args.temperature, max_tokens=args.report_max_tokens)
    with open(reply_path, 'a', encoding='utf-8') as stream:
        stream.write('\n\n===== Instance Report (LaTeX) =====\n\n' + report_tex)
    blocks = re.findall(r'```(?:latex|tex)?\s*\n(.*?)\n?```', report_tex, re.DOTALL)
    tex_body = max(blocks, key=len).strip() if blocks else report_tex.strip()
    if not tex_body:
        raise ValueError('LaTeX report is empty')
    return md_body + '\n', tex_body + '\n'


def _llm_request(llm, args, messages, **kwargs):
    size = sum(len(message.get('content', '')) for message in messages)
    if args.max_chars and size > args.max_chars:
        raise RequestTooLarge(f'File too large (complete request: {size} characters > --max-chars {args.max_chars})')
    return llm.chat(messages, reasoning_effort=args.reasoning_effort, **kwargs)


def _report_request(llm, args, messages, **kwargs):
    size = sum(len(message.get('content', '')) for message in messages)
    if args.report_max_chars and size > args.report_max_chars:
        raise RequestTooLarge(f'Report request too large (complete request: {size} characters > --report-max-chars {args.report_max_chars})')
    return llm.chat(messages, reasoning_effort=args.reasoning_effort, **kwargs)


def _zero_check_reason(code):
    """Certify only simple zero-defined scripts; uncertainty means syntax-only."""
    tree = ast.parse(code)
    aliases = {arg.arg for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
               and 'map' in node.name for arg in node.args.args[:1]}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == 'load':
            aliases.update(t.id for t in node.targets if isinstance(t, ast.Name))
    fields = INPUT_FIELDS
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id in aliases:
            if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str) and node.slice.value not in fields:
                raise ValueError('Unknown mapping input field: ' + node.slice.value)
    defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    safe_calls = {'open', 'dict', 'list', 'tuple', 'len', 'range', 'enumerate', 'zip', 'sum',
                  'float', 'int', 'bool', 'str', 'abs', 'print', 'isinstance',
                  'Exception', 'RuntimeError', 'ValueError', 'TypeError', 'KeyError', 'IndexError'} | defined
    safe_methods = {'load', 'dump', 'loads', 'dumps', 'items', 'keys', 'values', 'get', 'copy',
                    'append', 'extend', 'update', 'startswith', 'endswith', 'split', 'join', 'replace'}
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)):
            constant = node.right.value if isinstance(node.right, ast.Constant) else None
            if not isinstance(constant, (int, float)) or isinstance(constant, bool) or (
                    constant < 0 if isinstance(node.op, ast.Pow) else constant == 0):
                return 'Mapping arithmetic may require nonzero or restricted-domain inputs'
        if isinstance(node, (ast.If, ast.IfExp)) and not isinstance(node.test, ast.Constant):
            names = {n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)}
            if names != {'__name__'}:
                return 'Conditional reconstruction preconditions are unconfirmed for zero input'
        if isinstance(node, ast.comprehension) and node.ifs:
            return 'Filtered reconstruction may require selected or nonzero decision values'
        if isinstance(node, (ast.Assert, ast.While)):
            return 'Mapping input preconditions cannot be confirmed for an artificial zero solution'
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id not in safe_calls:
                return 'Zero applicability is unconfirmed for call: ' + node.func.id
            if isinstance(node.func, ast.Attribute) and node.func.attr not in safe_methods:
                return 'Zero applicability is unconfirmed for method: ' + node.func.attr
    return None


def _mapping_checks(code, instance_id, prepare, run):
    """Keep syntax, applicability, preparation failures, and runtime failures distinct."""
    try:
        compile(code, '<map.py>', 'exec')
        reason = _zero_check_reason(code)
    except (SyntaxError, ValueError, TypeError) as error:
        return check_record(instance_id, 'failed', str(error))
    record = check_record(instance_id, 'syntax_only', reason)
    if reason is None:
        try:
            probe = prepare()
            if isinstance(probe, backends.MappingProbeInput):
                validate_input(probe.payload)
                if probe.payload['instance_id'] != instance_id:
                    raise ValueError('Mapping probe must identify the current instance')
            elif not isinstance(probe, backends.MappingProbeUnavailable):
                raise TypeError('mapping_input must return MappingProbeInput or MappingProbeUnavailable')
        except (backends.ToolError, backends.InputPreparationError) as error:
            record = check_record(instance_id, 'failed', failure=error)
        except (ValueError, OSError) as error:
            failure = backends.InputPreparationError('Mapping input preparation failed: ' + str(error))
            record = check_record(instance_id, 'failed', failure=failure)
        except Exception as error:
            record = check_record(instance_id, 'failed', failure=error)
        else:
            if isinstance(probe, backends.MappingProbeUnavailable):
                record = check_record(instance_id, 'syntax_only', probe.reason)
            else:
                try:
                    run(code, probe.payload)
                    record = check_record(instance_id, 'passed')
                except backends.MappingExecutionError as error:
                    record = check_record(instance_id, 'failed', str(error))
                except Exception as error:
                    record = check_record(instance_id, 'failed', failure=error)
    return record


def _repair_solution(reply, messages, reply_path, args, llm, want_map, prompts,
                      model_check, mapping_check, metadata, checkpoint=None):
    """Use independent finite repair budgets while replacing the complete solution."""
    checkpoint = checkpoint or (lambda stage: None)
    def parts(text):
        model, mapping = extract_parts(text)
        return model if model and model.strip() else None, mapping if mapping and mapping.strip() else None

    def incomplete(stage):
        return check_record(metadata['instance_id'], 'failed',
            'Invalid reply: expected exactly one nonempty ```' + stage + ' code block')

    model, mapping = parts(reply)
    metadata['repairs'] = dict(model=0, mapping=0)
    def skipped(required):
        return check_record(metadata['instance_id'], 'skipped',
                            'fix_rounds=0' if required else 'mapping disabled')
    if args.fix_rounds == 0:
        if model is None or (want_map and mapping is None):
            raise InvalidSolution('Invalid reply: expected exactly one nonempty ```model block'
                                   + (' and one nonempty ```mapping block' if want_map else ''))
        record_check(metadata, 'model', skipped(True))
        record_check(metadata, 'mapping', skipped(want_map))
        checkpoint('validation')
        return model, mapping, reply
    need_model = True
    while True:
        if need_model:
            checkpoint('model_check')
            record = model_check(model) if model is not None else incomplete('model')
            stage = 'model'
        else:
            checkpoint('mapping_check')
            record = ((mapping_check(model, mapping) if mapping is not None else incomplete('mapping'))
                      if want_map else skipped(False))
            stage = 'mapping'
        record_check(metadata, stage, record)
        check = metadata['checks'][stage]
        checkpoint(stage + '_check')
        _raise_check_failure(record)
        if check['status'] != 'failed':
            if need_model:
                need_model = False
                continue
            return model, mapping, reply
        detail = metadata['instance_id'] + ': ' + check['message']
        if metadata['repairs'][stage] >= args.fix_rounds:
            raise InvalidSolution(stage.capitalize() + ' validation failed after repair budget: ' + detail)
        metadata['repairs'][stage] += 1
        checkpoint(stage + '_repair')
        instruction = prompts.repair_instruction(stage, detail)
        messages += [dict(role='assistant', content=reply), dict(role='user', content=instruction)]
        reply = _llm_request(llm, args, messages, temperature=args.temperature, max_tokens=args.max_tokens)
        with open(reply_path, 'a', encoding='utf-8') as stream:
            stream.write(f'\n\n===== {stage.capitalize()} repair round {metadata["repairs"][stage]} =====\n' + reply)
        new_model, mapping = parts(reply)
        need_model = stage == 'model' or new_model != model
        model = new_model
        if need_model:
            metadata['checks'].pop('mapping', None)


def _publish_solution(out_path, model, mapping, metadata, started, want_map):
    """Write the complete model bundle before writing reports."""
    if want_map:
        write_text(map_path(out_path), mapping)
    write_text(out_path, model)
    seconds = round(time.monotonic() - started, 2)
    metadata = dict(metadata, seconds=seconds, status='success', stage='complete',
                    mapping_required=bool(want_map))
    write_json(stem(out_path) + '.meta.json', metadata)
    return seconds


def _run_reports(messages, reply, reply_path, stem, args, llm, *, language='ampl'):
    """Write reports, keeping report failures separate from the model."""
    started = time.monotonic()
    metadata_path = stem + '.report.meta.json'

    def save(status, message=None):
        record = dict(status=status, seconds=round(time.monotonic()-started, 2))
        if message:
            record['message'] = message
        write_json(metadata_path, record)
        return record

    try:
        if args.no_report:
            return save('skipped')
        save('processing')
        report_llm = LLMClient(llm.base_url, llm.api_key, llm.model,
                               timeout=args.report_timeout, wire_api=llm.wire_api,
                               timeout_option='--report-timeout')
        markdown, latex = generate_reports(messages, reply, reply_path, stem, args, report_llm, language=language)
        write_text(stem + '.report.md', markdown)
        write_text(stem + '.report.tex', latex)
        return save('success')
    except (Exception, KeyboardInterrupt) as error:
        interrupted = isinstance(error, KeyboardInterrupt)
        report = dict(status='interrupted' if interrupted else 'failed',
                      message='Interrupted' if interrupted else str(error),
                      seconds=round(time.monotonic()-started, 2))
        print('# !! Report processing ' + ('interrupted' if interrupted else 'failed')
              + ': ' + report['message'], file=sys.stderr)
        try:
            write_json(metadata_path, report)
        except OSError as write_error:
            print('# !! Could not record report status: ' + str(write_error), file=sys.stderr)
        if interrupted:
            raise
        return report


def extract_parts(llm_text):
    """Extract explicit model/mapping roles without inferring a source language.

    Missing, empty, or duplicate role blocks return None for that role so the
    shared repair loop can request a complete, unambiguous replacement.
    Other fenced blocks are ignored, including language-only labels.
    """
    blocks = {'model': [], 'mapping': []}
    for label, code in _fence_re.findall(llm_text):
        role = label.strip().lower()
        if role in blocks:
            blocks[role].append(code.removesuffix('\n').removesuffix('\r'))
    def unique(role):
        solutions = blocks[role]
        return solutions[0] if len(solutions) == 1 and solutions[0].strip() else None
    return unique('model'), unique('mapping')


def retryable_error(error):
    """Only provider/transport failures may restart the request."""
    return isinstance(error, LLMRequestError) and error.retryable


def run(args):
    parser = argparse.ArgumentParser()
    if (args.fix_rounds < 0 or args.max_rounds < 0 or args.retry_interval < 0
            or args.tool_retries < 0 or not 0 <= args.tool_retry_interval < float('inf')):
        parser.error('repair/retry rounds and intervals cannot be negative')
    args.model_path = resolve_model_path(args.input)
    args.instance_id = stem(args.model_path.name)
    backend = backends.for_path(args.model_path)
    args.output_dir = str(Path(args.output_dir).expanduser().resolve()) if args.output_dir else str(DEFAULT_OUTPUT_ROOT)
    if args.output_dir == str(args.model_path.parent):
        parser.error('The output directory must differ from the input model directory')
    args.output_path = os.path.join(args.output_dir, output_name_for(args.model_path.name))
    args.logs_dir = os.path.join(args.output_dir, 'logs')

    if not args.api_key:
        sys.exit("Error: no API key provided (environment variables LLM_API_KEY / OPENAI_API_KEY)")

    llm = LLMClient(args.base_url, args.api_key, args.llm_model,
                    timeout=args.llm_timeout, wire_api=args.wire_api)

    try:
        reconcile_backend_settings(args, backend)
        with backends.tool_policy(args.tool_retries, args.tool_retry_interval):
            backend.configure(args, 'generation')
    except (ValueError, backends.ToolError) as error:
        parser.error(str(error))

    print(f"# Model file: {args.model_path}", file=sys.stderr)
    round_no = 0
    while True:
        round_no += 1
        if round_no > 1:
            limit = f"/{args.max_rounds}" if args.max_rounds > 0 else ""
            print(f"# Round {round_no}{limit}: retrying after {args.retry_interval}s ...", file=sys.stderr)
            time.sleep(args.retry_interval)
        result = process_one_file(args, llm)
        if result['ok']:
            print(f"# Round {round_no} finished: 1/1 succeeded", file=sys.stderr)
            print("# Complete: 1/1 succeeded", file=sys.stderr)
            return 0
        retryable = result.get('retryable', False)
        exhausted = retryable and args.max_rounds > 0 and round_no >= args.max_rounds
        if exhausted:
            print(f"# Maximum retry rounds reached (--max-rounds {args.max_rounds})", file=sys.stderr)
        print(f"# Round {round_no} finished: 0/1 succeeded", file=sys.stderr)
        if not retryable or exhausted:
            print("# Complete: 0/1 succeeded", file=sys.stderr)
            print("# Unsuccessful (no more retries): " + result['file'], file=sys.stderr)
            return 1


def reconcile_backend_settings(args, backend):
    backend.reconcile(args)


def process_one_file(args, llm):
    with backends.tool_policy(getattr(args, 'tool_retries', 2), getattr(args, 'tool_retry_interval', 1)):
        return _process_one_file(args, llm)


def _process_one_file(args, llm):
    started = time.monotonic()
    result = dict(file=args.model_path.name, ok=False, output=None, error=None, analysis=None, seconds=None)
    metadata, out_path, published = None, args.output_path, False
    failure = None

    def save_metadata(stage=None, **updates):
        if stage is not None:
            metadata['stage'] = stage
        metadata.update(updates, seconds=round(time.monotonic() - started, 2))
        write_json(stem(out_path) + '.meta.json', metadata)

    try:
        backend = backends.for_path(args.model_path)
        source, context, args.has_external_data = backend.prepare_input(args)
        want_map = not args.no_map
        metadata = dict(model_id=stem(args.model_path.name), source_model=str(args.model_path),
            instance_id=args.instance_id, language=backend.language, checks={},
            repairs=dict(model=0, mapping=0), mapping_required=want_map, status='processing')
        save_metadata('original_check')
        baseline = backend.original_checks(args) if args.fix_rounds > 0 else backend.skipped_original(args)
        record_check(metadata, 'original', baseline)
        save_metadata()
        _raise_check_failure(baseline)
        if args.fix_rounds > 0 and metadata['checks']['original']['status'] == 'failed':
            raise InvalidOriginalModel('Original model preflight failed: ' +
                metadata['checks']['original'].get('message', 'Original check failed'))
        save_metadata('input_preparation')
        content = backend.prompts.reformulation_request(args.model_path.name, source, context, want_map=want_map,
            has_external_data=args.has_external_data, language_variables=backend.prompt_options(args))
        messages = [dict(role='system', content=backend.prompts.SYSTEM_PROMPT), dict(role='user', content=content)]
        reply_path = stem(out_path) + '.reply.txt'
        save_metadata('generation')
        reply = _llm_request(llm, args, messages, temperature=args.temperature, max_tokens=args.max_tokens)
        write_text(reply_path, reply)
        def check_model(model):
            return backend.check_model(model, args)
        def check_mapping(model, mapping):
            return _mapping_checks(mapping, args.instance_id,
                lambda: backend.mapping_input(model, args), backend.check_mapping_run)
        save_metadata('validation')
        model, mapping, reply = _repair_solution(reply, messages, reply_path, args, llm, want_map,
            backend.prompts, check_model, check_mapping, metadata, checkpoint=save_metadata)
        save_metadata('publication')
        seconds = _publish_solution(out_path, model, mapping, metadata, started, want_map)
        published = True
        analysis = reply.split('```', 1)[0].strip()
        result.update(ok=True, output=out_path, analysis=analysis[:500], seconds=seconds)
        result['report'] = _run_reports(messages, reply, reply_path, stem(out_path), args, llm, language=backend.language)
    except Exception as error:
        failure = error
        result['error'] = str(error)
        result['retryable'] = retryable_error(error)
    except KeyboardInterrupt:
        result['error'] = 'Interrupted'
        raise
    finally:
        if metadata is not None and not published:
            try:
                message = result['error']
                if message and (isinstance(failure, InvalidOriginalModel)
                                or (isinstance(failure, InvalidSolution) and metadata['stage'].endswith('_check'))):
                    message = message.split(':', 1)[0]
                save_metadata(status='failed' if result['error'] != 'Interrupted' else 'interrupted',
                              message=message)
            except OSError as error:
                print('# !! Could not record generation status: ' + str(error), file=sys.stderr)
        if result['seconds'] is None:
            result['seconds'] = round(time.monotonic() - started, 2)
        summary = dict(model_id=stem(args.model_path.name), status='success' if result['ok'] else 'failed',
                       seconds=result['seconds'])
        if out_path:
            summary['meta_file'] = stem(out_path) + '.meta.json'
        if result['error']:
            summary['message'] = (metadata.get('message', result['error'])
                                  if metadata is not None else result['error'])
            if result['error'] == 'Interrupted':
                summary['status'] = 'interrupted'
        line = json.dumps(summary, ensure_ascii=False)
        print(line, flush=True)
        if getattr(args, 'logs_dir', None):
            os.makedirs(args.logs_dir, exist_ok=True)
            with open(os.path.join(args.logs_dir, 'reformulation.log'), 'a', encoding='utf-8') as stream:
                stream.write(line + '\n')
    return result


def main():
    args = build_parser().parse_args()
    status = 1
    try:
        status = run(args)
    except SystemExit as error:
        if isinstance(error.code, int):
            status = error.code
        else:
            print(error.code, file=sys.stderr)
    except KeyboardInterrupt:
        status = 130
    except Exception as error:
        print(f'# Analysis exception: {error}', file=sys.stderr)
    sys.exit(status)


if __name__ == "__main__":
    main()
