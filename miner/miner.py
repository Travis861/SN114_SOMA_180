"""SOMA Chain-of-Thought Compression miner for SWE agent transcripts.

Round 1 tasks are real coding-agent sessions. The most useful compressed form is
not a generic summary; it is a logical trace: user problem, key reasoning, tool
actions/results that changed state, code edits, verification, and final answer.
"""

from __future__ import annotations

import html
import json
import re

try:
    import tiktoken
except ModuleNotFoundError:
    tiktoken = None


ENCODER = tiktoken.get_encoding("cl100k_base") if tiktoken is not None else None
TOKEN_LIMIT_SAFETY_FACTOR = 0.975

EVENT_RE = re.compile(
    r'<message\s+role=["\'](?P<role>[^"\']+)["\'][^>]*>(?P<msg>.*?)</message>|'
    r'<(?:tool_result|tool_response|tool_output|function_result|function_response|'
    r'observation|command_result|exec_result|stdout|stderr)[^>]*>(?P<result>.*?)'
    r'</(?:tool_result|tool_response|tool_output|function_result|function_response|'
    r'observation|command_result|exec_result|stdout|stderr)>',
    re.S,
)
TOOL_CALL_RE = re.compile(
    r'<(?:tool_call|tool_use|function_call|function|action|command)\s+name=["\'](?P<name>[^"\']+)["\'][^>]*>'
    r'(?P<body>.*?)</(?:tool_call|tool_use|function_call|function|action|command)>',
    re.S,
)
LOOSE_FUNCTION_RE = re.compile(r"<function=(?P<name>[^>\s]+)>(?P<body>.*?)</function>", re.S)
PARAMETER_RE = re.compile(r"<parameter=(?P<key>[^>\s]+)>(?P<value>.*?)</parameter>", re.S)
TAG_RE = re.compile(r"</?(?:text|thinking|thought|reasoning|content)[^>]*>")
STEP_MARKER_RE = re.compile(
    r"(?i)(?:^|(?<=\s))(?P<label>"
    r"sample\s+\d+|step\s+\d+|problem understanding|understanding the problem|"
    r"defining variables|known quantities|given values|formulating the equation|"
    r"applying [^:]{1,40}|verification|verify|check|edge cases?|final answer|"
    r"conclusion|result|answer"
    r")\s*[:：]"
)
QA_MARKER_RE = re.compile(
    r"(?i)(?:^|(?<=\s))(?P<label>"
    r"(?:question|query|answer|response|passage|context|article|document|background|source|evidence)\s*\d*"
    r")\s*[:：]"
)
WORD_TOKEN_RE = re.compile(r"\s+|[A-Za-z0-9_]+|[^\sA-Za-z0-9_]")
TEXT_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_'-]*|\d+(?:[.,:/-]\d+)*|[\u4e00-\u9fff]+")
PROSE_SENTENCE_RE = re.compile(r"[^.!?\n。！？]+(?:[.!?。！？]+|$)")
JSON_ESCAPE_RE = re.compile(r'\\(?:u[0-9a-fA-F]{4}|U[0-9a-fA-F]{8}|x[0-9a-fA-F]{2}|["\\/bfnrt])')
CODE_SIGNAL_RE = re.compile(
    r"\b(?:import|export|from|require|module\.exports|interface|type|enum|"
    r"process\.env|const\s+[A-Z0-9_]+|describe|it|test|expect|beforeEach|afterEach|"
    r"app\.(?:get|post|put|patch|delete)|router\.(?:get|post|put|patch|delete)|"
    r"useQuery|useMutation|fetch|axios|schema|z\.object|yup|validation)\b|"
    r'"[A-Za-z0-9_.@/-]+"\s*:',
    re.I,
)
PATH_RE = re.compile(r"(?<![\w@])/(?:[\w.-]+/){2,}[\w.@+-]+")
GENERIC_LABEL_RE = re.compile(
    r"^\s*(?:[-*]\s*)?(?P<label>[\w\u00C0-\uFFFF _-]{1,40})\s*[:：]\s*(?P<body>.*)$",
    re.I,
)
NOISE_LINE_RE = re.compile(
    r"^\s*(?:[#=*>._~\-/\\|]{8,}|[.\s]*\d+%[.\s]*|"
    r"Collecting |Downloading |Installing collected|Requirement already satisfied|"
    r"Using cached |Preparing metadata|Building wheel|Successfully installed)",
    re.I,
)
SIGNAL_RE = re.compile(
    r"error|exception|traceback|fail|failed|warning|actual|expected|bug|issue|"
    r"fix|fixed|change|changed|replace|replaced|edit|success|successful|test|"
    r"verify|verified|responsible|because|problem|root cause|missing|invalid|"
    r"assert|assertion|regression|solution|constraint|edge case|equation|formula|"
    r"import|function|class|method|line|file|path|NameError|AssertionError",
    re.I,
)

FINAL_HINT_RE = re.compile(
    r"fix|fixed|responsible|issue|bug|problem|because|caused|verification|"
    r"confirmed|changed|added|removed|updated|implemented|passed|result|"
    r"therefore|conclusion|final answer|root cause|file|function|method|"
    r"corrig|arregl|répar|beheb|исправ|修复|修正|수정|修正",
    re.I,
)

TEXT_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can", "could",
    "did", "do", "does", "for", "from", "had", "has", "have", "he", "her", "his",
    "i", "if", "in", "into", "is", "it", "its", "may", "might", "not", "of", "on",
    "or", "our", "she", "should", "so", "than", "that", "the", "their", "them",
    "then", "there", "these", "they", "this", "those", "to", "was", "we", "were",
    "what", "when", "where", "which", "while", "who", "will", "with", "would",
    "you", "your", "also", "any", "each", "etc", "into", "more", "most", "such",
    "using", "via",
}

PROSE_CUE_RE = re.compile(
    r"\b(above|according|after|although|because|before|cause|caused|condition|"
    r"conclusion|constraint|contains|defined|despite|during|edge case|especially|"
    r"except|finally|first|given|however|important|include|includes|instead|key|"
    r"last|main|maximum|minimum|must|only|problem|reason|represents|result|"
    r"question|answer|passage|context|source|evidence|second|therefore|total|"
    r"unless|until|verified|whereas|where|without)\b",
    re.I,
)

KEY_QUALIFIER_RE = re.compile(
    r"\b("
    r"\d+(?:[.,:/-]\d+)*|"
    r"before|after|during|since|until|between|from|to|"
    r"at least|at most|more than|less than|no more than|no less than|"
    r"minimum|maximum|first|last|only|except|unless|without|"
    r"if|when|where|condition|constraint|require|required|requires|must|should"
    r")\b",
    re.I,
)

NEGATION_RE = re.compile(
    r"\b("
    r"not|no|never|none|neither|nor|cannot|can't|won't|doesn't|didn't|"
    r"wasn't|weren't|isn't|aren't|failed|unable|refused|denied|except|without|unless"
    r")\b",
    re.I,
)

HARD_NEGATION_RE = re.compile(
    r"\b(not|never|none|neither|nor|cannot|can't|won't|doesn't|didn't|"
    r"wasn't|weren't|isn't|aren't|failed|unable|refused|denied)\b",
    re.I,
)

LABEL_ALIASES = {
    "user": "user", "human": "user", "prompt": "user", "problem": "user", "issue": "user",
    "question": "user", "query": "user", "task": "user", "setting": "user",
    "problem_statement": "user", "title": "user", "description": "user",
    "instruction": "user", "instructions": "user", "input_request": "user", "request": "user",
    "usuario": "user", "usuário": "user", "utilisateur": "user", "benutzer": "user",
    "пользователь": "user", "用户": "user", "使用者": "user", "ユーザー": "user", "사용자": "user",
    "assistant": "assistant", "agent": "assistant", "ai": "assistant", "bot": "assistant",
    "model": "assistant", "assistant_message": "assistant", "asistente": "assistant", "assistente": "assistant",
    "assistent": "assistant", "ассистент": "assistant", "助手": "assistant", "アシスタント": "assistant",
    "어시스턴트": "assistant",
    "thought": "thought", "thinking": "thought", "reasoning": "thought", "analysis": "thought",
    "plan": "thought", "discussion": "thought", "reflection": "thought", "scratchpad": "thought", "chain_of_thought": "thought",
    "pensamiento": "thought", "razonamiento": "thought", "pensée": "thought", "raisonnement": "thought",
    "gedanke": "thought", "überlegung": "thought", "мысль": "thought", "рассуждение": "thought",
    "思考": "thought", "推理": "thought", "思考过程": "thought", "考え": "thought", "推論": "thought",
    "생각": "thought", "추론": "thought",
    "action": "action", "action_input": "action", "tool": "action", "tool_input": "action",
    "tool_call": "action", "tool_use": "action", "function_call": "action", "function": "action",
    "command": "action", "commands": "action", "cmd": "action", "exec": "action", "execute": "action", "run": "action",
    "bash": "action", "shell": "action", "terminal": "action", "edit": "action", "patch": "action",
    "change": "action", "acción": "action", "accion": "action",
    "ação": "action", "commande": "action", "befehl": "action", "aktion": "action",
    "действие": "action", "команда": "action", "行动": "action", "动作": "action", "命令": "action",
    "アクション": "action", "コマンド": "action", "행동": "action", "명령": "action",
    "observation": "result", "observed": "result", "expected": "result", "actual": "result",
    "environment_context": "result", "success_criteria": "result", "edge_cases_and_risks": "result",
    "result": "result", "resultado": "result",
    "passage": "result", "context": "result", "article": "result", "document": "result",
    "background": "result", "source": "result", "evidence": "result",
    "tool_result": "result", "tool_response": "result", "tool_output": "result",
    "function_result": "result", "function_response": "result", "command_result": "result",
    "exec_result": "result", "output": "result", "log": "result", "logs": "result",
    "error": "result", "stderr": "result", "stdout": "result", "observación": "result",
    "observacao": "result", "observação": "result", "résultat": "result", "ergebnis": "result",
    "fehler": "result", "ошибка": "result", "результат": "result", "观察": "result", "观测": "result",
    "结果": "result", "エラー": "result", "観察": "result", "結果": "result", "오류": "result",
    "관찰": "result", "결과": "result",
    "final": "final", "answer": "final", "response": "final", "solution": "final",
    "summary": "final", "conclusion": "final", "response_format": "final", "final_answer": "final", "respuesta": "final", "resposta": "final",
    "réponse": "final", "zusammenfassung": "final", "antwort": "final", "ответ": "final",
    "答案": "final", "最终": "final", "摘要": "final", "回答": "final", "最終": "final",
    "요약": "final", "답변": "final", "최종": "final",
}


def token_count(text: str) -> int:
    if not text:
        return 0
    if ENCODER is not None:
        return len(ENCODER.encode_ordinary(text))
    return len(WORD_TOKEN_RE.findall(text))


def effective_token_limit(source_tokens: int, ratio: float) -> int:
    nominal_limit = max(1, int(source_tokens * ratio))
    return max(1, int(nominal_limit * TOKEN_LIMIT_SAFETY_FACTOR))


def main(task: str, compression_ratio: float | None = None) -> str:
    if not isinstance(task, str) or not task:
        return ""
    ratio = 0.2 if compression_ratio is None else float(compression_ratio)
    ratio = max(0.01, min(1.0, ratio))
    limit = effective_token_limit(token_count(task), ratio)
    compressed = compress_cot_trace(task, limit, ratio)
    if compressed:
        return compressed
    return trim_to_tokens(clean_text(alias_paths(scrub_noise(task))), limit)


def competition_tier(ratio: float) -> str:
    """Map current validator ratios to explicit compression modes."""
    if ratio <= 0.10:
        return "ultra"
    if ratio <= 0.15:
        return "tight"
    if ratio <= 0.20:
        return "compact"
    if ratio <= 0.25:
        return "balanced"
    return "rich"


def compress_cot_trace(text: str, token_limit: int, ratio: float = 0.2) -> str:
    stripped = text.lstrip()
    text = alias_paths(text if stripped.startswith(("{", "[")) else scrub_noise(text))
    events = build_events(text)
    if not events:
        events = build_fallback_cot_events(text)
    tier = competition_tier(ratio)
    if not has_trace_structure(events):
        if has_stepwise_structure(events):
            if ratio > 0.60:
                return compress_high_fidelity_trace(text, events, token_limit)
            return compress_schema_trace(events, token_limit, tier=tier)
        if has_dialogue_structure(events):
            return compress_schema_trace(events, token_limit, tier=tier)
        events = build_fallback_cot_events(text)
        return compress_schema_trace(events, token_limit, tier=tier)

    if ratio <= 0.25:
        return compress_schema_trace(events, token_limit, tier=tier)
    if ratio > 0.60:
        return compress_high_fidelity_trace(text, events, token_limit)

    safety_margin = max(24, min(96, token_limit // 30))
    ordered = select_balanced_trace(events, max(1, token_limit - safety_margin), tier="balanced")
    repaired = repair_transcript_tags(ordered)
    return trim_to_tokens(repaired, token_limit)


def compress_high_fidelity_trace(
    cleaned_text: str,
    events: list[tuple[int, str, str]],
    token_limit: int,
) -> str:
    if token_count(cleaned_text) <= token_limit:
        return repair_transcript_tags(cleaned_text)

    safety_margin = max(32, min(128, token_limit // 40))
    ordered = select_balanced_trace(events, max(1, token_limit - safety_margin), tier="rich")
    repaired = repair_transcript_tags(ordered)
    return trim_to_tokens(repaired, token_limit)


def has_trace_structure(events: list[tuple[int, str, str]]) -> bool:
    labels = {label.split()[0] for _weight, label, _body in events}
    # A real agent trace needs at least one action/tool step. Inputs without
    # that structure are still compressed through the CoT schema path.
    return "ACTION" in labels and bool(labels & {"PROBLEM", "REASON", "RESULT", "FINAL"})


def has_stepwise_structure(events: list[tuple[int, str, str]]) -> bool:
    labels = [label.split()[0] for _weight, label, _body in events]
    return (
        labels.count("REASON") >= 2
        and bool(set(labels) & {"PROBLEM", "RESULT", "FINAL"})
        and "ACTION" not in labels
    )


def has_dialogue_structure(events: list[tuple[int, str, str]]) -> bool:
    labels = [label.split()[0] for _weight, label, _body in events]
    return (
        "PROBLEM" in labels
        and "FINAL" in labels
        and "ACTION" not in labels
        and "RESULT" not in labels
        and len(events) <= 8
    )


def build_fallback_cot_events(text: str) -> list[tuple[int, str, str]]:
    cleaned = clean_plain_text(text)
    if not cleaned:
        return [(100, "PROBLEM", "Empty input")]

    units = [unit for _paragraph, unit in split_prose_units(cleaned)]
    if len(units) < 2:
        units = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if not units:
        units = [cleaned]

    events: list[tuple[int, str, str]] = [(100, "PROBLEM", one_line(cleaned, 1200))]
    scored: list[tuple[int, int, str]] = []
    for idx, unit in enumerate(units[:240]):
        score = 48
        if idx < 3:
            score += 18
        if idx >= max(0, len(units) - 3):
            score += 8
        if SIGNAL_RE.search(unit):
            score += 20
        if FINAL_HINT_RE.search(unit):
            score += 16
        if KEY_QUALIFIER_RE.search(unit):
            score += 14
        if NEGATION_RE.search(unit):
            score += 16
        if looks_like_path_or_code(unit):
            score += 12
        if re.search(r"\d|[$€£%]", unit):
            score += 10
        if 40 <= len(unit) <= 360:
            score += 5
        elif len(unit) > 700:
            score -= 8
        scored.append((score, idx, unit))

    selected = sorted(sorted(scored, reverse=True)[:32], key=lambda item: item[1])
    result_count = 0
    reason_count = 0
    for score, _idx, unit in selected:
        if SIGNAL_RE.search(unit) or FINAL_HINT_RE.search(unit):
            reason_count += 1
            events.append((min(95, score), f"REASON {reason_count}", unit))
        else:
            result_count += 1
            events.append((min(90, score), f"RESULT {result_count}", unit))

    if units:
        events.append((98, "FINAL", one_line(units[-1], 900)))
    return dedupe_events(events)


def is_code_file_structure(events: list[tuple[int, str, str]]) -> bool:
    problem = first_body(events, "PROBLEM")
    labels = [label.split()[0] for _weight, label, _body in events]
    return problem.startswith("File: ") and "RESULT" in labels


def compress_code_file_trace(events: list[tuple[int, str, str]], token_limit: int) -> str:
    problem = first_body(events, "PROBLEM")
    code = first_result_body(events)
    prefix = problem + "\nCode:\n"
    if token_limit <= 0:
        return ""
    budget = token_limit - token_count(prefix)
    if budget <= 8:
        return trim_to_tokens(problem + "\n" + code, token_limit)
    return trim_to_tokens(prefix + trim_to_tokens(code, budget), token_limit)


def compress_plain_text(text: str, token_limit: int) -> str:
    text = clean_plain_text(text)
    if token_count(text) <= token_limit:
        return text
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if not lines:
        return trim_to_tokens(text, token_limit)
    if is_probably_prose(text, lines):
        return compress_prose_text(text, token_limit)

    return compress_line_text(lines, token_limit)


def compress_line_text(lines: list[str], token_limit: int) -> str:
    code_dominant = is_code_dominant(lines)
    scored = []
    for idx, line in enumerate(lines):
        score = 0
        stripped = line.strip()
        if idx < 8:
            score += 8
        if re.search(r"^(#|//|/\*|\*)", stripped):
            score += 8
        if looks_like_path_or_code(stripped):
            score += 7
        if SIGNAL_RE.search(stripped):
            score += 9
        if code_dominant and CODE_SIGNAL_RE.search(stripped):
            score += 8
        if KEY_QUALIFIER_RE.search(stripped) or NEGATION_RE.search(stripped):
            score += 6
        if re.search(r"\b(main|class|def|function|func|fn|struct|interface|return|raise|throw|assert)\b", stripped):
            score += 5
        if len(stripped) > 180:
            score -= 2
        scored.append((score, idx, stripped))

    selected: set[int] = set()
    used = 0
    output: list[str] = []

    def add(idx: int, line: str) -> None:
        nonlocal used
        if idx in selected or used >= token_limit:
            return
        cost = token_count(line + "\n")
        if used + cost <= token_limit:
            selected.add(idx)
            output.append(line)
            used += cost
        elif token_limit - used > 20:
            output.append(trim_to_tokens(line, token_limit - used))
            used = token_limit
            selected.add(idx)

    for score, idx, line in scored[:8]:
        add(idx, line)
    for score, idx, line in sorted(scored, reverse=True):
        if score > 0:
            add(idx, line)
    for score, idx, line in scored:
        add(idx, line)

    ordered = sorted(output, key=lambda line: lines.index(line) if line in lines else 10**9)
    rendered = "\n".join(ordered)
    if token_count(rendered) < max(1, int(token_limit * 0.6)):
        return trim_to_tokens("\n".join(lines), token_limit)
    return trim_to_tokens(rendered, token_limit)


def is_code_dominant(lines: list[str]) -> bool:
    if not lines:
        return False
    code_hits = 0
    for line in lines:
        stripped = line.strip()
        if looks_like_path_or_code(stripped) or CODE_SIGNAL_RE.search(stripped) or re.search(r"[{};=<>]|\b(?:npm|pytest|cargo|go test)\b", stripped):
            code_hits += 1
    return code_hits / max(1, len(lines)) >= 0.35


def is_probably_prose(text: str, lines: list[str]) -> bool:
    if not lines:
        return False
    line_count = len(lines)
    code_like = 0
    prose_like = 0
    for line in lines:
        stripped = line.strip()
        words = TEXT_WORD_RE.findall(stripped)
        if looks_like_path_or_code(stripped) or re.search(r"[{};<>]|^\s*(?:#include|import |from |require\()", stripped):
            code_like += 1
        if len(words) >= 7 or re.search(r"[.!?。！？]$", stripped):
            prose_like += 1
    sentence_marks = len(re.findall(r"[.!?。！？]", text))
    avg_words = sum(len(TEXT_WORD_RE.findall(line)) for line in lines) / max(1, line_count)
    required_prose_lines = 1 if line_count <= 2 else max(2, line_count // 4)
    return (
        sentence_marks >= 3
        and avg_words >= 5
        and prose_like >= required_prose_lines
        and code_like / max(1, line_count) < 0.35
    )


def compress_prose_text(text: str, token_limit: int) -> str:
    units = split_prose_units(text)
    if not units:
        return trim_to_tokens(text, token_limit)

    frequencies = prose_frequencies([unit for _paragraph, unit in units])
    candidates = []
    total = len(units)
    for idx, (paragraph_idx, unit) in enumerate(units):
        score = prose_unit_score(unit, idx, total, paragraph_idx, frequencies)
        cost = max(1, token_count(unit + " "))
        candidates.append(
            {
                "idx": idx,
                "paragraph": paragraph_idx,
                "unit": unit,
                "score": score,
                "cost": cost,
                "density": score / cost,
            }
        )

    selected: set[int] = set()
    used = 0
    target_floor = int(token_limit * 0.94)

    def add(candidate: dict, force: bool = False) -> None:
        nonlocal used
        idx = int(candidate["idx"])
        if idx in selected or used >= token_limit:
            return
        unit = str(candidate["unit"])
        cost = int(candidate["cost"])
        if used + cost <= token_limit:
            selected.add(idx)
            used += cost
        elif force and token_limit - used > 20:
            selected.add(idx)
            used = token_limit

    # Lead sentences usually carry topic and framing, but under tight budgets a
    # generic opener should not crowd out dates, counts, conditions, or negation.
    for candidate in candidates[:1]:
        if token_limit >= 80 or float(candidate["score"]) >= 32:
            add(candidate)

    for candidate in candidates:
        if looks_like_heading(str(candidate["unit"])):
            add(candidate)

    for candidate in sorted(candidates, key=lambda item: (item["score"], item["density"]), reverse=True):
        if used >= target_floor:
            break
        add(candidate)

    # Fill remaining space chronologically, preserving qualifiers that may be
    # asked about later even if their global score was not high.
    for candidate in candidates:
        if used >= target_floor:
            break
        add(candidate)

    if not selected:
        return trim_to_tokens(text, token_limit)

    selected = fit_selected_prose(candidates, selected, token_limit)
    rendered = render_prose_units(candidates, selected)
    return trim_to_tokens(rendered, token_limit)


def split_prose_units(text: str) -> list[tuple[int, str]]:
    units: list[tuple[int, str]] = []
    cleaned = clean_plain_text(text)
    if len(QA_MARKER_RE.findall(cleaned)) >= 2:
        cleaned = QA_MARKER_RE.sub(lambda match: "\n\n" + match.group("label").strip() + ": ", cleaned)
    paragraphs = re.split(r"\n\s*\n+", cleaned)
    for paragraph_idx, paragraph in enumerate(paragraphs):
        paragraph = re.sub(r"\s+", " ", paragraph.strip())
        if not paragraph:
            continue
        if looks_like_heading(paragraph) or len(paragraph) <= 120 and not re.search(r"[.!?。！？]", paragraph):
            units.append((paragraph_idx, paragraph))
            continue
        parts = PROSE_SENTENCE_RE.findall(paragraph) or [paragraph]
        for part in parts:
            unit = part.strip()
            if unit:
                units.append((paragraph_idx, unit))
    return units


def prose_frequencies(units: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for unit in units:
        seen = set()
        for word in prose_words(unit):
            if word in seen:
                continue
            seen.add(word)
            counts[word] = counts.get(word, 0) + 1
    return counts


def prose_words(text: str) -> list[str]:
    words = []
    for raw in TEXT_WORD_RE.findall(text.lower()):
        if raw in TEXT_STOPWORDS or len(raw) <= 2 and not raw.isdigit():
            continue
        words.append(raw)
    return words


def prose_unit_score(
    unit: str,
    idx: int,
    total: int,
    paragraph_idx: int,
    frequencies: dict[str, int],
) -> float:
    words = prose_words(unit)
    unique_words = set(words)
    score = 0.0
    if idx == 0:
        score += 18
    elif idx < 4:
        score += 9
    if idx >= max(0, total - 3):
        score += 5
    if paragraph_idx == 0:
        score += 4
    if looks_like_heading(unit):
        score += 12
    label = leading_qa_label(unit)
    if label:
        if re.match(r"(?i)answer|response", label):
            score += 24
        elif re.match(r"(?i)question|query", label):
            score += 16
        else:
            score += 9
    score += min(20, sum(frequencies.get(word, 0) for word in unique_words) * 0.7)
    if re.search(r"\d|[$€£%]", unit):
        score += 18
    if re.search(r"(?:\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b)", unit):
        score += 6
    if re.search(r"['\"“”‘’()]", unit):
        score += 3
    if PROSE_CUE_RE.search(unit):
        score += 8
    if KEY_QUALIFIER_RE.search(unit):
        score += 12
    if NEGATION_RE.search(unit):
        score += 16
    if HARD_NEGATION_RE.search(unit):
        score += 10
    if SIGNAL_RE.search(unit):
        score += 6
    length = len(unit)
    if 60 <= length <= 260:
        score += 5
    elif length < 25:
        score -= 4
    elif length > 420:
        score -= 5
    return score


def looks_like_heading(text: str) -> bool:
    stripped = text.strip()
    if not stripped or len(stripped) > 100:
        return False
    if stripped.endswith((".", "!", "?", "。", "！", "？")):
        return False
    return bool(
        re.match(r"^(?:#{1,6}\s+|[-*]\s+)?[A-Z0-9][^.!?。！？]{2,}$", stripped)
        or re.match(r"^\d+(?:\.\d+)*[.)]?\s+\S+", stripped)
        or leading_qa_label(stripped)
    )


def leading_qa_label(text: str) -> str:
    match = re.match(
        r"(?i)^\s*((?:question|query|answer|response|passage|context|article|document|background|source|evidence)\s*\d*)\s*[:：]",
        text,
    )
    return match.group(1) if match else ""


def render_prose_units(candidates: list[dict], selected: set[int]) -> str:
    parts: list[str] = []
    last_paragraph = -1
    for candidate in sorted((c for c in candidates if int(c["idx"]) in selected), key=lambda item: int(item["idx"])):
        paragraph = int(candidate["paragraph"])
        unit = str(candidate["unit"]).strip()
        if not unit:
            continue
        if parts and paragraph != last_paragraph:
            parts.append("")
            parts.append(unit)
        elif parts and not looks_like_heading(parts[-1]) and not looks_like_heading(unit):
            parts[-1] = parts[-1].rstrip() + " " + unit
        else:
            parts.append(unit)
        last_paragraph = paragraph
    return "\n".join(parts).strip()


def fit_selected_prose(candidates: list[dict], selected: set[int], token_limit: int) -> set[int]:
    selected = set(selected)
    by_idx = {int(candidate["idx"]): candidate for candidate in candidates}

    def removal_priority(idx: int) -> tuple[int, float, float, int]:
        unit = str(by_idx[idx]["unit"])
        protected = 0
        if KEY_QUALIFIER_RE.search(unit):
            protected += 1
        if NEGATION_RE.search(unit):
            protected += 2
        if HARD_NEGATION_RE.search(unit):
            protected += 1
        return (
            protected,
            float(by_idx[idx]["score"]) / max(1, int(by_idx[idx]["cost"])),
            float(by_idx[idx]["score"]),
            -idx,
        )

    while selected and token_count(render_prose_units(candidates, selected)) > token_limit:
        removable = [idx for idx in selected if idx != 0]
        if not removable:
            break
        weakest = min(removable, key=removal_priority)
        selected.remove(weakest)
    return selected


def schema_settings(tier: str) -> dict[str, int]:
    if tier == "ultra":
        return {
            "causes": 3,
            "edits": 4,
            "checks": 5,
            "discovery": 6,
            "problem_cap": 420,
            "cause_cap": 260,
            "edit_cap": 340,
            "check_cap": 260,
            "final_cap": 360,
            "trace_cap": 220,
            "chrono_cap": 180,
            "strong_chrono_cap": 240,
        }
    if tier == "tight":
        return {
            "causes": 4,
            "edits": 5,
            "checks": 6,
            "discovery": 8,
            "problem_cap": 560,
            "cause_cap": 340,
            "edit_cap": 420,
            "check_cap": 320,
            "final_cap": 460,
            "trace_cap": 280,
            "chrono_cap": 240,
            "strong_chrono_cap": 340,
        }
    if tier == "balanced":
        return {
            "causes": 6,
            "edits": 8,
            "checks": 10,
            "discovery": 12,
            "problem_cap": 820,
            "cause_cap": 500,
            "edit_cap": 640,
            "check_cap": 520,
            "final_cap": 760,
            "trace_cap": 430,
            "chrono_cap": 360,
            "strong_chrono_cap": 500,
        }
    return {
        "causes": 5,
        "edits": 7,
        "checks": 8,
        "discovery": 10,
        "problem_cap": 700,
        "cause_cap": 420,
        "edit_cap": 540,
        "check_cap": 420,
        "final_cap": 620,
        "trace_cap": 360,
        "chrono_cap": 300,
        "strong_chrono_cap": 420,
    }


def compress_schema_trace(
    events: list[tuple[int, str, str]],
    token_limit: int,
    *,
    tier: str,
) -> str:
    if token_limit < 96:
        return trim_to_tokens(compact_plain_schema(events), token_limit)

    settings = schema_settings(tier)
    core_settings = schema_settings("ultra")
    small = token_limit < 180
    problem = first_body(events, "PROBLEM")
    final = last_body(events, "FINAL")
    core_causes = ranked_bodies(events, is_root_cause_reason, core_settings["causes"])
    core_edits = ranked_bodies(events, is_edit_action, core_settings["edits"])
    core_checks = ranked_bodies(events, is_error_or_success_result, core_settings["checks"])
    core_discovery = ranked_bodies(events, is_discovery_event, core_settings["discovery"])

    lines: list[str] = []
    if problem:
        lines.append("Issue: " + one_line(problem, 220 if small else core_settings["problem_cap"]))
    for body in core_causes:
        lines.append("Cause: " + one_line(body, 150 if small else core_settings["cause_cap"]))
    for body in core_edits:
        lines.append("Edit: " + one_line(body, 160 if small else core_settings["edit_cap"]))
    for body in core_checks:
        lines.append("Verify/Error: " + one_line(body, 150 if small else core_settings["check_cap"]))
    if final:
        lines.append("Final: " + one_line(final, 180 if small else core_settings["final_cap"]))
    for body in core_discovery:
        lines.append("Trace: " + one_line(body, 120 if small else core_settings["trace_cap"]))

    # Higher-ratio modes must not dilute the high-scoring 0.1 core. They append
    # only extra high-confidence facts after the same ultra skeleton.
    if tier != "ultra":
        extra_causes = ranked_bodies(events, is_root_cause_reason, settings["causes"])[core_settings["causes"] :]
        extra_edits = ranked_bodies(events, is_edit_action, settings["edits"])[core_settings["edits"] :]
        extra_checks = ranked_bodies(events, is_error_or_success_result, settings["checks"])[core_settings["checks"] :]
        extra_discovery = ranked_bodies(events, is_discovery_event, settings["discovery"])[core_settings["discovery"] :]
        for body in extra_causes:
            lines.append("Cause detail: " + one_line(body, 150 if small else settings["cause_cap"]))
        for body in extra_edits:
            lines.append("Edit detail: " + one_line(body, 160 if small else settings["edit_cap"]))
        for body in extra_checks:
            lines.append("Verify detail: " + one_line(body, 150 if small else settings["check_cap"]))
        for body in extra_discovery:
            lines.append("Trace detail: " + one_line(body, 120 if small else settings["trace_cap"]))

    # Use spare budget for unknown future questions. Ultra can still include a
    # thin chronological trace; larger tiers add only events with strong factual
    # anchors such as errors, edits, file/function paths, qualifiers, or negation.
    for weight, label, body in events:
        if label == "PROBLEM":
            continue
        candidate = {"idx": 0, "weight": weight, "label": label, "body": body}
        if tier != "ultra" and not is_supporting_schema_event(candidate):
            continue
        prefix = schema_prefix(label, body)
        cap = core_settings["chrono_cap"] if tier == "ultra" else settings["chrono_cap"]
        if weight >= 85:
            cap = core_settings["strong_chrono_cap"] if tier == "ultra" else settings["strong_chrono_cap"]
        if small:
            cap = min(cap, 140)
        lines.append(prefix + one_line(body, cap))

    lines = dedupe_lines(lines)
    if not lines:
        lines = [one_line(events[-1][2], 800)]

    header = '<message role="user">\n<text>\n'
    footer = '\n</text>\n</message>'
    budget = max(1, token_limit - token_count(header + footer) - 8)
    body_lines: list[str] = []
    used = 0
    for line in lines:
        cost = token_count(line + "\n")
        if used + cost <= budget:
            body_lines.append(line)
            used += cost
        elif budget - used > max(8, min(20, budget // 4)):
            body_lines.append(trim_to_tokens(line, budget - used))
            break
    body = "\n".join(body_lines).strip()
    while token_count(header + body + footer) > token_limit and body:
        body = trim_to_tokens(body, max(1, token_count(body) - 8)).rstrip()
    return header + body + footer


def compact_plain_schema(events: list[tuple[int, str, str]]) -> str:
    problem = first_body(events, "PROBLEM")
    final = last_body(events, "FINAL")
    edits = ranked_bodies(events, is_edit_action, 1)
    checks = ranked_bodies(events, is_error_or_success_result, 1)
    actions = ranked_bodies(events, lambda c: c["label"].startswith("ACTION"), 1)
    results = ranked_bodies(events, lambda c: c["label"].startswith("RESULT"), 1)
    reasons = ranked_bodies(events, lambda c: c["label"].startswith("REASON"), 4)
    parts = []
    if problem:
        parts.append("Issue: " + one_line(problem, 90))
    if checks:
        parts.append("Result: " + one_line(checks[0], 160))
    elif results:
        parts.append("Result: " + one_line(results[0], 180))
    if edits:
        parts.append("Edit: " + one_line(edits[0], 160))
    elif actions:
        parts.append("Action: " + one_line(actions[0], 140))
    if final:
        parts.append("Final: " + one_line(final, 220))
    if not (edits or actions or checks or results or final):
        for reason in reasons[:4]:
            parts.append("Reason: " + one_line(reason, 120))
    else:
        for reason in reasons[:2]:
            parts.append("Reason: " + one_line(reason, 90))
    return " | ".join(parts) if parts else one_line(events[-1][2], 240)


def clean_plain_text(text: str) -> str:
    text = unescape_jsonish(text)
    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def select_balanced_trace(
    events: list[tuple[int, str, str]],
    token_limit: int,
    *,
    tier: str = "balanced",
) -> str:
    candidates = [
        {
            "idx": idx,
            "weight": weight,
            "label": label,
            "body": body,
            "rendered": render_event(label, body),
        }
        for idx, (weight, label, body) in enumerate(events)
    ]

    output_parts: list[tuple[str, str]] = []
    selected: set[int] = set()
    used = 0

    def add(candidate: dict, cap: int, force: bool = False) -> None:
        nonlocal used
        idx = candidate["idx"]
        if idx in selected or used >= token_limit:
            return
        part = fit_event(candidate["rendered"], cap)
        cost = token_count(part + "\n")
        if used + cost <= token_limit:
            output_parts.append((candidate["label"], part))
            used += cost
            selected.add(idx)
        elif force and token_limit - used > 24:
            output_parts.append((candidate["label"], trim_to_tokens(part, token_limit - used)))
            used = token_limit
            selected.add(idx)

    if tier == "rich":
        anchor_cap = max(140, token_limit // 5)
        thin_cap = max(70, token_limit // 11)
        medium_cap = max(100, token_limit // 7)
    else:
        anchor_cap = max(72, token_limit // 7)
        thin_cap = max(36, token_limit // 16)
        medium_cap = max(52, token_limit // 10)

    # Unknown QA can ask about any phase, so preserve a compact skeleton first.
    for candidate in candidates:
        if candidate["label"] == "PROBLEM":
            add(candidate, max(anchor_cap, token_limit // 4), force=True)

    for predicate, cap in (
        (is_edit_action, anchor_cap),
        (is_error_or_success_result, medium_cap),
        (is_final_state, anchor_cap),
        (is_root_cause_reason, medium_cap),
        (is_discovery_event, thin_cap),
    ):
        for candidate in sorted((c for c in candidates if predicate(c)), key=event_rank, reverse=True):
            add(candidate, cap)

    if tier == "rich":
        for candidate in candidates:
            if candidate["weight"] >= 70:
                add(candidate, medium_cap)

    # Then fill chronologically with a thin trace so unexpected questions about
    # intermediate commands, files, failed attempts, or tests still have anchors.
    for candidate in candidates:
        add(candidate, thin_cap)

    if not output_parts:
        last = candidates[-1]
        output_parts = [(last["label"], fit_event(last["rendered"], token_limit))]

    ordered = restore_trace_order(output_parts, events)
    return ordered


def event_rank(candidate: dict) -> tuple[int, int]:
    return int(candidate["weight"]), -int(candidate["idx"])


def is_edit_action(candidate: dict) -> bool:
    text = candidate["body"].lower()
    return candidate["label"].startswith("ACTION") and bool(
        re.search(
            r"edit|patch|apply_patch|replace|str_replace_editor|oldtext|old_str|"
            r"newtext|new_str|insert|create|successfully replaced|path=",
            text,
        )
    )


def is_error_or_success_result(candidate: dict) -> bool:
    text = candidate["body"].lower()
    return candidate["label"].startswith("RESULT") and bool(
        re.search(
            r"traceback|error|exception|failed|warning|successfully|passed|"
            r"assert|exit code|tests pass|no changes|response clipped|observation",
            text,
        )
    )


def is_final_state(candidate: dict) -> bool:
    return candidate["label"] == "FINAL"


def is_root_cause_reason(candidate: dict) -> bool:
    text = candidate["body"].lower()
    return candidate["label"].startswith("REASON") and bool(
        re.search(r"bug|issue|problem|because|root cause|fix|wrong|missing|invalid|clear", text)
    )


def is_discovery_event(candidate: dict) -> bool:
    text = candidate["body"].lower()
    return bool(
        candidate["label"].startswith("ACTION")
        and re.search(
            r"grep|rg |find |sed |cat |read |view|bash|test|pytest|python|npm|"
            r"cargo|go test|str_replace_editor",
            text,
        )
    ) or bool(
        candidate["label"].startswith("RESULT")
        and re.search(
            r"@\w/|/testbed|\.py|\.js|\.ts|\.go|\.rs|diff --git|@@ |"
            r"def |class |function |line \d+",
            text,
        )
    )


def is_supporting_schema_event(candidate: dict) -> bool:
    body = candidate["body"]
    return (
        int(candidate["weight"]) >= 85
        or is_edit_action(candidate)
        or is_error_or_success_result(candidate)
        or is_final_state(candidate)
        or is_root_cause_reason(candidate)
        or is_discovery_event(candidate)
        or KEY_QUALIFIER_RE.search(body) is not None
        or NEGATION_RE.search(body) is not None
    )


def first_body(events: list[tuple[int, str, str]], label: str) -> str:
    for _weight, event_label, body in events:
        if event_label == label:
            return body
    return ""


def last_body(events: list[tuple[int, str, str]], label: str) -> str:
    for _weight, event_label, body in reversed(events):
        if event_label == label:
            return body
    return ""


def first_result_body(events: list[tuple[int, str, str]]) -> str:
    for _weight, event_label, body in events:
        if event_label.startswith("RESULT"):
            return body
    return ""


def ranked_bodies(events: list[tuple[int, str, str]], predicate, limit: int) -> list[str]:
    candidates = [
        {"idx": idx, "weight": weight, "label": label, "body": body}
        for idx, (weight, label, body) in enumerate(events)
    ]
    picked = sorted((c for c in candidates if predicate(c)), key=event_rank, reverse=True)
    return [c["body"] for c in picked[:limit]]


def one_line(text: str, char_limit: int) -> str:
    text = clean_text(text)
    text = re.sub(r"\s+", " ", text)
    if len(text) <= char_limit:
        return text
    return text[: max(0, char_limit - 3)].rstrip() + "..."


def schema_prefix(label: str, body: str) -> str:
    if label.startswith("REASON"):
        return "Reason: "
    if label.startswith("ACTION"):
        return "Action: "
    if label.startswith("RESULT"):
        return "Result: "
    if label == "FINAL":
        return "Final: "
    return "Trace: "


def dedupe_lines(lines: list[str]) -> list[str]:
    seen = set()
    out = []
    for line in lines:
        marker = re.sub(r"\W+", " ", line.lower())[:180]
        if marker and marker not in seen:
            seen.add(marker)
            out.append(line)
    return out


def build_events(text: str) -> list[tuple[int, str, str]]:
    json_events = build_json_events(text)
    if json_events:
        return dedupe_events(json_events)

    events: list[tuple[int, str, str]] = []
    call_counter = 0
    result_counter = 0

    for match in EVENT_RE.finditer(text):
        role = match.group("role")
        if role:
            body = match.group("msg") or ""
            role = role.lower()
            if role in {"user", "human"}:
                issue = extract_user_problem(body)
                if issue:
                    events.append((100, "PROBLEM", issue))
            elif role in {"assistant", "agent"}:
                for thought in extract_reason_blocks(body):
                    thought = compress_thinking(thought)
                    if thought:
                        events.append((72, "REASON", thought))

                for name, payload in extract_tool_calls(body):
                    call_counter += 1
                    rendered = render_tool_call(name, payload)
                    weight = 86 if name in {"edit", "write", "apply_patch"} else 66
                    events.append((weight, f"ACTION {call_counter}", rendered))

                text_blocks = extract_tag_blocks(body, "text")
                plain = strip_markup(remove_tool_calls(body))
                if text_blocks:
                    plain = "\n".join(text_blocks)
                final = compress_final_answer(plain)
                if final:
                    events.append((98, "FINAL", final))
            continue

        result = match.group("result")
        if result is not None:
            result_counter += 1
            rendered, weight = compress_tool_result(result)
            if rendered:
                events.append((weight, f"RESULT {result_counter}", rendered))

    if events:
        return dedupe_events(events)

    stepwise_events = build_stepwise_events(text)
    if stepwise_events:
        return dedupe_events(stepwise_events)

    generic_events = build_generic_events(text)
    if generic_events:
        return dedupe_events(generic_events)
    return []


def build_json_events(text: str) -> list[tuple[int, str, str]]:
    stripped = text.strip()
    if not stripped or stripped[0] not in "[{":
        return []
    try:
        obj = json.loads(stripped)
    except Exception:
        return []

    if isinstance(obj, dict):
        shaped = build_shaped_json_events(obj)
        if shaped:
            return shaped

    records = obj if isinstance(obj, list) else obj.get("messages") or obj.get("conversation") or obj.get("events") or [obj]
    if not isinstance(records, list):
        return []

    events: list[tuple[int, str, str]] = []
    action_count = 0
    result_count = 0
    for item in records:
        if not isinstance(item, dict):
            continue
        role = normalize_json_role(item)
        content = jsonish_text(
            item.get("content")
            or item.get("text")
            or item.get("message")
            or item.get("output")
            or item.get("result")
            or item.get("observation")
            or ""
        )
        if role in {"user", "human", "prompt", "user_prompt", "input", "request"}:
            problem = extract_user_problem(content)
            if problem:
                events.append((100, "PROBLEM", problem))
        elif role in {"assistant", "agent", "model"}:
            thought = jsonish_text(item.get("thinking") or item.get("thought") or item.get("reasoning") or "")
            if thought:
                events.append((72, "REASON", compress_thinking(thought)))
            for name, payload, params in extract_loose_function_calls(content):
                if name == "finish":
                    finish_text = jsonish_text(params.get("result") or params.get("text") or params.get("message") or payload)
                    final_text = compress_final_answer(finish_text) or clean_text(finish_text)
                    if final_text:
                        events.append((98, "FINAL", final_text))
                    continue
                action_count += 1
                events.append((action_weight(name, payload), f"ACTION {action_count}", render_tool_call(name, payload)))
            assistant_text = remove_loose_function_calls(content)
            final = compress_assistant_json_content(assistant_text)
            if final:
                events.append((98, "FINAL", final))
        elif role in {"tool", "function", "observation", "result", "tool_result", "tool_response"}:
            result_count += 1
            rendered, weight = compress_tool_result(content)
            if rendered:
                events.append((weight, f"RESULT {result_count}", rendered))
        elif role in {"action", "tool_call", "function_call", "command"}:
            action_count += 1
            name, payload = extract_json_action(item)
            events.append((action_weight(name, payload), f"ACTION {action_count}", render_tool_call(name, payload)))

        calls = item.get("tool_calls") or item.get("actions") or []
        if isinstance(calls, dict):
            calls = [calls]
        for call in calls if isinstance(calls, list) else []:
            if not isinstance(call, dict):
                continue
            action_count += 1
            name, payload = extract_json_action(call)
            events.append((action_weight(name, payload), f"ACTION {action_count}", render_tool_call(name, payload)))

        if "function_call" in item and isinstance(item["function_call"], dict):
            action_count += 1
            call = item["function_call"]
            name, payload = extract_json_action(call)
            events.append((action_weight(name, payload), f"ACTION {action_count}", render_tool_call(name, payload)))

    return events


def jsonish_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(jsonish_text(item.get("text") or item.get("content") or item.get("output") or item))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def normalize_json_role(item: dict) -> str:
    role = str(
        item.get("role")
        or item.get("message_type")
        or item.get("type")
        or item.get("speaker")
        or ""
    ).lower()
    if role == "text" and item.get("tool_call_id"):
        return "tool"
    if role == "function" and ("arguments" in item or "function" in item):
        return "action"
    return role


def extract_json_action(item: dict) -> tuple[str, str]:
    function = item.get("function")
    if isinstance(function, dict):
        name = str(function.get("name") or item.get("name") or "function")
        payload = function.get("arguments") or function.get("args") or function.get("input") or function
        return name, jsonish_text(payload)
    name = str(item.get("name") or item.get("tool") or item.get("type") or item.get("action") or "tool")
    payload = item.get("arguments") or item.get("args") or item.get("input") or item.get("parameters") or item
    return name, jsonish_text(payload)


def action_weight(name: str, payload: str) -> int:
    text = f"{name} {payload}".lower()
    if re.search(r"file_editor|str_replace_editor|str_replace|edit|patch|apply_patch|old_str|new_str|insert|create", text):
        return 92
    if re.search(r"bash|command|pytest|python|npm|cargo|go test|grep|rg |sed |cat ", text):
        return 74
    return 66


def build_shaped_json_events(obj: dict) -> list[tuple[int, str, str]]:
    if "messages" in obj or "conversation" in obj or "events" in obj:
        return []

    patch = obj.get("model_patch") or obj.get("patch") or obj.get("diff")
    if isinstance(patch, str) and patch.strip():
        return build_patch_record_events(obj, patch)

    if "file_name" in obj and isinstance(obj.get("text"), str):
        meta = [
            f"File: {obj.get('file_name')}",
            f"is_test={obj.get('is_test')}",
            f"is_cypress={obj.get('is_cypress')}",
        ]
        return [
            (100, "PROBLEM", " ".join(meta)),
            (66, "ACTION 1", "compress source file while preserving imports, configuration keys, exported symbols, tests, and constants"),
            (82, "RESULT 1", clean_plain_text(str(obj.get("text", "")))),
        ]

    scenario_keys = {
        "scenario_title",
        "domain",
        "problem_statement",
        "environment_context",
        "input_request",
        "agent_expected_actions",
        "edge_cases_and_risks",
        "success_criteria",
    }
    if scenario_keys & set(obj):
        title = str(obj.get("scenario_title") or "")
        domain = str(obj.get("domain") or "")
        problem = str(obj.get("problem_statement") or "")
        request = str(obj.get("input_request") or "")
        context = str(obj.get("environment_context") or "")
        actions = obj.get("agent_expected_actions") or []
        risks = obj.get("edge_cases_and_risks") or []
        success = obj.get("success_criteria") or []

        events: list[tuple[int, str, str]] = []
        events.append((100, "PROBLEM", clean_text(f"{title}. Domain: {domain}. Problem: {problem}. Request: {request}")))
        if context:
            events.append((78, "RESULT 1", clean_text("Context: " + context)))
        for idx, action in enumerate(actions if isinstance(actions, list) else [actions], 1):
            events.append((76, f"ACTION {idx}", clean_text(str(action))))
        if risks:
            body = "; ".join(str(item) for item in (risks if isinstance(risks, list) else [risks]))
            events.append((88, "RESULT 2", clean_text("Edge cases/risks: " + body)))
        if success:
            body = "; ".join(str(item) for item in (success if isinstance(success, list) else [success]))
            events.append((98, "FINAL", clean_text("Success criteria: " + body)))
        return events

    return []


def build_patch_record_events(obj: dict, patch: str) -> list[tuple[int, str, str]]:
    instance_id = str(obj.get("instance_id") or obj.get("task_id") or obj.get("id") or "")
    model_name = str(obj.get("model_name_or_path") or obj.get("model") or "").split("__")[0]
    files, hunks, removed, added = summarize_unified_patch(patch)
    file_text = ", ".join(files[:12]) if files else "unknown files"
    meta = clean_text(f"SWE patch trajectory instance={instance_id} model={model_name} files={file_text}")
    summary_bits = [
        f"files={file_text}",
        f"hunks={'; '.join(hunks[:8])}" if hunks else "",
        f"removed={'; '.join(removed[:16])}" if removed else "",
        f"added={'; '.join(added[:18])}" if added else "",
    ]
    summary = clean_text("Patch summary: " + " | ".join(bit for bit in summary_bits if bit))
    return [
        (100, "PROBLEM", meta),
        (92, "ACTION 1", clean_text("apply model_patch to " + file_text)),
        (94, "RESULT 1", summary),
        (98, "FINAL", "Model patch:\n" + clean_plain_text(patch)),
    ]


def summarize_unified_patch(patch: str) -> tuple[list[str], list[str], list[str], list[str]]:
    files: list[str] = []
    hunks: list[str] = []
    removed: list[str] = []
    added: list[str] = []
    for raw_line in patch.splitlines():
        line = raw_line.rstrip()
        file_match = re.match(r"diff --git a/(.*?) b/(.*)", line)
        if file_match:
            path = file_match.group(2)
            if path not in files:
                files.append(path)
            continue
        if line.startswith("@@"):
            hunks.append(line[:180])
            continue
        if line.startswith("---") or line.startswith("+++"):
            continue
        if line.startswith("-") and len(line) > 1:
            removed.append(line[:220])
        elif line.startswith("+") and len(line) > 1:
            added.append(line[:220])
    return files, hunks, removed, added


def compress_assistant_json_content(content: str) -> str:
    if not content:
        return ""
    if "```" in content or looks_like_path_or_code(content):
        lines = [line.rstrip() for line in clean_plain_text(content).splitlines() if line.strip()]
        signal = [
            line
            for line in lines
            if SIGNAL_RE.search(line)
            or KEY_QUALIFIER_RE.search(line)
            or NEGATION_RE.search(line)
            or looks_like_path_or_code(line)
        ]
        kept = (signal or lines)[:18]
        return clean_text("\n".join(kept))
    return compress_final_answer(content)


def build_stepwise_events(text: str) -> list[tuple[int, str, str]]:
    cleaned = clean_plain_text(text)
    matches = list(STEP_MARKER_RE.finditer(cleaned))
    if len(matches) < 3 or not any(match.group("label").lower().startswith("step") for match in matches):
        return []

    events: list[tuple[int, str, str]] = []
    for idx, match in enumerate(matches):
        label = clean_text(match.group("label"))
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(cleaned)
        body = clean_text(cleaned[start:end])
        if not body:
            continue
        rendered = f"{label}: {body}"
        event_label, weight = classify_stepwise_event(label, body, idx)
        events.append((weight, event_label, rendered))
    return events


def classify_stepwise_event(label: str, body: str, idx: int) -> tuple[str, int]:
    label_text = label.lower()
    text = f"{label} {body}".lower()
    if (
        idx == 0
        or label_text.startswith("sample")
        or re.search(r"\b(problem understanding|understanding the problem|known quantities|given values)\b", label_text)
    ):
        return "PROBLEM", 100
    if re.search(r"\b(final answer|conclusion|answer|therefore|final result)\b", text):
        return "FINAL", 98
    if re.search(r"\b(verification|verify|check|edge case|test|valid|invalid|constraint)\b", text):
        return "RESULT", 86
    if re.search(r"\b(equation|formula|compute|calculate|derive|apply|enumerat|translate|define)\b", text):
        return "REASON", 78
    return "REASON", 72


def build_generic_events(text: str) -> list[tuple[int, str, str]]:
    events: list[tuple[int, str, str]] = []
    current_label = ""
    current_lines: list[str] = []
    counters = {"action": 0, "result": 0}

    def flush() -> None:
        nonlocal current_label, current_lines
        body = clean_text("\n".join(current_lines))
        if not body:
            current_label = ""
            current_lines = []
            return
        add_generic_event(events, current_label, body, counters)
        current_label = ""
        current_lines = []

    for line in text.splitlines():
        match = GENERIC_LABEL_RE.match(line)
        if match and normalize_label(match.group("label")):
            flush()
            current_label = normalize_label(match.group("label"))
            current_lines = [match.group("body")]
        elif current_label:
            current_lines.append(line)
        elif SIGNAL_RE.search(line) or looks_like_path_or_code(line):
            add_generic_event(events, "result", line.strip(), counters)
    flush()
    return events


def add_generic_event(
    events: list[tuple[int, str, str]],
    label: str,
    body: str,
    counters: dict[str, int],
) -> None:
    label = normalize_label(label)
    if label in {"user", "human", "prompt", "problem", "issue"}:
        events.append((100, "PROBLEM", extract_user_problem(body) or body))
    elif label == "thought":
        events.append((72, "REASON", compress_thinking(body)))
    elif label == "action":
        counters["action"] += 1
        name = "edit" if re.search(r"\b(edit|patch|change|replace)\b", body, re.I) else "action"
        weight = 86 if name == "edit" or SIGNAL_RE.search(body) else 66
        events.append((weight, f"ACTION {counters['action']}", render_loose_action(name, body)))
    elif label == "result":
        counters["result"] += 1
        rendered, weight = compress_tool_result(body)
        if SIGNAL_RE.search(body):
            weight = max(weight, 88)
        events.append((weight, f"RESULT {counters['result']}", rendered or body))
    elif label in {"assistant", "final"}:
        final = compress_final_answer(body) or compress_thinking(body)
        events.append((98 if label == "final" else 76, "FINAL", final))


def normalize_label(label: str) -> str:
    label = clean_label(label)
    return LABEL_ALIASES.get(label, "")


def clean_label(label: str) -> str:
    return re.sub(r"[\s-]+", "_", label.strip().lower())


def extract_user_problem(body: str) -> str:
    text = "\n".join(extract_tag_blocks(body, "text")) or strip_markup(body)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    kept: list[str] = []
    for line in lines:
        if len(kept) < 2 or SIGNAL_RE.search(line) or line.startswith(("#", "$", "File ")):
            kept.append(line)
        if len(" ".join(kept)) > 1400:
            break
    return clean_text("\n".join(kept))


def extract_tag_blocks(body: str, tag: str) -> list[str]:
    pattern = re.compile(rf"<{tag}[^>]*>(.*?)</{tag}>", re.S)
    return [clean_text(m.group(1)) for m in pattern.finditer(body)]


def extract_reason_blocks(body: str) -> list[str]:
    blocks: list[str] = []
    for tag in ("thinking", "thought", "reasoning"):
        blocks.extend(extract_tag_blocks(body, tag))
    return blocks


def extract_tool_calls(body: str) -> list[tuple[str, str]]:
    calls = []
    for match in TOOL_CALL_RE.finditer(body):
        calls.append((match.group("name"), clean_text(match.group("body"))))
    for name, payload, _params in extract_loose_function_calls(body):
        calls.append((name, payload))
    return calls


def extract_loose_function_calls(text: str) -> list[tuple[str, str, dict[str, str]]]:
    calls: list[tuple[str, str, dict[str, str]]] = []
    if not text or "<function=" not in text:
        return calls
    for match in LOOSE_FUNCTION_RE.finditer(text):
        name = clean_label(match.group("name"))
        body = match.group("body")
        params: dict[str, str] = {}
        for param_match in PARAMETER_RE.finditer(body):
            key = clean_label(param_match.group("key"))
            value = html.unescape(param_match.group("value").strip())
            if key:
                params[key] = value
        if params:
            payload = json.dumps(params, ensure_ascii=False)
        else:
            payload = json.dumps({"body": clean_text(body)}, ensure_ascii=False)
        calls.append((name, payload, params))
    return calls


def remove_loose_function_calls(text: str) -> str:
    return LOOSE_FUNCTION_RE.sub(" ", text)


def render_tool_call(name: str, payload: str) -> str:
    try:
        obj = json.loads(payload)
    except Exception:
        payload = unescape_jsonish(payload)
        try:
            obj = json.loads(payload)
        except Exception:
            obj = None
    if isinstance(obj, dict):
        editor_command = str(obj.get("command") or obj.get("action") or "").lower()
        if (
            name == "str_replace_editor"
            or "str_replace_editor" in name
            or name in {"file_editor", "editor"}
            or editor_command in {"view", "create", "insert", "str_replace", "replace"}
        ):
            bits = []
            for key in ("command", "action", "path", "insert_line", "offset", "limit", "view_range", "concise"):
                if key in obj:
                    bits.append(f"{key}={one_line(str(obj[key]), 140)}")
            old_value = obj.get("old_str") or obj.get("oldText")
            new_value = obj.get("new_str") or obj.get("newText")
            file_text = obj.get("file_text") or obj.get("text")
            if old_value:
                bits.append("old=" + one_line(str(old_value), 180))
            if new_value:
                bits.append("new=" + one_line(str(new_value), 240))
            if file_text:
                bits.append("file_text=" + one_line(str(file_text), 260))
            return clean_text(f"{name} " + " ".join(bits))
        command = obj.get("command") or obj.get("cmd")
        if command is not None:
            return f"{name} command={short_command(str(command))}"
        if "path" in obj:
            bits = [f"{key}={obj[key]}" for key in ("path", "offset", "limit") if key in obj]
            return f"{name} " + " ".join(bits)
        if "edits" in obj:
            path = obj.get("path", "")
            edits = obj.get("edits") or []
            snippets = []
            for edit in edits[:2]:
                old = clean_text(str(edit.get("oldText", "")))[:180]
                new = clean_text(str(edit.get("newText", "")))[:220]
                snippets.append(f"old[{old}] -> new[{new}]")
            return clean_text(f"{name} path={path} " + " ; ".join(snippets))
    return clean_text(f"{name} {payload[:900]}")


def render_loose_action(name: str, body: str) -> str:
    body = clean_text(body)
    if body.startswith("{") and body.endswith("}"):
        return render_tool_call(name, body)
    if name in {"command", "cmd"}:
        return f"exec command={short_command(body)}"
    if re.search(r"\b(apply_patch|patch|edit|replace|oldText|newText)\b", body, re.I):
        return clean_text(f"edit {body[:900]}")
    return clean_text(f"{name} {body[:900]}")


def render_event(label: str, body: str) -> str:
    body = clean_text(body)
    if not body:
        return ""
    if label == "PROBLEM":
        return f'<message role="user">\n<text>\n{body}\n</text>\n</message>'
    if label == "FINAL":
        return f'<message role="assistant">\n<text>\n{body}\n</text>\n</message>'
    if label.startswith("REASON"):
        return f'<message role="assistant">\n<thinking>\n{body}\n</thinking>\n</message>'
    if label.startswith("ACTION"):
        name, payload = split_action_body(body)
        return f'<message role="assistant">\n<tool_call name="{name}">\n{payload}\n</tool_call>\n</message>'
    if label.startswith("RESULT"):
        return f"<tool_result>\n{body}\n</tool_result>"
    return body


def split_action_body(body: str) -> tuple[str, str]:
    body = clean_text(body)
    first, _, rest = body.partition(" ")
    name = re.sub(r"[^A-Za-z0-9_-]", "", first or "tool") or "tool"
    payload = rest.strip() or body
    if name in {"exec", "command", "cmd"} and payload.startswith("command="):
        command = payload[len("command="):]
        payload = json.dumps({"command": command}, ensure_ascii=False)
        name = "exec"
    elif name == "edit" and not payload.startswith("{"):
        payload = payload[:1000]
    return name[:40], payload


def compress_tool_result(result: str) -> tuple[str, int]:
    text = clean_text(result)
    if not text:
        return "", 0
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    signal = [
        line
        for line in lines
        if SIGNAL_RE.search(line)
        or KEY_QUALIFIER_RE.search(line)
        or NEGATION_RE.search(line)
        or looks_like_path_or_code(line)
        or re.search(r"/testbed|diff --git|@@ |response clipped|OBSERVATION", line, re.I)
    ]

    if any(word in text.lower() for word in ("traceback", "error", "failed", "exception", "warning")):
        return "\n".join((signal or lines)[:18]), 88
    if re.search(r"\b(passed|successfully|tests pass|no changes)\b", text, re.I):
        return "\n".join((signal or lines)[:12]), 94
    if "Successfully replaced" in text or "Successfully" in text:
        return "\n".join(lines[:6]), 94
    if "Command exited with code" in text:
        return "\n".join((signal or lines)[:10]), 82
    if signal:
        return "\n".join(signal[:14]), 70
    if len(lines) <= 5:
        return "\n".join(lines), 45
    return "\n".join(lines[:3] + ["...", *lines[-2:]]), 35


def compress_thinking(text: str) -> str:
    sentences = split_sentences(text)
    kept = [s for s in sentences if FINAL_HINT_RE.search(s) or looks_like_path_or_code(s)]
    if not kept:
        kept = sentences[:2]
    return clean_text(" ".join(kept[:6]))


def compress_final_answer(text: str) -> str:
    text = strip_markup(text)
    if not text or "<tool_call" in text:
        return ""
    sentences = split_sentences(text)
    kept = [s for s in sentences if FINAL_HINT_RE.search(s) or looks_like_path_or_code(s)]
    if not kept and len(text) > 40:
        kept = sentences[:5]
    return clean_text(" ".join(kept[:10]))


def scrub_noise(text: str) -> str:
    kept = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            kept.append("")
            continue
        if NOISE_LINE_RE.search(stripped):
            continue
        if re.search(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}", stripped):
            continue
        if re.search(r"[^\w\s]{10,}", stripped) and not SIGNAL_RE.search(stripped):
            continue
        kept.append(line)
    return "\n".join(kept)


def alias_paths(text: str) -> str:
    def repl(match: re.Match) -> str:
        path = match.group(0)
        replacements = (
            ("/home/user/.openclaw/workspace/", "@W/"),
            ("/home/user/workspace/", "@W/"),
            ("/home/user/", "@H/"),
            ("/tmp/", "@T/"),
        )
        for prefix, alias in replacements:
            if path.startswith(prefix):
                return alias + path[len(prefix):]
        parts = path.strip("/").split("/")
        return "@P/" + "/".join(parts[-4:])

    return PATH_RE.sub(repl, text)


def clean_text(text: str) -> str:
    text = unescape_jsonish(text)
    text = TAG_RE.sub("", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_markup(text: str) -> str:
    return clean_text(remove_tool_calls(text))


def remove_tool_calls(text: str) -> str:
    return LOOSE_FUNCTION_RE.sub(" ", TOOL_CALL_RE.sub(" ", text))


def unescape_jsonish(text: str) -> str:
    def decode_json_escape(match: re.Match) -> str:
        escape = match.group(0)[1:]
        simple = {
            '"': '"',
            "\\": "\\",
            "/": "/",
            "b": "\b",
            "f": "\f",
            "n": "\n",
            "r": "\r",
            "t": "\t",
        }
        if escape in simple:
            return simple[escape]
        if escape.startswith(("u", "U")):
            return chr(int(escape[1:], 16))
        if escape.startswith("x"):
            return chr(int(escape[1:], 16))
        return match.group(0)

    text = JSON_ESCAPE_RE.sub(decode_json_escape, text)
    # html.unescape handles named entities plus decimal/hex numeric entities:
    # &lt;, &amp;, &#61;, &#x3d;, etc. Two passes also decode &amp;lt;.
    for _ in range(2):
        decoded = html.unescape(text)
        if decoded == text:
            break
        text = decoded
    return text


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", clean_text(text))
    return [part.strip() for part in parts if part.strip()]


def looks_like_path_or_code(line: str) -> bool:
    return bool(
        re.search(
            r"@\w/|[/\w.-]+\.(?:py|js|ts|tsx|jsx|java|go|rs|rb|php|cs|cpp|c|h|hpp|swift|kt|scala|"
            r"sh|sql|json|yaml|yml|toml|md)\b|"
            r"\b(?:def|class|function|func|fn|public|private|import|require|return|raise|throw|assert|"
            r"const|let|var|if|for|while|try|catch)\b|\bline \d+\b",
            line,
        )
    )


def short_command(command: str) -> str:
    command = clean_text(command)
    noisy = ("pip install", "npm install", "apt-get", "docker pull")
    if any(command.startswith(prefix) for prefix in noisy):
        return command.split("&&")[-1].strip()[:240]
    return command[:420]


def fit_event(text: str, max_tokens: int) -> str:
    if token_count(text) <= max_tokens:
        return text
    lines = text.splitlines()
    if len(lines) > 1:
        head = "\n".join(lines[:8])
        tail = "\n".join(lines[-4:])
        return trim_to_tokens(head + "\n...\n" + tail, max_tokens)
    return trim_to_tokens(text, max_tokens)


def restore_trace_order(parts: list[tuple[str, str]], events: list[tuple[int, str, str]]) -> str:
    order = {label: idx for idx, (_w, label, _body) in enumerate(events)}
    parts = sorted(parts, key=lambda item: order.get(item[0], 10_000))
    return "\n".join(part for _label, part in parts if part.strip())


def repair_transcript_tags(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    fixes = []
    for tag in ("thinking", "text", "tool_call", "message", "tool_result"):
        opens = len(re.findall(rf"<{tag}(?:\s[^>]*)?>", text))
        closes = len(re.findall(rf"</{tag}>", text))
        for _ in range(max(0, opens - closes)):
            fixes.append(f"</{tag}>")
    if fixes:
        text = text.rstrip() + "\n" + "\n".join(fixes)
    return text


def dedupe_events(events: list[tuple[int, str, str]]) -> list[tuple[int, str, str]]:
    seen = set()
    out = []
    for weight, label, body in events:
        body = clean_text(body)
        if not body:
            continue
        marker = re.sub(r"\W+", " ", body.lower())[:220]
        if marker in seen:
            continue
        seen.add(marker)
        out.append((weight, label, body))
    return out


def trim_to_tokens(text: str, token_limit: int) -> str:
    if token_limit <= 0:
        return ""
    if ENCODER is not None:
        ids = ENCODER.encode_ordinary(text)
        if len(ids) <= token_limit:
            return text.strip()
        return ENCODER.decode(ids[:token_limit]).rstrip()
    tokens = WORD_TOKEN_RE.findall(text)
    if len(tokens) <= token_limit:
        return text.strip()
    return "".join(tokens[:token_limit]).rstrip()
