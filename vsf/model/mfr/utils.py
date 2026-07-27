# Copyright (c) Opendatalab. All rights reserved.
import re

LEFT_PATTERN = re.compile(r'(\\left)(\S*)')
RIGHT_PATTERN = re.compile(r'(\\right)(\S*)')
LEFT_COUNT_PATTERN = re.compile(r'\\left(?![a-zA-Z])')
RIGHT_COUNT_PATTERN = re.compile(r'\\right(?![a-zA-Z])')
LEFT_RIGHT_REMOVE_PATTERN = re.compile(r'\\left\.?|\\right\.?')

def fix_latex_left_right(s, fix_delimiter=True):
    """
    Implementation detail.
    Implementation detail.
    Implementation detail.
    """
    # Implementation detail.
    valid_delims_list = [r'(', r')', r'[', r']', r'{', r'}', r'/', r'|',
                         r'\{', r'\}', r'\lceil', r'\rceil', r'\lfloor',
                         r'\rfloor', r'\backslash', r'\uparrow', r'\downarrow',
                         r'\Uparrow', r'\Downarrow', r'\|', r'\.']

    # Add the value to the result.
    def fix_delim(match, is_left=True):
        cmd = match.group(1)  # Implementation detail.
        rest = match.group(2) if len(match.groups()) > 1 else ""
        if not rest or rest not in valid_delims_list:
            return cmd + "."
        return match.group(0)

    # Match the expected pattern.
    # Implementation detail.
    # Implementation detail.
    if fix_delimiter:
        s = LEFT_PATTERN.sub(lambda m: fix_delim(m, True), s)
        s = RIGHT_PATTERN.sub(lambda m: fix_delim(m, False), s)

    # Calculate the result.
    left_count = len(LEFT_COUNT_PATTERN.findall(s))  # Match the expected pattern.
    right_count = len(RIGHT_COUNT_PATTERN.findall(s))  # Match the expected pattern.

    if left_count == right_count:
        # Validate the current value.
        return fix_left_right_pairs(s)
        # return s
    else:
        # Remove invalid or unnecessary data.
        # logger.debug(f"latex:{s}")
        # logger.warning(f"left_count: {left_count}, right_count: {right_count}")
        return LEFT_RIGHT_REMOVE_PATTERN.sub('', s)


def fix_left_right_pairs(latex_formula):
    """
    Process formula content.

    Args:
        Process formula content.

    Returns:
        Process formula content.
    """
    # Implementation detail.
    brace_stack = []
    # Implementation detail.
    left_stack = []
    # Implementation detail.
    adjustments = []

    i = 0
    while i < len(latex_formula):
        # Validate the current value.
        if i > 0 and latex_formula[i - 1] == '\\':
            backslash_count = 0
            j = i - 1
            while j >= 0 and latex_formula[j] == '\\':
                backslash_count += 1
                j -= 1

            if backslash_count % 2 == 1:
                i += 1
                continue

        # Implementation detail.
        if i + 5 < len(latex_formula) and latex_formula[i:i + 5] == "\\left" and i + 5 < len(latex_formula):
            delimiter = latex_formula[i + 5]
            left_stack.append((i, len(brace_stack), delimiter))
            i += 6  # Remove invalid or unnecessary data.
            continue

        # Implementation detail.
        elif i + 6 < len(latex_formula) and latex_formula[i:i + 6] == "\\right" and i + 6 < len(latex_formula):
            delimiter = latex_formula[i + 6]

            if left_stack:
                left_pos, left_depth, left_delim = left_stack.pop()

                # Implementation detail.
                if left_depth != len(brace_stack):
                    # Implementation detail.
                    target_pos = find_group_end(latex_formula, left_pos, left_depth)
                    if target_pos != -1:
                        # Implementation detail.
                        adjustments.append((i, i + 7, target_pos))

            i += 7  # Remove invalid or unnecessary data.
            continue

        # Process the current item.
        if latex_formula[i] == '{':
            brace_stack.append(i)
        elif latex_formula[i] == '}':
            if brace_stack:
                brace_stack.pop()

        i += 1

    # Process the current item.
    if not adjustments:
        return latex_formula

    result = list(latex_formula)
    adjustments.sort(reverse=True, key=lambda x: x[0])

    for start, end, target in adjustments:
        # Extract the required value.
        right_part = result[start:end]
        # Remove invalid or unnecessary data.
        del result[start:end]
        # Add the value to the result.
        result.insert(target, ''.join(right_part))

    return ''.join(result)


def find_group_end(text, pos, depth):
    """Match the expected pattern."""
    current_depth = depth
    i = pos

    while i < len(text):
        if text[i] == '{' and (i == 0 or not is_escaped(text, i)):
            current_depth += 1
        elif text[i] == '}' and (i == 0 or not is_escaped(text, i)):
            current_depth -= 1
            if current_depth < depth:
                return i
        i += 1

    return -1  # Implementation detail.


def is_escaped(text, pos):
    """Validate the current value."""
    backslash_count = 0
    j = pos - 1
    while j >= 0 and text[j] == '\\':
        backslash_count += 1
        j -= 1

    return backslash_count % 2 == 1


def fix_unbalanced_braces(latex_formula):
    """
    Remove invalid or unnecessary data.

    Args:
        Process formula content.

    Returns:
        Remove invalid or unnecessary data.
    """
    stack = []  # Implementation detail.
    unmatched = set()  # Match the expected pattern.
    i = 0

    while i < len(latex_formula):
        # Validate the current value.
        if latex_formula[i] in ['{', '}']:
            # Calculate the result.
            backslash_count = 0
            j = i - 1
            while j >= 0 and latex_formula[j] == '\\':
                backslash_count += 1
                j -= 1

            # Match the expected pattern.
            if backslash_count % 2 == 1:
                i += 1
                continue

            # Match the expected pattern.
            if latex_formula[i] == '{':
                stack.append(i)
            else:  # latex_formula[i] == '}'
                if stack:  # Implementation detail.
                    stack.pop()
                else:  # Implementation detail.
                    unmatched.add(i)

        i += 1

    # Match the expected pattern.
    unmatched.update(stack)

    # Build the required output.
    return ''.join(char for i, char in enumerate(latex_formula) if i not in unmatched)


def process_latex(input_string):
    """
        Process formula content.
        Implementation detail.
        Implementation detail.
        Add the value to the result.

        Args:
            Process formula content.

        Returns:
            Process formula content.
        """

    def replace_func(match):
        # Extract the required value.
        next_char = match.group(1)

        # Implementation detail.
        if next_char in "#$%&~_^|\\{} \t\n\r\v\f":
            return match.group(0)

        # Validate the current value.
        if 'a' <= next_char <= 'z' or 'A' <= next_char <= 'Z':
            pos = match.start() + 2  # Implementation detail.
            if pos < len(input_string) and ('a' <= input_string[pos] <= 'z' or 'A' <= input_string[pos] <= 'Z'):
                # Implementation detail.
                return match.group(0)

        # Add the value to the result.
        return '\\' + ' ' + next_char

    # Match the expected pattern.
    pattern = r'\\(.)'

    return re.sub(pattern, replace_func, input_string)

# Implementation detail.
ENV_TYPES = ['array', 'matrix', 'pmatrix', 'bmatrix', 'vmatrix',
             'Bmatrix', 'Vmatrix', 'cases', 'aligned', 'gathered', 'align', 'align*']
ENV_BEGIN_PATTERNS = {env: re.compile(r'\\begin\{' + env + r'\}') for env in ENV_TYPES}
ENV_END_PATTERNS = {env: re.compile(r'\\end\{' + env + r'\}') for env in ENV_TYPES}
ENV_FORMAT_PATTERNS = {env: re.compile(r'\\begin\{' + env + r'\}\{([^}]*)\}') for env in ENV_TYPES}

def fix_latex_environments(s):
    """
    Match the expected pattern.
    Add the value to the result.
    Add the value to the result.
    """
    for env in ENV_TYPES:
        begin_count = len(ENV_BEGIN_PATTERNS[env].findall(s))
        end_count = len(ENV_END_PATTERNS[env].findall(s))

        if begin_count != end_count:
            if end_count > begin_count:
                format_match = ENV_FORMAT_PATTERNS[env].search(s)
                default_format = '{c}' if env == 'array' else ''
                format_str = '{' + format_match.group(1) + '}' if format_match else default_format

                missing_count = end_count - begin_count
                begin_command = '\\begin{' + env + '}' + format_str + ' '
                s = begin_command * missing_count + s
            else:
                missing_count = begin_count - end_count
                s = s + (' \\end{' + env + '}') * missing_count

    return s


REPLACEMENTS_PATTERNS = {
    re.compile(r'\\underbar'): r'\\underline',
    re.compile(r'\\Bar'): r'\\hat',
    re.compile(r'\\Hat'): r'\\hat',
    re.compile(r'\\Tilde'): r'\\tilde',
    re.compile(r'\\slash'): r'/',
    re.compile(r'\\textperthousand'): r'‰',
    re.compile(r'\\sun'): r'☉',
    re.compile(r'\\textunderscore'): r'\\_',
    re.compile(r'\\fint'): r'⨏',
    re.compile(r'\\up '): r'\\ ',
    re.compile(r'\\vline = '): r'\\models ',
    re.compile(r'\\vDash '): r'\\models ',
    re.compile(r'\\sq \\sqcup '): r'\\square ',
    re.compile(r'\\copyright'): r'©',
    re.compile(r'\\Dot'): r'\\dot',
}
QQUAD_PATTERN = re.compile(r'\\qquad(?!\s)')


def remove_up_commands(s: str):
    """Remove unnecessary up commands from LaTeX code."""
    UP_PATTERN = re.compile(r'\\up([a-zA-Z]+)')
    s = UP_PATTERN.sub(
        lambda m: m.group(0) if m.group(1) in ["arrow", "downarrow", "lus", "silon"] else f"\\{m.group(1)}", s
    )
    return s


def remove_unsupported_commands(s: str):
    """Remove unsupported LaTeX commands."""
    COMMANDS_TO_REMOVE_PATTERN = re.compile(
        r'\\(?:lefteqn|boldmath|ensuremath|centering|textsubscript|sides|textsl|textcent|emph|protect|null)')
    s = COMMANDS_TO_REMOVE_PATTERN.sub('', s)
    return s


def latex_rm_whitespace(s: str):
    """Remove unnecessary whitespace from LaTeX code."""
    s = fix_unbalanced_braces(s)
    s = fix_latex_left_right(s)
    s = fix_latex_environments(s)

    s = remove_up_commands(s)
    s = remove_unsupported_commands(s)

    # Implementation detail.
    for pattern, replacement in REPLACEMENTS_PATTERNS.items():
        s = pattern.sub(replacement, s)

    # Process the current item.
    s = process_latex(s)

    # Implementation detail.
    s = QQUAD_PATTERN.sub(r'\\qquad ', s)

    # Implementation detail.
    while s.endswith('\\'):
        s = s[:-1]

    return s


def largest_power_of_two_leq(value: int) -> int:
    if value < 1:
        return 0
    return 2 ** (value.bit_length() - 1)


def get_mfr_effective_batch_size(num_items: int, requested_batch_size: int) -> int:
    return min(
        requested_batch_size,
        largest_power_of_two_leq(max(1, num_items)),
    )


def get_mfr_min_dynamic_batch_size(requested_batch_size: int) -> int:
    return max(16, requested_batch_size // 4)


def finalize_mfr_batch_groups(
    batch_groups: list[list[int]],
    total_count: int,
    requested_batch_size: int,
) -> list[list[int]]:
    if not batch_groups:
        return []

    if len(batch_groups) == 1:
        if total_count <= 1 or requested_batch_size <= total_count:
            return batch_groups

        first_group_size = largest_power_of_two_leq(total_count - 1)
        if first_group_size < 1:
            return batch_groups

        source_group = batch_groups[0]
        first_group = source_group[:first_group_size]
        second_group = source_group[first_group_size:]
        if not first_group or not second_group:
            return batch_groups
        return [first_group, second_group]

    while (
        len(batch_groups) >= 3
        and len(batch_groups[-1]) < len(batch_groups[-2])
    ):
        tail_group = batch_groups.pop()
        batch_groups[-1].extend(tail_group)

    return batch_groups


def build_mfr_batch_groups(sorted_areas: list[int], requested_batch_size: int) -> list[list[int]]:
    if not sorted_areas:
        return []

    total_count = len(sorted_areas)
    effective_batch_size = get_mfr_effective_batch_size(
        total_count,
        requested_batch_size,
    )
    if effective_batch_size < 1:
        return []

    min_dynamic_batch_size = get_mfr_min_dynamic_batch_size(requested_batch_size)
    batch_groups = []
    if total_count < min_dynamic_batch_size:
        batch_groups.append(list(range(total_count)))
        return finalize_mfr_batch_groups(
            batch_groups,
            total_count,
            requested_batch_size,
        )

    base_mean_area = sum(sorted_areas[:effective_batch_size]) / effective_batch_size
    cursor = 0

    while cursor < total_count:
        remaining_count = total_count - cursor
        if remaining_count < min_dynamic_batch_size:
            batch_groups.append(list(range(cursor, total_count)))
            break

        probe_size = min(effective_batch_size, remaining_count)
        current_mean_area = sum(sorted_areas[cursor : cursor + probe_size]) / probe_size
        ratio = 1 if base_mean_area <= 0 else current_mean_area / base_mean_area

        candidate_batch_size = effective_batch_size
        threshold = 4
        while (
            ratio >= threshold
            and candidate_batch_size // 2 >= min_dynamic_batch_size
        ):
            candidate_batch_size //= 2
            threshold *= 2

        candidate_batch_size = min(
            candidate_batch_size,
            largest_power_of_two_leq(remaining_count),
        )
        batch_groups.append(list(range(cursor, cursor + candidate_batch_size)))
        cursor += candidate_batch_size

    return finalize_mfr_batch_groups(
        batch_groups,
        total_count,
        requested_batch_size,
    )
