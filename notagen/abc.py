"""ABC post-processing shared by batch inference and the demo.

``rest_unreduce`` restores the whole-rest bars that training data omits, producing the
interleaved ABC CLaMP 2 consumes. This is the Gradio copy, which also closes unbalanced
``%%score`` brackets; the batch-inference copy lacked that step.
"""

import re

from abctoolkit.duration import calculate_bartext_duration
from abctoolkit.transpose import Note_list as _Note_list
from abctoolkit.transpose import Pitch_sign_list  # noqa: F401  (kept for parity with legacy imports)
from abctoolkit.utils import Barline_regexPattern, Exclaim_re, Quote_re, SquareBracket_re  # noqa: F401

Note_list = _Note_list + ["z", "x"]


def complete_brackets(s):
    stack = []
    bracket_map = {"{": "}", "[": "]", "(": ")"}

    # Iterate through each character, handle bracket matching
    for char in s:
        if char in bracket_map:
            stack.append(char)
        elif char in bracket_map.values():
            # Find the corresponding left bracket
            for key, value in bracket_map.items():
                if value == char:
                    if stack and stack[-1] == key:
                        stack.pop()
                    break  # Found matching right bracket, process next character

    # Complete missing right brackets (in reverse order of remaining left brackets in stack)
    completion = "".join(bracket_map[c] for c in reversed(stack))
    return s + completion


def rest_unreduce(abc_lines):

    tunebody_index = None
    for i in range(len(abc_lines)):
        if abc_lines[i].startswith("%%score"):
            abc_lines[i] = complete_brackets(abc_lines[i])
        if "[V:" in abc_lines[i]:
            tunebody_index = i
            break

    metadata_lines = abc_lines[:tunebody_index]
    tunebody_lines = abc_lines[tunebody_index:]

    part_symbol_list = []
    voice_group_list = []
    for line in metadata_lines:
        if line.startswith("%%score"):
            for round_bracket_match in re.findall(r"\((.*?)\)", line):
                voice_group_list.append(round_bracket_match.split())
            existed_voices = [item for sublist in voice_group_list for item in sublist]
        if line.startswith("V:"):
            symbol = line.split()[0]
            part_symbol_list.append(symbol)
            if symbol[2:] not in existed_voices:
                voice_group_list.append([symbol[2:]])
    z_symbol_list = []  # voices that use z as rest
    x_symbol_list = []  # voices that use x as rest
    for voice_group in voice_group_list:
        z_symbol_list.append("V:" + voice_group[0])
        for j in range(1, len(voice_group)):
            x_symbol_list.append("V:" + voice_group[j])

    part_symbol_list.sort(key=lambda x: int(x[2:]))

    unreduced_tunebody_lines = []

    for i, line in enumerate(tunebody_lines):
        unreduced_line = ""

        line = re.sub(r"^\[r:[^\]]*\]", "", line)

        pattern = r"\[V:(\d+)\](.*?)(?=\[V:|$)"
        matches = re.findall(pattern, line)

        line_bar_dict = {}
        for match in matches:
            key = f"V:{match[0]}"
            value = match[1]
            line_bar_dict[key] = value

        # calculate duration and collect barline
        dur_dict = {}
        for symbol, bartext in line_bar_dict.items():
            right_barline = "".join(re.split(Barline_regexPattern, bartext)[-2:])
            bartext = bartext[: -len(right_barline)]
            try:
                bar_dur = calculate_bartext_duration(bartext)
            except:
                bar_dur = None
            if bar_dur is not None:
                if bar_dur not in dur_dict.keys():
                    dur_dict[bar_dur] = 1
                else:
                    dur_dict[bar_dur] += 1

        try:
            ref_dur = max(dur_dict, key=dur_dict.get)
        except:
            pass  # use last ref_dur

        if i == 0:
            prefix_left_barline = line.split("[V:")[0]
        else:
            prefix_left_barline = ""

        for symbol in part_symbol_list:
            if symbol in line_bar_dict.keys():
                symbol_bartext = line_bar_dict[symbol]
            else:
                if symbol in z_symbol_list:
                    symbol_bartext = prefix_left_barline + "z" + str(ref_dur) + right_barline
                elif symbol in x_symbol_list:
                    symbol_bartext = prefix_left_barline + "x" + str(ref_dur) + right_barline
            unreduced_line += "[" + symbol + "]" + symbol_bartext

        unreduced_tunebody_lines.append(unreduced_line + "\n")

    unreduced_lines = metadata_lines + unreduced_tunebody_lines

    return unreduced_lines


def strip_prompt_and_number(unreduced_lines: list[str]) -> str:
    """Demo output form: drop the ``%period``/``%composer``/... comment lines and add ``X:1``."""
    lines = [l for l in unreduced_lines if not (l.startswith("%") and not l.startswith("%%"))]
    return "".join(["X:1\n"] + lines)


def to_lines(abc_text: str) -> list[str]:
    return [line + "\n" for line in filter(None, abc_text.split("\n"))]
