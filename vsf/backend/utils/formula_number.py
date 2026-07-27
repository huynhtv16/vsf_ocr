# Copyright (c) Opendatalab. All rights reserved.
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from mineru.utils.boxbase import (
    calculate_overlap_area_2_minbox_area_ratio,
    calculate_overlap_area_in_bbox1_area_ratio,
)
from mineru.utils.char_utils import full_to_half
from mineru.utils.enum_class import BlockType, ContentType
from mineru.utils.visual_magic_model_utils import isolated_formula_clean

Block = dict[str, Any]


def formula_number_max_overlap_ratio(span: Block, block_bbox: Sequence[float]) -> float:
    """Process formula content."""
    return max(
        calculate_overlap_area_in_bbox1_area_ratio(span["bbox"], block_bbox),
        calculate_overlap_area_2_minbox_area_ratio(span["bbox"], block_bbox),
    )


def extract_formula_number_text(block: Block) -> str:
    """Extract the required value."""
    content = block.get("content")
    if isinstance(content, str) and content.strip():
        return content.strip()

    text_parts = []
    for line in block.get("lines", []):
        for span in line.get("spans", []):
            if span.get("type") == ContentType.TEXT:
                text_parts.append(span.get("content", ""))
    return "".join(text_parts).strip()


def normalize_formula_tag_content(tag_content: str) -> str:
    """Convert the value to the required format."""
    tag_content = full_to_half(tag_content.strip())
    if tag_content.startswith("("):
        tag_content = tag_content[1:].strip()
    if tag_content.endswith(")"):
        tag_content = tag_content[:-1].strip()
    return tag_content


def _normalize_formula_content_for_tag(formula_content: str) -> str:
    """Merge the related values."""
    return isolated_formula_clean(formula_content or "")


def build_tagged_formula_content(
    formula_content: str,
    formula_number_block: Block,
) -> str | None:
    """Process formula content."""
    formula_content = _normalize_formula_content_for_tag(formula_content)
    if not formula_content:
        return None
    tag_content = normalize_formula_tag_content(
        extract_formula_number_text(formula_number_block)
    )
    return f"{formula_content}\\tag{{{tag_content}}}"


def get_interline_equation_span(block: Block) -> Block | None:
    """Match the expected pattern."""
    for line in block.get("lines", []):
        for span in line.get("spans", []):
            if span.get("type") == ContentType.INTERLINE_EQUATION:
                return span
    return None


def append_formula_number_tag(
    equation_block: Block,
    formula_number_block: Block,
) -> bool:
    """Merge the related values."""
    equation_span = get_interline_equation_span(equation_block)
    if equation_span is not None:
        tagged_content = build_tagged_formula_content(
            equation_span.get("content", ""),
            formula_number_block,
        )
        if tagged_content is None:
            return False
        equation_span["content"] = tagged_content
        return True
    return False


def _optimize_formula_number_sequence(
    blocks: Sequence[Block],
    is_formula_number: Callable[[Block], bool],
    is_equation: Callable[[Block], bool],
    append_tag: Callable[[Block, Block], bool],
    downgrade_block: Callable[[Block], None],
) -> list[Block]:
    """Process formula content."""
    optimized_blocks = []
    for index, block in enumerate(blocks):
        if not is_formula_number(block):
            optimized_blocks.append(block)
            continue

        prev_block = blocks[index - 1] if index > 0 else None
        if prev_block and is_equation(prev_block):
            if append_tag(prev_block, block):
                continue
            downgrade_block(block)
            optimized_blocks.append(block)
            continue

        next_block = blocks[index + 1] if index + 1 < len(blocks) else None
        next_next_block = blocks[index + 2] if index + 2 < len(blocks) else None
        if (
            next_block
            and is_equation(next_block)
            and (next_next_block is None or not is_formula_number(next_next_block))
        ):
            if append_tag(next_block, block):
                continue
            downgrade_block(block)
            optimized_blocks.append(block)
            continue

        downgrade_block(block)
        optimized_blocks.append(block)

    return optimized_blocks


def _downgrade_formula_number_to_text(block: Block) -> None:
    """Match the expected pattern."""
    block["type"] = BlockType.TEXT


def _append_hybrid_formula_number_tag(
    equation_block: Block,
    formula_number_block: Block,
) -> bool:
    """Merge the related values."""
    tagged_content = build_tagged_formula_content(
        equation_block.get("content", ""),
        formula_number_block,
    )
    if tagged_content is None:
        return False
    equation_block["content"] = tagged_content
    return True


def optimize_formula_number_blocks(pdf_info_list: Iterable[Block]) -> None:
    """Match the expected pattern."""
    for page_info in pdf_info_list:
        blocks = page_info.get("preproc_blocks", [])
        page_info["preproc_blocks"] = _optimize_formula_number_sequence(
            blocks,
            lambda block: block.get("type") == BlockType.FORMULA_NUMBER,
            lambda block: block.get("type") == BlockType.INTERLINE_EQUATION,
            append_formula_number_tag,
            _downgrade_formula_number_to_text,
        )


def optimize_hybrid_formula_number_blocks(model_list: Iterable[list[Block]]) -> None:
    """Process formula content."""
    for page_model_list in model_list:
        optimized_blocks = _optimize_formula_number_sequence(
            page_model_list or [],
            lambda block: block.get("type") == BlockType.FORMULA_NUMBER,
            lambda block: block.get("type") == BlockType.EQUATION,
            _append_hybrid_formula_number_tag,
            _downgrade_formula_number_to_text,
        )
        page_model_list[:] = optimized_blocks
