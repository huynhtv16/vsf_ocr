# Copyright (c) Opendatalab. All rights reserved.
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from bs4 import BeautifulSoup

from vsf.backend.vlm.vlm_middle_json_mkcontent import merge_para_with_text
from vsf.utils.char_utils import full_to_half
from vsf.utils.enum_class import BlockType, SplitFlag
from vsf.utils.table_continuation import is_table_continuation_text


MAX_HEADER_ROWS = 5


@dataclass
class RowMetrics:
    row_idx: int
    effective_cols: int
    actual_cols: int
    visual_cols: int


@dataclass
class RowSignature:
    effective_cols: int
    colspans: tuple[int, ...]
    rowspans: tuple[int, ...]
    normalized_texts: tuple[str, ...]
    display_texts: tuple[str, ...]

    @property
    def cell_count(self) -> int:
        return len(self.colspans)


@dataclass
class RenderedCellSegment:
    text: str
    start_col: int
    end_col: int


@dataclass
class RowScanResult:
    row_effective_cols: list[int]
    row_metrics: list[RowMetrics]
    total_cols: int
    last_nonempty_row_metrics: RowMetrics | None
    tail_occupied: dict[int, set[int]]


@dataclass
class TableMergeState:
    owner_block: dict[str, Any]
    body_span: dict[str, Any]
    soup: Any
    tbody: Any
    rows: list[Any]
    total_cols: int
    front_header_info: list[RowSignature]
    front_first_data_row_metrics: dict[int, RowMetrics]
    last_data_row_metrics: RowMetrics | None
    row_effective_cols: list[int]
    tail_occupied: dict[int, set[int]]
    dirty: bool = False


def _normalize_cell_text(cell) -> str:
    return "".join(full_to_half(cell.get_text()).split())


def _display_cell_text(cell) -> str:
    return full_to_half(cell.get_text().strip())


def _scan_rows(rows, initial_occupied: dict[int, set[int]] | None = None, start_row_idx: int = 0) -> RowScanResult:
    """Scan rows once and cache effective-column metrics.

    initial_occupied stores future-row occupancy relative to the first scanned row
    and preserves rowspans that cross a merge boundary.
    """
    occupied: dict[int, dict[int, bool]] = {}
    max_cols = 0

    for row_offset, cols in (initial_occupied or {}).items():
        if not cols:
            continue
        occupied[row_offset] = {col: True for col in cols}
        max_cols = max(max_cols, max(cols) + 1)

    row_effective_cols: list[int] = []
    row_metrics: list[RowMetrics] = []
    last_nonempty_row_metrics: RowMetrics | None = None

    for local_idx, row in enumerate(rows):
        occupied_row = occupied.setdefault(local_idx, {})
        col_idx = 0
        cells = row.find_all(["td", "th"])
        actual_cols = 0

        for cell in cells:
            while col_idx in occupied_row:
                col_idx += 1

            colspan = int(cell.get("colspan", 1))
            rowspan = int(cell.get("rowspan", 1))
            actual_cols += colspan

            for row_offset in range(rowspan):
                target_idx = local_idx + row_offset
                occupied_target = occupied.setdefault(target_idx, {})
                for col in range(col_idx, col_idx + colspan):
                    occupied_target[col] = True

            col_idx += colspan
            max_cols = max(max_cols, col_idx)

        effective_cols = max(occupied_row.keys()) + 1 if occupied_row else 0
        row_effective_cols.append(effective_cols)
        max_cols = max(max_cols, effective_cols)

        metrics = RowMetrics(
            row_idx=start_row_idx + local_idx,
            effective_cols=effective_cols,
            actual_cols=actual_cols,
            visual_cols=len(cells),
        )
        row_metrics.append(metrics)
        if cells:
            last_nonempty_row_metrics = metrics

    tail_occupied = {
        row_idx - len(rows): set(cols.keys())
        for row_idx, cols in occupied.items()
        if row_idx >= len(rows) and cols
    }

    return RowScanResult(
        row_effective_cols=row_effective_cols,
        row_metrics=row_metrics,
        total_cols=max_cols,
        last_nonempty_row_metrics=last_nonempty_row_metrics,
        tail_occupied=tail_occupied,
    )


def _build_row_signature(row, effective_cols: int) -> RowSignature:
    cells = row.find_all(["td", "th"])
    return RowSignature(
        effective_cols=effective_cols,
        colspans=tuple(int(cell.get("colspan", 1)) for cell in cells),
        rowspans=tuple(int(cell.get("rowspan", 1)) for cell in cells),
        normalized_texts=tuple(_normalize_cell_text(cell) for cell in cells),
        display_texts=tuple(_display_cell_text(cell) for cell in cells),
    )


def _build_front_cache(rows, max_header_rows: int = MAX_HEADER_ROWS) -> tuple[list[RowSignature], dict[int, RowMetrics]]:
    front_limit = min(len(rows), max_header_rows + 1)
    front_rows = rows[:front_limit]
    front_scan = _scan_rows(front_rows)

    front_header_info = [
        _build_row_signature(front_rows[idx], front_scan.row_effective_cols[idx])
        for idx in range(min(len(front_rows), max_header_rows))
    ]
    front_first_data_row_metrics = {
        idx: metrics for idx, metrics in enumerate(front_scan.row_metrics)
    }
    return front_header_info, front_first_data_row_metrics


def _find_table_body_block(table_block):
    """Match the expected pattern."""
    for block in table_block["blocks"]:
        if block["type"] == BlockType.TABLE_BODY:
            return block
    return None


def _build_post_body_child_index(table_block, offset: int) -> int | float | None:
    """Build the required output."""
    body_block = _find_table_body_block(table_block)
    if body_block is None:
        return None

    body_index = body_block.get("index")
    if not isinstance(body_index, (int, float)):
        return None

    return body_index + offset


def _find_table_body_span(table_block):
    body_block = _find_table_body_block(table_block)
    if body_block and body_block["lines"] and body_block["lines"][0]["spans"]:
        return body_block["lines"][0]["spans"][0]
    return None


def _is_continuation_caption(caption_block) -> bool:
    """Validate the current value."""
    return is_table_continuation_text(merge_para_with_text(caption_block))


def _is_post_table_non_continuation_caption(table_block, caption_block) -> bool:
    """Validate the current value.

    Implementation detail.
    Merge the related values.
    """
    if _is_continuation_caption(caption_block):
        return False

    body_block = _find_table_body_block(table_block)
    if body_block is None:
        return False

    body_bbox = body_block.get("bbox")
    caption_bbox = caption_block.get("bbox")
    if not body_bbox or not caption_bbox:
        return False

    return caption_bbox[1] >= body_bbox[3]


def _get_post_table_caption_blocks(table_block):
    """Process table content."""
    return [
        block for block in table_block["blocks"]
        if block["type"] == BlockType.TABLE_CAPTION
        and _is_post_table_non_continuation_caption(table_block, block)
    ]


def _restore_post_table_captions_as_text(page_info, table_block, caption_blocks) -> None:
    """Process table content."""
    if not caption_blocks:
        return

    para_blocks = page_info.get("para_blocks", [])
    try:
        insert_idx = para_blocks.index(table_block) + 1
    except ValueError:
        return

    restored_blocks = []
    for caption_block in caption_blocks:
        text_block = deepcopy(caption_block)
        text_block["type"] = BlockType.TEXT
        restored_blocks.append(text_block)

    para_blocks[insert_idx:insert_idx] = restored_blocks
    restored_caption_ids = {id(block) for block in caption_blocks}
    table_block["blocks"] = [
        block for block in table_block["blocks"]
        if id(block) not in restored_caption_ids
    ]


def _refresh_table_state_metrics(state: TableMergeState) -> None:
    scan = _scan_rows(state.rows)
    state.row_effective_cols = scan.row_effective_cols
    state.total_cols = scan.total_cols
    state.last_data_row_metrics = scan.last_nonempty_row_metrics
    state.tail_occupied = scan.tail_occupied
    state.front_header_info, state.front_first_data_row_metrics = _build_front_cache(state.rows)


def build_table_state_from_html(
    html: str,
    max_header_rows: int = MAX_HEADER_ROWS,
) -> TableMergeState | None:
    """Build the required output.

    Process table content.
    Prepare the output value.
    """
    if not html:
        return None

    soup = BeautifulSoup(html, "html.parser")
    tbody = soup.find("tbody") or soup.find("table")
    rows = soup.find_all("tr")
    if not rows:
        return None

    scan = _scan_rows(rows)
    front_header_info, front_first_data_row_metrics = _build_front_cache(rows, max_header_rows=max_header_rows)

    return TableMergeState(
        owner_block={},
        body_span={},
        soup=soup,
        tbody=tbody,
        rows=rows,
        total_cols=scan.total_cols,
        front_header_info=front_header_info,
        front_first_data_row_metrics=front_first_data_row_metrics,
        last_data_row_metrics=scan.last_nonempty_row_metrics,
        row_effective_cols=scan.row_effective_cols,
        tail_occupied=scan.tail_occupied,
    )


def _build_table_state(table_block, max_header_rows: int = MAX_HEADER_ROWS) -> TableMergeState | None:
    body_span = _find_table_body_span(table_block)
    if body_span is None:
        return None

    html = body_span.get("html", "")
    if not html:
        return None

    soup = BeautifulSoup(html, "html.parser")
    tbody = soup.find("tbody") or soup.find("table")
    rows = soup.find_all("tr")
    scan = _scan_rows(rows)
    front_header_info, front_first_data_row_metrics = _build_front_cache(rows, max_header_rows=max_header_rows)

    return TableMergeState(
        owner_block=table_block,
        body_span=body_span,
        soup=soup,
        tbody=tbody,
        rows=rows,
        total_cols=scan.total_cols,
        front_header_info=front_header_info,
        front_first_data_row_metrics=front_first_data_row_metrics,
        last_data_row_metrics=scan.last_nonempty_row_metrics,
        row_effective_cols=scan.row_effective_cols,
        tail_occupied=scan.tail_occupied,
    )


def _get_or_create_table_state(
    table_block,
    state_cache: dict[int, TableMergeState],
    max_header_rows: int = MAX_HEADER_ROWS,
) -> TableMergeState | None:
    cache_key = id(table_block)
    state = state_cache.get(cache_key)
    if state is not None:
        return state

    state = _build_table_state(table_block, max_header_rows=max_header_rows)
    if state is not None:
        state_cache[cache_key] = state
    return state


def _serialize_table_state_html(state: TableMergeState) -> None:
    state.body_span["html"] = str(state.soup)
    state.dirty = False


def calculate_table_total_columns(soup):
    """Parse the input data."""
    rows = soup.find_all("tr")
    return _scan_rows(rows).total_cols if rows else 0


def build_table_occupied_matrix(soup):
    """Build the required output."""
    rows = soup.find_all("tr")
    if not rows:
        return {}

    scan = _scan_rows(rows)
    return {
        row_idx: effective_cols
        for row_idx, effective_cols in enumerate(scan.row_effective_cols)
    }


def calculate_row_effective_columns(soup, row_idx):
    """Calculate the result."""
    row_effective_cols = build_table_occupied_matrix(soup)
    return row_effective_cols.get(row_idx, 0)


def calculate_row_columns(row):
    """Calculate the result."""
    cells = row.find_all(["td", "th"])
    column_count = 0

    for cell in cells:
        colspan = int(cell.get("colspan", 1))
        column_count += colspan

    return column_count


def calculate_visual_columns(row):
    """Calculate the result."""
    cells = row.find_all(["td", "th"])
    return len(cells)


def _scan_row_visual_sources(
    rows,
    target_row_index: int,
    initial_occupied: dict[int, set[int]] | None = None,
) -> tuple[dict[int, tuple[int, int]], int]:
    """Implementation detail.

    Implementation detail.
    Calculate the result.
    Implementation detail.
    """
    if target_row_index < 0:
        target_row_index += len(rows)
    if target_row_index < 0 or target_row_index >= len(rows):
        return {}, 0

    # occupied[row_idx][col_idx] = (source_row_idx, source_cell_idx)
    occupied: dict[int, dict[int, tuple[int, int]]] = {}
    total_cols = 0
    for row_offset, cols in (initial_occupied or {}).items():
        if not cols:
            continue
        occupied[row_offset] = {
            col: (-1, col) for col in cols
        }
        total_cols = max(total_cols, max(cols) + 1)

    for r_idx in range(target_row_index + 1):
        occupied_row = occupied.setdefault(r_idx, {})
        col_idx = 0
        cells = rows[r_idx].find_all(["td", "th"])
        for cell_idx, cell in enumerate(cells):
            while col_idx in occupied_row:
                col_idx += 1
            colspan = int(cell.get("colspan", 1))
            rowspan = int(cell.get("rowspan", 1))
            source_marker = (r_idx, cell_idx)
            for ro in range(rowspan):
                target_idx = r_idx + ro
                occ = occupied.setdefault(target_idx, {})
                for c in range(col_idx, col_idx + colspan):
                    occ[c] = source_marker
            col_idx += colspan
            total_cols = max(total_cols, col_idx)

    return occupied.get(target_row_index, {}), total_cols


def build_visual_col_mapping(
    rows,
    target_row_index: int,
    initial_occupied: dict[int, set[int]] | None = None,
) -> list[int]:
    """Build the required output.

    Implementation detail.
    Implementation detail.
    """
    if target_row_index < 0:
        target_row_index += len(rows)
    if target_row_index < 0 or target_row_index >= len(rows):
        return []

    target_occupied, _ = _scan_row_visual_sources(
        rows,
        target_row_index,
        initial_occupied=initial_occupied,
    )

    col_idx = 0
    mapping = []
    target_cells = rows[target_row_index].find_all(["td", "th"])
    for cell in target_cells:
        while col_idx in target_occupied and target_occupied[col_idx][0] < target_row_index:
            col_idx += 1
        mapping.append(col_idx)
        colspan = int(cell.get("colspan", 1))
        col_idx += colspan
    return mapping


def build_row_rendered_cell_segments(
    rows,
    target_row_index: int,
    initial_occupied: dict[int, set[int]] | None = None,
) -> list[RenderedCellSegment]:
    """Build the required output.

    Process table content.
    Implementation detail.
    Prepare the output value.
    """
    if target_row_index < 0:
        target_row_index += len(rows)
    if target_row_index < 0 or target_row_index >= len(rows):
        return []

    target_occupied, total_cols = _scan_row_visual_sources(
        rows,
        target_row_index,
        initial_occupied=initial_occupied,
    )
    if total_cols == 0:
        return []

    segments: list[RenderedCellSegment] = []
    current_marker: tuple[int, int] | None = None
    current_start_col: int | None = None
    current_text = ""

    # Merge the related values.
    for col_idx in range(total_cols):
        marker = target_occupied.get(col_idx)
        if marker is None:
            if current_marker is not None and current_start_col is not None:
                segments.append(RenderedCellSegment(text=current_text, start_col=current_start_col, end_col=col_idx))
            current_marker = None
            current_start_col = None
            current_text = ""
            continue

        if marker != current_marker:
            if current_marker is not None and current_start_col is not None:
                segments.append(RenderedCellSegment(text=current_text, start_col=current_start_col, end_col=col_idx))
            current_marker = marker
            current_start_col = col_idx
            source_row_idx, source_cell_idx = marker
            current_text = ""
            if source_row_idx >= 0:
                source_cells = rows[source_row_idx].find_all(["td", "th"])
                if source_cell_idx < len(source_cells):
                    current_text = _display_cell_text(source_cells[source_cell_idx])

    if current_marker is not None and current_start_col is not None:
        segments.append(RenderedCellSegment(text=current_text, start_col=current_start_col, end_col=total_cols))

    return segments


def calculate_row_rendered_segments(rows, target_row_index: int) -> int:
    """Calculate the result.

    Calculate the result.
    Implementation detail.
    Implementation detail.
    Implementation detail.
    """
    target_occupied, total_cols = _scan_row_visual_sources(rows, target_row_index)
    if total_cols == 0:
        return 0

    segment_count = 0
    previous_marker: tuple[int, int] | None = None

    for col_idx in range(total_cols):
        marker = target_occupied.get(col_idx)
        if marker is None:
            previous_marker = None
            continue
        if marker != previous_marker:
            segment_count += 1
            previous_marker = marker

    return segment_count


def detect_table_headers(state1: TableMergeState, state2: TableMergeState, max_header_rows: int = MAX_HEADER_ROWS):
    """Process table content."""
    front_rows1 = state1.front_header_info[:max_header_rows]
    front_rows2 = state2.front_header_info[:max_header_rows]

    min_rows = min(len(front_rows1), len(front_rows2), max_header_rows)
    header_rows = 0
    headers_match = True
    header_texts = []

    for row_idx in range(min_rows):
        row1 = front_rows1[row_idx]
        row2 = front_rows2[row_idx]
        structure_match = (
            row1.cell_count == row2.cell_count
            and row1.effective_cols == row2.effective_cols
            and row1.colspans == row2.colspans
            and row1.rowspans == row2.rowspans
            and row1.normalized_texts == row2.normalized_texts
        )

        if structure_match:
            header_rows += 1
            header_texts.append(list(row1.display_texts))
        else:
            headers_match = header_rows > 0
            break

    if header_rows == 0:
        header_rows, headers_match, header_texts = _detect_table_headers_visual(
            state1, state2, max_header_rows=max_header_rows
        )

    return header_rows, headers_match, header_texts


def _detect_table_headers_visual(
    state1: TableMergeState,
    state2: TableMergeState,
    max_header_rows: int = MAX_HEADER_ROWS,
):
    """Process text content."""
    front_rows1 = state1.front_header_info[:max_header_rows]
    front_rows2 = state2.front_header_info[:max_header_rows]

    min_rows = min(len(front_rows1), len(front_rows2), max_header_rows)
    header_rows = 0
    headers_match = True
    header_texts = []

    for row_idx in range(min_rows):
        row1 = front_rows1[row_idx]
        row2 = front_rows2[row_idx]
        # Implementation detail.
        rendered_segments1 = calculate_row_rendered_segments(state1.rows, row_idx)
        rendered_segments2 = calculate_row_rendered_segments(state2.rows, row_idx)
        if row1.normalized_texts == row2.normalized_texts and rendered_segments1 == rendered_segments2:
            header_rows += 1
            header_texts.append(list(row1.display_texts))
        else:
            headers_match = header_rows > 0
            break

    if header_rows == 0:
        headers_match = False

    return header_rows, headers_match, header_texts


def _expand_header_count_by_rowspan(rows, header_count: int) -> int:
    """Remove invalid or unnecessary data.

    Remove invalid or unnecessary data.
    Merge the related values.
    Remove invalid or unnecessary data.
    """
    if header_count <= 0 or not rows:
        return header_count

    expanded_header_count = min(header_count, len(rows))
    row_idx = 0
    while row_idx < expanded_header_count:
        row = rows[row_idx]
        for cell in row.find_all(["td", "th"]):
            rowspan = int(cell.get("rowspan", 1))
            if rowspan > 1:
                expanded_header_count = max(expanded_header_count, row_idx + rowspan)
                expanded_header_count = min(expanded_header_count, len(rows))
        row_idx += 1

    return expanded_header_count


def can_merge_by_structure(
    current_state: TableMergeState,
    previous_state: TableMergeState,
    current_bbox: tuple[float, float, float, float] | None = None,
    previous_bbox: tuple[float, float, float, float] | None = None,
) -> bool:
    """Validate the current value.

    Validate the current value.
    """
    if current_bbox is not None and previous_bbox is not None:
        x0_t1, _, x1_t1, _ = current_bbox
        x0_t2, _, x1_t2, _ = previous_bbox
        table1_width = x1_t1 - x0_t1
        table2_width = x1_t2 - x0_t2
        if table1_width > 0 and table2_width > 0:
            if abs(table1_width - table2_width) / min(table1_width, table2_width) >= 0.1:
                return False

    if previous_state.total_cols == current_state.total_cols:
        return True

    return check_rows_match(previous_state, current_state)


def can_merge_tables(current_state: TableMergeState, previous_state: TableMergeState):
    """Validate the current value."""
    current_table_block = current_state.owner_block
    previous_table_block = previous_state.owner_block

    if "blocks" not in previous_table_block or "blocks" not in current_table_block:
        raise ValueError(
            "can_merge_tables() requires owner_block with 'blocks' key. "
            "For HTML-only states from build_table_state_from_html(), use can_merge_by_structure() instead."
        )

    footnote_count = sum(
        1 for block in previous_table_block["blocks"] if block["type"] == BlockType.TABLE_FOOTNOTE
    )
    caption_blocks = [
        block for block in current_table_block["blocks"] if block["type"] == BlockType.TABLE_CAPTION
    ]
    merge_caption_blocks = [
        block for block in caption_blocks
        if not _is_post_table_non_continuation_caption(current_table_block, block)
    ]
    if merge_caption_blocks:
        has_continuation_marker = any(
            _is_continuation_caption(block) for block in merge_caption_blocks
        )

        if not has_continuation_marker:
            return False

        if footnote_count > 1:
            return False
    elif footnote_count > 0:
        return False

    x0_t1, _, x1_t1, _ = current_table_block["bbox"]
    x0_t2, _, x1_t2, _ = previous_table_block["bbox"]
    table1_width = x1_t1 - x0_t1
    table2_width = x1_t2 - x0_t2

    if abs(table1_width - table2_width) / min(table1_width, table2_width) >= 0.1:
        return False

    if previous_state.total_cols == current_state.total_cols:
        return True

    return check_rows_match(previous_state, current_state)


def check_rows_match(previous_state: TableMergeState, current_state: TableMergeState):
    """Validate the current value."""
    last_row_metrics = previous_state.last_data_row_metrics
    if last_row_metrics is None:
        return False

    header_count, _, _ = detect_table_headers(previous_state, current_state)
    header_count = _expand_header_count_by_rowspan(current_state.rows, header_count)
    first_data_row_metrics = current_state.front_first_data_row_metrics.get(header_count)
    if first_data_row_metrics is None:
        return False

    previous_rendered_segments = calculate_row_rendered_segments(previous_state.rows, last_row_metrics.row_idx)
    current_rendered_segments = calculate_row_rendered_segments(current_state.rows, first_data_row_metrics.row_idx)

    return (
        last_row_metrics.effective_cols == first_data_row_metrics.effective_cols
        or last_row_metrics.actual_cols == first_data_row_metrics.actual_cols
        or previous_rendered_segments == current_rendered_segments
    )


def check_row_columns_match(row1, row2):
    cells1 = row1.find_all(["td", "th"])
    cells2 = row2.find_all(["td", "th"])
    if len(cells1) != len(cells2):
        return False
    for cell1, cell2 in zip(cells1, cells2):
        colspan1 = int(cell1.get("colspan", 1))
        colspan2 = int(cell2.get("colspan", 1))
        if colspan1 != colspan2:
            return False
    return True


def adjust_table_rows_colspan(
    rows,
    start_idx,
    end_idx,
    row_effective_cols,
    reference_structure,
    reference_visual_cols,
    target_cols,
    match_reference_row,
):
    """Match the expected pattern."""
    reference_row_copy = deepcopy(match_reference_row)

    for row_idx in range(start_idx, end_idx):
        row = rows[row_idx]
        cells = row.find_all(["td", "th"])
        if not cells:
            continue

        current_row_effective_cols = row_effective_cols[row_idx]
        current_row_cols = calculate_row_columns(row)

        if current_row_effective_cols >= target_cols or current_row_cols >= target_cols:
            continue

        if (
            calculate_visual_columns(row) == reference_visual_cols
            and check_row_columns_match(row, reference_row_copy)
        ):
            if len(cells) <= len(reference_structure):
                for cell_idx, cell in enumerate(cells):
                    if cell_idx < len(reference_structure) and reference_structure[cell_idx] > 1:
                        cell["colspan"] = str(reference_structure[cell_idx])
        else:
            cols_diff = target_cols - current_row_effective_cols
            if cols_diff > 0:
                last_cell = cells[-1]
                current_last_span = int(last_cell.get("colspan", 1))
                last_cell["colspan"] = str(current_last_span + cols_diff)


def _cell_has_semantic_content(cell) -> bool:
    """Validate the current value."""
    if cell.get_text(strip=True):
        return True

    return (
        cell.find(["img", "svg", "math", "eq", "table", "figure", "object", "embed", "canvas"])
        is not None
    )


def _row_has_semantic_content(row) -> bool:
    """Validate the current value."""
    return any(_cell_has_semantic_content(cell) for cell in row.find_all(["td", "th"]))


def _insert_cell_before_visual_column(rows, target_row_index: int, start_vcol: int, cell) -> None:
    """Add the value to the result."""
    target_row = rows[target_row_index]
    target_cells = target_row.find_all(["td", "th"])
    target_vcol_map = build_visual_col_mapping(rows, target_row_index)

    for idx, target_start_vcol in enumerate(target_vcol_map):
        if target_start_vcol >= start_vcol:
            target_cells[idx].insert_before(cell)
            return

    target_row.append(cell)


def _carry_rowspan_structure_to_next_row(rows, row_idx: int) -> None:
    """Remove invalid or unnecessary data."""
    next_row_idx = row_idx + 1
    if next_row_idx >= len(rows):
        return

    current_row = rows[row_idx]
    current_cells = current_row.find_all(["td", "th"])
    current_vcol_map = build_visual_col_mapping(rows, row_idx)
    carried_cells = []

    for cell, start_vcol in zip(current_cells, current_vcol_map):
        rowspan = int(cell.get("rowspan", 1))
        if rowspan <= 1 or _cell_has_semantic_content(cell):
            continue

        carried_cell = deepcopy(cell)
        new_rowspan = rowspan - 1
        if new_rowspan > 1:
            carried_cell["rowspan"] = str(new_rowspan)
        else:
            carried_cell.attrs.pop("rowspan", None)
        carried_cells.append((start_vcol, carried_cell))

    for start_vcol, carried_cell in sorted(carried_cells, key=lambda item: item[0], reverse=True):
        _insert_cell_before_visual_column(rows, next_row_idx, start_vcol, carried_cell)


def _clip_overlapped_blank_rowspan_cells(
    rows,
    initial_occupied: dict[int, set[int]],
) -> bool:
    """Implementation detail.

    Process table content.
    Build the required output.
    Merge the related values.
    Process the current item.
    """
    if not rows or not initial_occupied:
        return False

    cells_to_remove = []
    cells_to_move = []

    for row_idx, row in enumerate(rows):
        cells = row.find_all(["td", "th"])
        visual_col_map = build_visual_col_mapping(rows, row_idx)
        for cell, start_vcol in zip(cells, visual_col_map):
            rowspan = int(cell.get("rowspan", 1))
            if rowspan <= 1 or _cell_has_semantic_content(cell):
                continue

            colspan = int(cell.get("colspan", 1))
            occupied_cols = set(range(start_vcol, start_vcol + colspan))
            if not occupied_cols:
                continue

            overlap_rows = 0
            while overlap_rows < rowspan:
                covered_cols = initial_occupied.get(row_idx + overlap_rows, set())
                if not occupied_cols.issubset(covered_cols):
                    break
                overlap_rows += 1

            if overlap_rows == 0:
                continue

            remaining_rowspan = rowspan - overlap_rows
            target_row_idx = row_idx + overlap_rows
            if remaining_rowspan > 0 and target_row_idx >= len(rows):
                continue

            cells_to_remove.append(cell)
            if remaining_rowspan > 0:
                moved_cell = deepcopy(cell)
                if remaining_rowspan > 1:
                    moved_cell["rowspan"] = str(remaining_rowspan)
                else:
                    moved_cell.attrs.pop("rowspan", None)
                cells_to_move.append((target_row_idx, start_vcol, moved_cell))

    if not cells_to_remove:
        return False

    for cell in cells_to_remove:
        cell.extract()

    for target_row_idx, start_vcol, moved_cell in sorted(
        cells_to_move,
        key=lambda item: (item[0], item[1]),
        reverse=True,
    ):
        _insert_cell_before_visual_column(rows, target_row_idx, start_vcol, moved_cell)

    return True


def _apply_cell_merge(
    previous_state: TableMergeState,
    current_state: TableMergeState,
    header_count: int,
) -> None:
    """Merge the related values.

    Implementation detail.
    Remove invalid or unnecessary data.
    Merge the related values.

    Build the required output.
    Process table content.
    """
    cell_merge = current_state.owner_block.get("cell_merge")
    if not cell_merge:
        return

    rows2 = current_state.rows
    if header_count >= len(rows2):
        return
    if not previous_state.rows:
        return

    first_data_row = rows2[header_count]
    last_row = previous_state.rows[-1]

    cells1 = last_row.find_all(["td", "th"])
    cells2 = first_data_row.find_all(["td", "th"])

    # Build the required output.
    last_row_idx = len(previous_state.rows) - 1
    vcol_map1 = build_visual_col_mapping(previous_state.rows, last_row_idx)
    current_merge_rows = rows2[header_count:]
    vcol_map2 = build_visual_col_mapping(
        current_merge_rows,
        0,
        initial_occupied=previous_state.tail_occupied,
    )

    # Build the required output.
    vcol_to_cell1: dict[int, int] = {}
    for ci, start_vcol in enumerate(vcol_map1):
        colspan = int(cells1[ci].get("colspan", 1))
        for c in range(start_vcol, start_vcol + colspan):
            vcol_to_cell1[c] = ci
    vcol_to_cell2: dict[int, int] = {}
    for ci, start_vcol in enumerate(vcol_map2):
        colspan = int(cells2[ci].get("colspan", 1))
        for c in range(start_vcol, start_vcol + colspan):
            vcol_to_cell2[c] = ci

    # Process the current item.
    transferred_pairs: set[tuple[int, int]] = set()
    for vi, merge_flag in enumerate(cell_merge):
        if merge_flag == 1:
            ci1 = vcol_to_cell1.get(vi)
            ci2 = vcol_to_cell2.get(vi)
            if ci1 is not None and ci2 is not None:
                pair = (ci1, ci2)
                if pair not in transferred_pairs:
                    for child in list(cells2[ci2].children):
                        cells1[ci1].append(child.extract())
                    transferred_pairs.add(pair)

    # Implementation detail.
    cleared_ci2: set[int] = set()
    for vi, merge_flag in enumerate(cell_merge):
        if merge_flag == 1:
            ci1 = vcol_to_cell1.get(vi)
            ci2 = vcol_to_cell2.get(vi)
            if ci1 is not None and ci2 is not None and ci2 not in cleared_ci2:
                cells2[ci2].clear()
                cleared_ci2.add(ci2)

    if not _row_has_semantic_content(first_data_row):
        _carry_rowspan_structure_to_next_row(rows2, header_count)
        first_data_row.extract()
        if first_data_row in rows2:
            rows2.remove(first_data_row)


def perform_table_merge(
    previous_state: TableMergeState,
    current_state: TableMergeState,
    previous_table_block,
    wait_merge_table_footnotes,
):
    """Merge the related values."""
    header_count, _, _ = detect_table_headers(previous_state, current_state)
    header_count = _expand_header_count_by_rowspan(current_state.rows, header_count)

    rows1 = previous_state.rows
    rows2 = current_state.rows

    previous_adjusted = False

    if header_count < len(rows2):
        current_merge_rows = rows2[header_count:]
        if _clip_overlapped_blank_rowspan_cells(current_merge_rows, previous_state.tail_occupied):
            _refresh_table_state_metrics(current_state)

    if rows1 and rows2 and header_count < len(rows2):
        last_row1 = rows1[-1]
        first_data_row2 = rows2[header_count]
        table_cols1 = previous_state.total_cols
        table_cols2 = current_state.total_cols

        if table_cols1 > table_cols2:
            reference_structure = [
                int(cell.get("colspan", 1)) for cell in last_row1.find_all(["td", "th"])
            ]
            reference_visual_cols = calculate_visual_columns(last_row1)
            adjust_table_rows_colspan(
                rows2,
                header_count,
                len(rows2),
                current_state.row_effective_cols,
                reference_structure,
                reference_visual_cols,
                table_cols1,
                first_data_row2,
            )
        elif table_cols2 > table_cols1:
            reference_structure = [
                int(cell.get("colspan", 1)) for cell in first_data_row2.find_all(["td", "th"])
            ]
            reference_visual_cols = calculate_visual_columns(first_data_row2)
            adjust_table_rows_colspan(
                rows1,
                0,
                len(rows1),
                previous_state.row_effective_cols,
                reference_structure,
                reference_visual_cols,
                table_cols2,
                last_row1,
            )
            previous_adjusted = True

    if previous_adjusted:
        _refresh_table_state_metrics(previous_state)

    _apply_cell_merge(previous_state, current_state, header_count)

    appended_rows = rows2[header_count:]
    append_start_idx = len(previous_state.rows)
    merged_rows = []

    if previous_state.tbody and current_state.tbody:
        for row in appended_rows:
            row.extract()
            previous_state.tbody.append(row)
            merged_rows.append(row)

    previous_state.rows.extend(merged_rows)

    if merged_rows:
        appended_scan = _scan_rows(
            merged_rows,
            initial_occupied=previous_state.tail_occupied,
            start_row_idx=append_start_idx,
        )
        previous_state.row_effective_cols.extend(appended_scan.row_effective_cols)
        previous_state.total_cols = max(previous_state.total_cols, appended_scan.total_cols)
        if appended_scan.last_nonempty_row_metrics is not None:
            previous_state.last_data_row_metrics = appended_scan.last_nonempty_row_metrics
        previous_state.tail_occupied = appended_scan.tail_occupied

    previous_table_block["blocks"] = [
        block for block in previous_table_block["blocks"] if block["type"] != BlockType.TABLE_FOOTNOTE
    ]
    for footnote_offset, table_footnote in enumerate(wait_merge_table_footnotes, start=1):
        temp_table_footnote = table_footnote.copy()
        temp_table_footnote[SplitFlag.CROSS_PAGE] = True
        post_body_index = _build_post_body_child_index(previous_table_block, footnote_offset)
        if post_body_index is None:
            temp_table_footnote.pop("index", None)
        else:
            temp_table_footnote["index"] = post_body_index
        previous_table_block["blocks"].append(temp_table_footnote)

    previous_state.dirty = True


def merge_table(page_info_list):
    """Merge the related values."""
    state_cache: dict[int, TableMergeState] = {}
    merged_away_blocks: set[int] = set()

    for page_idx in range(len(page_info_list) - 1, -1, -1):
        if page_idx == 0:
            continue

        page_info = page_info_list[page_idx]
        previous_page_info = page_info_list[page_idx - 1]

        if not (page_info["para_blocks"] and page_info["para_blocks"][0]["type"] == BlockType.TABLE):
            continue

        if not (
            previous_page_info["para_blocks"]
            and previous_page_info["para_blocks"][-1]["type"] == BlockType.TABLE
        ):
            continue

        current_table_block = page_info["para_blocks"][0]
        previous_table_block = previous_page_info["para_blocks"][-1]

        current_state = _get_or_create_table_state(current_table_block, state_cache)
        previous_state = _get_or_create_table_state(previous_table_block, state_cache)
        if current_state is None or previous_state is None:
            continue

        post_table_caption_blocks = _get_post_table_caption_blocks(current_table_block)
        wait_merge_table_footnotes = [
            block for block in current_table_block["blocks"] if block["type"] == BlockType.TABLE_FOOTNOTE
        ]

        if not can_merge_tables(current_state, previous_state):
            continue

        perform_table_merge(
            previous_state,
            current_state,
            previous_table_block,
            wait_merge_table_footnotes,
        )
        _restore_post_table_captions_as_text(
            page_info,
            current_table_block,
            post_table_caption_blocks,
        )

        merged_away_blocks.add(id(current_table_block))
        for block in current_table_block["blocks"]:
            block["lines"] = []
            block[SplitFlag.LINES_DELETED] = True

    for state in state_cache.values():
        if state.dirty and id(state.owner_block) not in merged_away_blocks:
            _serialize_table_state_html(state)
