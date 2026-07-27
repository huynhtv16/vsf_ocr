# Copyright (c) Opendatalab. All rights reserved.
import collections
import math
import re
import statistics

import cv2
import numpy as np
from loguru import logger

from mineru.utils.boxbase import calculate_overlap_area_in_bbox1_area_ratio
from mineru.utils.enum_class import BlockType, ContentType
from mineru.utils.pdf_image_tools import get_crop_img
from mineru.utils.pdf_text_tool import get_lines_from_chars, get_page_chars
from mineru.utils.pdfium_guard import close_pdfium_child, pdfium_guard

MAX_NATIVE_TEXT_CHARS_PER_PAGE = 65535
PRIVATE_USE_AREA_START = 0xE000
PRIVATE_USE_AREA_END = 0xF8FF
PRIVATE_USE_TEXT_COUNT_THRESHOLD = 2
PRIVATE_USE_TEXT_RATIO_THRESHOLD = 0.05
PRIVATE_USE_TEXT_RUN_THRESHOLD = 2
POST_OCR_FALLBACK_CONTENT_KEY = '_post_ocr_fallback_content'
POST_OCR_FALLBACK_SCORE_KEY = '_post_ocr_fallback_score'
POST_OCR_REASON_KEY = '_post_ocr_reason'
POST_OCR_REASON_PRIVATE_USE_TEXT = 'private_use_text'


def __replace_ligatures(text: str):
    ligatures = {
        'ﬁ': 'fi', 'ﬂ': 'fl', 'ﬀ': 'ff', 'ﬃ': 'ffi', 'ﬄ': 'ffl', 'ﬅ': 'ft', 'ﬆ': 'st'
    }
    return re.sub('|'.join(map(re.escape, ligatures.keys())), lambda m: ligatures[m.group()], text)

def __replace_unicode(text: str):
    ligatures = {
        '\r\n': '', '\u0002': '-',
    }
    return re.sub('|'.join(map(re.escape, ligatures.keys())), lambda m: ligatures[m.group()], text)


"""pdf_text dict\u65b9\u6848 char\u7ea7\u522b"""
def txt_spans_extract(pdf_page, spans, pil_img, scale, all_bboxes, all_discarded_blocks):
    page_char_count = None
    textpage = None
    try:
        try:
            with pdfium_guard():
                textpage = pdf_page.get_textpage()
                page_char_count = textpage.count_chars()
        except Exception as exc:
            logger.debug(f"Failed to get page char count before txt extraction: {exc}")

        if page_char_count is not None and page_char_count > MAX_NATIVE_TEXT_CHARS_PER_PAGE:
            logger.info(
                "Fallback to post-OCR in txt_spans_extract due to high char count: "
                f"count_chars={page_char_count}"
            )
            need_ocr_spans = [
                span for span in spans if span.get('type') == ContentType.TEXT
            ]
            return _prepare_post_ocr_spans(need_ocr_spans, spans, pil_img, scale)

        page_chars = get_page_chars(
            pdf_page,
            textpage=textpage,
            page_char_count=page_char_count,
        )
        page_all_chars = _get_chars_for_span_fill(page_chars)

        # Calculate the result.
        span_height_list = []
        for span in spans:
            if span['type'] in [ContentType.TEXT]:
                span_height = span['bbox'][3] - span['bbox'][1]
                span['height'] = span_height
                span['width'] = span['bbox'][2] - span['bbox'][0]
                span_height_list.append(span_height)
        if len(span_height_list) == 0:
            return spans
        else:
            median_span_height = statistics.median(span_height_list)

        useful_spans = []
        unuseful_spans = []
        # Implementation detail.
        vertical_spans = []
        for span in spans:
            if span['type'] in [ContentType.TEXT]:
                for block in all_bboxes + all_discarded_blocks:
                    if block[7] in [BlockType.IMAGE_BODY, BlockType.TABLE_BODY, BlockType.INTERLINE_EQUATION]:
                        continue
                    if calculate_overlap_area_in_bbox1_area_ratio(span['bbox'], block[0:4]) > 0.5:
                        if span['height'] > median_span_height * 2.3 and span['height'] > span['width'] * 2.3:
                            vertical_spans.append(span)
                        elif block in all_bboxes:
                            useful_spans.append(span)
                        else:
                            unuseful_spans.append(span)
                        break

        """\u5782\u76f4\u7684span\u6846\u76f4\u63a5\u7528line\u8fdb\u884c\u586b\u5145"""
        if len(vertical_spans) > 0:
            page_all_lines = [
                line for line in get_lines_from_chars(page_chars['chars'])
                if _is_supported_rotation(line['rotation'])
            ]
            for pdfium_line in page_all_lines:
                for span in vertical_spans:
                    if calculate_overlap_area_in_bbox1_area_ratio(pdfium_line['bbox'].bbox, span['bbox']) > 0.5:
                        for pdfium_span in pdfium_line['spans']:
                            span['content'] += pdfium_span['text']
                        break

            for span in vertical_spans:
                if len(span['content']) == 0:
                    spans.remove(span)

        """\u6c34\u5e73\u7684span\u6846\u5148\u7528char\u586b\u5145\uff0c\u518d\u7528ocr\u586b\u5145\u7a7a\u7684span\u6846"""
        new_spans = []

        for span in useful_spans + unuseful_spans:
            if span['type'] in [ContentType.TEXT]:
                span['chars'] = []
                new_spans.append(span)

        need_ocr_spans = fill_char_in_spans(new_spans, page_all_chars, median_span_height)

        return _prepare_post_ocr_spans(need_ocr_spans, spans, pil_img, scale)
    finally:
        close_pdfium_child(textpage)


def _is_supported_rotation(rotation) -> bool:
    """Validate the current value."""
    rotation_degrees = math.degrees(rotation)
    return any(abs(rotation_degrees - angle) < 0.1 for angle in [0, 90, 180, 270])


def _get_char_fill_key(char):
    """Build the required output."""
    char_idx = char.get('char_idx')
    if char_idx is not None:
        return ('char_idx', char_idx)
    return ('object_id', id(char))


def _iter_line_chars(line):
    """Implementation detail."""
    for span in line.get('spans', []):
        for char in span.get('chars', []):
            yield char


def _is_visible_standard_rotation_char(char) -> bool:
    """Validate the current value."""
    text = char.get('char', '')
    if not text or text.isspace() or text in {'\r', '\n'}:
        return False

    bbox = char.get('bbox')
    if not bbox:
        return False

    x0, y0, x1, y1 = [float(v) for v in bbox]
    return (
        x1 > x0
        and y1 > y0
        and _is_supported_rotation(char.get('rotation', 0))
    )


def _get_chars_for_span_fill(page_chars):
    """Remove invalid or unnecessary data."""
    all_chars = page_chars['chars']
    fill_char_keys = {
        _get_char_fill_key(char)
        for char in all_chars
        if _is_supported_rotation(char.get('rotation', 0))
    }

    rotated_chars = [
        char for char in all_chars
        if not _is_supported_rotation(char.get('rotation', 0))
    ]
    if not rotated_chars:
        return [
            char for char in all_chars
            if _get_char_fill_key(char) in fill_char_keys
        ]

    for line in get_lines_from_chars(all_chars):
        if not _is_supported_rotation(line.get('rotation', 0)):
            continue

        line_chars = list(_iter_line_chars(line))
        if not any(_is_visible_standard_rotation_char(char) for char in line_chars):
            continue

        # Implementation detail.
        for char in line_chars:
            if not _is_supported_rotation(char.get('rotation', 0)):
                fill_char_keys.add(_get_char_fill_key(char))

    return [
        char for char in all_chars
        if _get_char_fill_key(char) in fill_char_keys
    ]


def _prepare_post_ocr_spans(need_ocr_spans, spans, pil_img, scale):
    if len(need_ocr_spans) == 0:
        return spans

    for span in need_ocr_spans:
        # Implementation detail.
        span_pil_img = get_crop_img(span['bbox'], pil_img, scale)
        span_img = cv2.cvtColor(np.array(span_pil_img), cv2.COLOR_RGB2BGR)
        # Calculate the result.
        if calculate_contrast(span_img, img_mode='bgr') < 0.17:
            if _restore_post_ocr_fallback(span):
                continue
            if span in spans:
                spans.remove(span)
            continue

        span['content'] = ''
        span['score'] = 1.0
        span['np_img'] = span_img

    return spans


class SpanBlockMatcher:
    """Sort items into the required order."""

    def __init__(self, spans):
        self.spans = list(spans)
        self.used_span_indices = set()
        self.grid_size = self._get_grid_size(self.spans)
        self.grid = self._build_grid(self.spans)

    @staticmethod
    def _get_grid_size(spans):
        """Implementation detail."""
        heights = [
            span['bbox'][3] - span['bbox'][1]
            for span in spans
            if span.get('bbox') and span['bbox'][3] > span['bbox'][1]
        ]
        if not heights:
            return 1
        return max(1, statistics.median(heights))

    def _build_grid(self, spans):
        """Implementation detail."""
        grid = collections.defaultdict(list)
        for index, span in enumerate(spans):
            bbox = span.get('bbox')
            if not bbox:
                continue
            start_cell, end_cell = self._cell_range(bbox)
            for cell_idx in range(start_cell, end_cell + 1):
                grid[cell_idx].append(index)
        return grid

    def _cell_range(self, bbox):
        """Calculate the result."""
        return (
            int(bbox[1] / self.grid_size),
            int(bbox[3] / self.grid_size),
        )

    def _candidate_indices_for_block(self, block_bbox):
        """Implementation detail."""
        start_cell, end_cell = self._cell_range(block_bbox)
        candidate_indices = set()
        for cell_idx in range(start_cell, end_cell + 1):
            candidate_indices.update(self.grid.get(cell_idx, []))
        return sorted(candidate_indices)

    def collect_for_block(self, block_bbox, overlap_ratio_getter=None, threshold=0.5):
        """Prepare the output value."""
        if overlap_ratio_getter is None:
            overlap_ratio_getter = self._default_overlap_ratio

        block_spans = []
        for span_idx in self._candidate_indices_for_block(block_bbox):
            if span_idx in self.used_span_indices:
                continue
            span = self.spans[span_idx]
            if overlap_ratio_getter(span, block_bbox) > threshold:
                block_spans.append(span)
                self.used_span_indices.add(span_idx)
        return block_spans

    def remaining_spans(self):
        """Prepare the output value."""
        return [
            span
            for index, span in enumerate(self.spans)
            if index not in self.used_span_indices
        ]

    @staticmethod
    def _default_overlap_ratio(span, block_bbox):
        """Calculate the result."""
        return calculate_overlap_area_in_bbox1_area_ratio(span['bbox'], block_bbox)


def fill_char_in_spans(spans, all_chars, median_span_height):
    # Implementation detail.
    spans = sorted(spans, key=lambda x: x['bbox'][1])

    grid_size = max(1, median_span_height)
    grid = collections.defaultdict(list)
    span_bboxes = []
    for i, span in enumerate(spans):
        span_bbox = span['bbox']
        span_bboxes.append(span_bbox)
        start_cell = int(span_bbox[1] / grid_size)
        end_cell = int(span_bbox[3] / grid_size)
        for cell_idx in range(start_cell, end_cell + 1):
            grid[cell_idx].append(i)

    for char in all_chars:
        char_bbox = char['bbox']
        char_center_x = (char_bbox[0] + char_bbox[2]) / 2
        char_center_y = (char_bbox[1] + char_bbox[3]) / 2
        cell_idx = int(char_center_y / grid_size)

        candidate_span_indices = grid.get(cell_idx, [])

        for span_idx in candidate_span_indices:
            span = spans[span_idx]
            span_bbox = span_bboxes[span_idx]
            if (
                char['char'] not in LINE_STOP_FLAG
                and char['char'] not in LINE_START_FLAG
                and not span_bbox[0] < char_center_x < span_bbox[2]
            ):
                continue
            if calculate_char_in_span(char_bbox, span_bbox, char['char']):
                span['chars'].append(char)
                break

    need_ocr_spans = []
    for span in spans:
        private_use_signal = _get_private_use_text_signal(span['chars'])
        should_post_ocr_private_use = _should_fallback_to_post_ocr_for_private_use_text(
            private_use_signal
        )
        chars_to_content(span)
        if should_post_ocr_private_use and span.get('content'):
            span[POST_OCR_FALLBACK_CONTENT_KEY] = span['content']
            span[POST_OCR_FALLBACK_SCORE_KEY] = span.get('score', 1.0)
            span[POST_OCR_REASON_KEY] = POST_OCR_REASON_PRIVATE_USE_TEXT
            need_ocr_spans.append(span)
        # Remove invalid or unnecessary data.
        elif len(span['content']) * span['height'] < span['width'] * 0.5:
            # logger.info(f"maybe empty span: {len(span['content'])}, {span['height']}, {span['width']}")
            need_ocr_spans.append(span)
        del span['height'], span['width']
    return need_ocr_spans


LINE_STOP_FLAG = (
    '.', '!', '?', '\u3002', '\uff01', '\uff1f', ')', '\uff09', '"', '\u201d', ':', '\uff1a', ';',
    '\uff1b', ']', '\u3011', '}', '}', '>', '\u300b', '\u3001', ',', '\uff0c', '-', '—', '–',
)
LINE_START_FLAG = (
    '(', '\uff08', '"', '\u201c', '\u3010', '{', '\u300a', '<', '\u300c', '\u300e', '\u3010', '[',
)

Span_Height_Ratio = 0.33  # Implementation detail.
SCRIPT_BODY_HEIGHT_RATIO = 0.9
SCRIPT_CENTER_TOLERANCE_RATIO = 0.15


def _is_private_use_char(char: str) -> bool:
    """Validate the current value."""
    return (
        len(char) == 1
        and PRIVATE_USE_AREA_START <= ord(char) <= PRIVATE_USE_AREA_END
    )


def _get_private_use_text_signal(chars):
    """Calculate the result."""
    pua_count = 0
    text_char_count = 0
    current_pua_run = 0
    max_pua_run = 0

    for char in chars:
        for text_char in char.get('char', ''):
            if text_char.isspace():
                current_pua_run = 0
                continue

            text_char_count += 1
            if _is_private_use_char(text_char):
                pua_count += 1
                current_pua_run += 1
                max_pua_run = max(max_pua_run, current_pua_run)
            else:
                current_pua_run = 0

    pua_ratio = 0.0
    if text_char_count > 0:
        pua_ratio = pua_count / text_char_count

    return {
        'pua_count': pua_count,
        'text_char_count': text_char_count,
        'pua_ratio': pua_ratio,
        'max_pua_run': max_pua_run,
    }


def _should_fallback_to_post_ocr_for_private_use_text(signal) -> bool:
    """Implementation detail."""
    pua_count = signal['pua_count']
    if pua_count < PRIVATE_USE_TEXT_COUNT_THRESHOLD:
        return False

    return (
        signal['max_pua_run'] >= PRIVATE_USE_TEXT_RUN_THRESHOLD
        or signal['pua_ratio'] >= PRIVATE_USE_TEXT_RATIO_THRESHOLD
    )


def _clear_post_ocr_fallback(span):
    """Remove invalid or unnecessary data."""
    span.pop(POST_OCR_FALLBACK_CONTENT_KEY, None)
    span.pop(POST_OCR_FALLBACK_SCORE_KEY, None)
    span.pop(POST_OCR_REASON_KEY, None)


def _restore_post_ocr_fallback(span) -> bool:
    """Process text content."""
    if POST_OCR_FALLBACK_CONTENT_KEY not in span:
        _clear_post_ocr_fallback(span)
        return False

    span['content'] = span[POST_OCR_FALLBACK_CONTENT_KEY]
    if POST_OCR_FALLBACK_SCORE_KEY in span:
        span['score'] = span[POST_OCR_FALLBACK_SCORE_KEY]
    _clear_post_ocr_fallback(span)
    return True


def calculate_char_in_span(char_bbox, span_bbox, char, span_height_ratio=Span_Height_Ratio):
    char_center_x = (char_bbox[0] + char_bbox[2]) / 2
    char_center_y = (char_bbox[1] + char_bbox[3]) / 2
    span_center_y = (span_bbox[1] + span_bbox[3]) / 2
    span_height = span_bbox[3] - span_bbox[1]

    if (
        span_bbox[0] < char_center_x < span_bbox[2]
        and span_bbox[1] < char_center_y < span_bbox[3]
        # Implementation detail.
        and abs(char_center_y - span_center_y) < span_height * span_height_ratio
    ):
        return True
    else:
        # Implementation detail.
        # Implementation detail.
        if char in LINE_STOP_FLAG:
            if (
                (span_bbox[2] - span_height) < char_bbox[0] < span_bbox[2]
                and char_center_x > span_bbox[0]
                and span_bbox[1] < char_center_y < span_bbox[3]
                and abs(char_center_y - span_center_y) < span_height * span_height_ratio
            ):
                return True
        elif char in LINE_START_FLAG:
            if (
                span_bbox[0] < char_bbox[2] < (span_bbox[0] + span_height)
                and char_center_x < span_bbox[2]
                and span_bbox[1] < char_center_y < span_bbox[3]
                and abs(char_center_y - span_center_y) < span_height * span_height_ratio
            ):
                return True
        else:
            return False


def _get_char_bbox_metrics(char):
    """Extract the required value."""
    bbox = char['bbox']
    x0, y0, x1, y1 = [float(v) for v in bbox]
    return {
        'width': x1 - x0,
        'height': y1 - y0,
        'center_y': (y0 + y1) / 2,
    }


def _get_char_bbox_metrics_list(chars):
    """Validate the current value."""
    return [_get_char_bbox_metrics(char) for char in chars]


def _is_valid_script_reference_char(char, metrics) -> bool:
    """Remove invalid or unnecessary data."""
    if char['char'] in {' ', '\r', '\n'}:
        return False

    return metrics['height'] > 1 and metrics['width'] > 0


def _get_body_axis(chars, char_metrics):
    """Implementation detail."""
    valid_metrics = [
        metrics for char, metrics in zip(chars, char_metrics)
        if _is_valid_script_reference_char(char, metrics)
    ]
    if not valid_metrics:
        return None

    max_height = max(metrics['height'] for metrics in valid_metrics)
    body_metrics = [
        metrics for metrics in valid_metrics
        if metrics['height'] >= max_height * SCRIPT_BODY_HEIGHT_RATIO
    ]
    if not body_metrics:
        return None

    return {
        'center_y': statistics.median(metrics['center_y'] for metrics in body_metrics),
        'height': statistics.median(metrics['height'] for metrics in body_metrics),
    }


def _classify_char_script_roles(chars, char_metrics):
    """Validate the current value."""
    body_axis = _get_body_axis(chars, char_metrics)
    if body_axis is None or body_axis['height'] <= 0:
        return ['body'] * len(chars)

    tolerance = body_axis['height'] * SCRIPT_CENTER_TOLERANCE_RATIO
    roles = []
    for char, metrics in zip(chars, char_metrics):
        if not _is_valid_script_reference_char(char, metrics):
            roles.append('body')
            continue

        char_center_y = metrics['center_y']
        if char_center_y < body_axis['center_y'] - tolerance:
            roles.append('sup')
        elif char_center_y > body_axis['center_y'] + tolerance:
            roles.append('sub')
        else:
            roles.append('body')
    return roles


def _append_script_wrapped_text(parts, role, text):
    """Process text content."""
    if not text:
        return
    if role == 'sup':
        parts.append(f'<sup>{text}</sup>')
    elif role == 'sub':
        parts.append(f'<sub>{text}</sub>')
    else:
        parts.append(text)


def _wrap_script_runs(role_text_parts):
    """Build the required output."""
    wrapped_parts = []
    current_role = None
    current_text_parts = []

    for role, text in role_text_parts:
        if role != current_role:
            _append_script_wrapped_text(
                wrapped_parts,
                current_role,
                ''.join(current_text_parts),
            )
            current_role = role
            current_text_parts = [text]
        else:
            current_text_parts.append(text)

    _append_script_wrapped_text(
        wrapped_parts,
        current_role,
        ''.join(current_text_parts),
    )
    return ''.join(wrapped_parts)


def _remove_control_line_break_chars(chars):
    """Remove invalid or unnecessary data."""
    return [
        char for char in chars
        if char.get('char') not in {'\r', '\n'}
    ]


def chars_to_content(span):
    # Validate the current value.
    if len(span['chars']) != 0:
        chars = span['chars']
        # Sort items into the required order.
        if any(
            chars[idx]['char_idx'] > chars[idx + 1]['char_idx']
            for idx in range(len(chars) - 1)
        ):
            chars = sorted(chars, key=lambda x: x['char_idx'])

        chars = _remove_control_line_break_chars(chars)
        if len(chars) == 0:
            span['content'] = ''
        else:
            char_metrics = _get_char_bbox_metrics_list(chars)
            # Calculate the width of each character
            char_widths = [metrics['width'] for metrics in char_metrics]
            # Calculate the median width
            median_width = statistics.median(char_widths)
            script_roles = _classify_char_script_roles(chars, char_metrics)

            role_text_parts = []
            for idx, char1 in enumerate(chars):
                char2 = chars[idx + 1] if idx + 1 < len(chars) else None
                role1 = script_roles[idx]
                role2 = script_roles[idx + 1] if char2 else None

                # Add the value to the result.
                role_text_parts.append((role1, char1['char']))
                if (
                    char2
                    and char2['bbox'][0] - char1['bbox'][2] > median_width * 0.25
                    and char1['char'] != ' '
                    and char2['char'] != ' '
                ):
                    space_role = role1 if role1 == role2 else 'body'
                    role_text_parts.append((space_role, ' '))

            content = _wrap_script_runs(role_text_parts)
            content = __replace_unicode(content)
            content = __replace_ligatures(content)
            span['content'] = content.strip()

    del span['chars']


def calculate_contrast(img, img_mode) -> float:
    """
    Calculate the result.
    Process image content.
    Process image content.
    Process image content.
    """
    if img_mode == 'rgb':
        # Convert the value to the required format.
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif img_mode == 'bgr':
        # Convert the value to the required format.
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("Invalid image mode. Please provide 'rgb' or 'bgr'.")

    # Calculate the result.
    mean_value = np.mean(gray_img)
    std_dev = np.std(gray_img)
    # Implementation detail.
    contrast = std_dev / (mean_value + 1e-6)
    # logger.debug(f"contrast: {contrast}")
    return round(contrast, 2)
