# Copyright (c) Opendatalab. All rights reserved.
from vsf.utils.enum_class import ContentType
from vsf.utils.ocr_utils import _is_overlaps_y_exceeds_threshold, _is_overlaps_x_exceeds_threshold

VERTICAL_SPAN_HEIGHT_TO_WIDTH_RATIO_THRESHOLD = 2
VERTICAL_SPAN_IN_BLOCK_THRESHOLD = 0.8


def is_vertical_text_block_by_spans(spans):
    """Validate the current value."""
    valid_span_count = 0
    vertical_span_count = 0
    for span in spans:
        bbox = span.get('bbox')
        if not bbox or len(bbox) < 4:
            continue

        span_width = bbox[2] - bbox[0]
        span_height = bbox[3] - bbox[1]
        if span_width <= 0 or span_height <= 0:
            continue

        valid_span_count += 1
        if span_height / span_width > VERTICAL_SPAN_HEIGHT_TO_WIDTH_RATIO_THRESHOLD:
            vertical_span_count += 1

    if valid_span_count == 0:
        return False

    return vertical_span_count / valid_span_count > VERTICAL_SPAN_IN_BLOCK_THRESHOLD


def fix_text_block(block):
    # Convert the value to the required format.
    for span in block['spans']:
        if span['type'] == ContentType.INTERLINE_EQUATION:
            span['type'] = ContentType.INLINE_EQUATION

    if is_vertical_text_block_by_spans(block['spans']):
        # Process text content.
        block_lines = merge_spans_to_vertical_line(block['spans'])
        sort_block_lines = vertical_line_sort_spans_from_top_to_bottom(block_lines)
    else:
        block_lines = merge_spans_to_line(block['spans'])
        sort_block_lines = line_sort_spans_by_left_to_right(block_lines)

    block['lines'] = sort_block_lines
    del block['spans']
    return block


def merge_spans_to_line(spans, threshold=0.6):
    if len(spans) == 0:
        return []
    else:
        # Sort items into the required order.
        spans.sort(key=lambda span: span['bbox'][1])

        lines = []
        current_line = [spans[0]]
        for span in spans[1:]:
            # Implementation detail.
            # Implementation detail.
            if span['type'] in [
                    ContentType.INTERLINE_EQUATION, ContentType.IMAGE,
                    ContentType.TABLE
            ] or any(s['type'] in [
                    ContentType.INTERLINE_EQUATION, ContentType.IMAGE,
                    ContentType.TABLE
            ] for s in current_line):
                # Implementation detail.
                lines.append(current_line)
                current_line = [span]
                continue

            # Add the value to the result.
            if _is_overlaps_y_exceeds_threshold(span['bbox'], current_line[-1]['bbox'], threshold):
                current_line.append(span)
            else:
                # Implementation detail.
                lines.append(current_line)
                current_line = [span]

        # Add the value to the result.
        if current_line:
            lines.append(current_line)

        return lines


def merge_spans_to_vertical_line(spans, threshold=0.6):
    """Merge the related values."""
    if len(spans) == 0:
        return []
    else:
        # Sort items into the required order.
        spans.sort(key=lambda span: span['bbox'][2], reverse=True)

        vertical_lines = []
        current_line = [spans[0]]

        for span in spans[1:]:
            # Implementation detail.
            if span['type'] in [
                ContentType.INTERLINE_EQUATION, ContentType.IMAGE,
                ContentType.TABLE
            ] or any(s['type'] in [
                ContentType.INTERLINE_EQUATION, ContentType.IMAGE,
                ContentType.TABLE
            ] for s in current_line):
                vertical_lines.append(current_line)
                current_line = [span]
                continue

            # Add the value to the result.
            if _is_overlaps_x_exceeds_threshold(span['bbox'], current_line[-1]['bbox'], threshold):
                current_line.append(span)
            else:
                vertical_lines.append(current_line)
                current_line = [span]

        # Add the value to the result.
        if current_line:
            vertical_lines.append(current_line)

        return vertical_lines


# Sort items into the required order.
def line_sort_spans_by_left_to_right(lines):
    line_objects = []
    for line in lines:
        # Sort items into the required order.
        line.sort(key=lambda span: span['bbox'][0])
        line_bbox = [
            min(span['bbox'][0] for span in line),  # x0
            min(span['bbox'][1] for span in line),  # y0
            max(span['bbox'][2] for span in line),  # x1
            max(span['bbox'][3] for span in line),  # y1
        ]
        line_objects.append({
            'bbox': line_bbox,
            'spans': line,
        })
    return line_objects


def vertical_line_sort_spans_from_top_to_bottom(vertical_lines):
    line_objects = []
    for line in vertical_lines:
        # Sort items into the required order.
        line.sort(key=lambda span: span['bbox'][1])

        # Calculate the result.
        line_bbox = [
            min(span['bbox'][0] for span in line),  # x0
            min(span['bbox'][1] for span in line),  # y0
            max(span['bbox'][2] for span in line),  # x1
            max(span['bbox'][3] for span in line),  # y1
        ]

        # Prepare the output value.
        line_objects.append({
            'bbox': line_bbox,
            'spans': line,
        })
    return line_objects
