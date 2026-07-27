# Copyright (c) Opendatalab. All rights reserved.
import html as html_lib
import re
from typing import Literal
from urllib.parse import urlparse

from loguru import logger

from vsf.utils.enum_class import ContentType, BlockType
from vsf.utils.magic_model_utils import tie_up_category_by_index


class MagicModel:
    def __init__(self, page_blocks: list):
        self.page_blocks = page_blocks

        blocks = []
        self.all_spans = []

        # Implementation detail.
        page_blocks = classify_caption_blocks(page_blocks)

        # Parse the input data.
        for index, block_info in enumerate(page_blocks):

            block_type = block_info["type"]
            block_content = block_info.get("content", "")
            if not block_content and block_type != BlockType.CHART:
                continue

            if block_type in [
                "text",
                "title",
                "image_caption",
                "table_caption",
                "chart_caption",
                "header",
                "footer",
                "page_footnote",
            ]:
                span = parse_text_block_spans(block_content)

            elif block_type in ["image"]:
                block_type = BlockType.IMAGE_BODY
                span = {
                    "type": ContentType.IMAGE,
                    "image_base64": block_content,
                }
            elif block_type in ["table"]:
                block_type = BlockType.TABLE_BODY
                span = {
                    "type": ContentType.TABLE,
                    "html": clean_table_html(block_content),
                }
            elif block_type in ["chart"]:
                block_type = BlockType.CHART_BODY
                span = {
                    "type": ContentType.CHART,
                    "content": block_content,
                }
                if block_info.get("image_base64"):
                    span["image_base64"] = block_info["image_base64"]
            elif block_type in ["equation"]:
                block_type = BlockType.INTERLINE_EQUATION
                span = {
                    "type": ContentType.INTERLINE_EQUATION,
                    "content": block_content,
                }
            elif block_type in ["list"]:
                # Parse the input data.
                parsed_list = parse_list_block(block_info)
                if parsed_list:
                    # Implementation detail.
                    parsed_list["index"] = index
                    blocks.append(parsed_list)
                continue
            elif block_type in ["index"]:
                # Parse the input data.
                parsed_index = parse_index_block(block_info)
                if parsed_index:
                    parsed_index["index"] = index
                    blocks.append(parsed_index)
                continue
            else:
                # Remove invalid or unnecessary data.
                continue

            # Add the value to the result.
            if isinstance(span, dict):
                line = {
                    "spans": [span]
                }
            elif isinstance(span, list):
                line = {
                    "spans":span
                }
            else:
                raise ValueError(f"Unsupported span type: {type(span)}")

            block = {
                    "type": block_type,
                    "lines": [line],
                    "index": index,
            }
            anchor = block_info.get("anchor")
            if (
                isinstance(anchor, str)
                and anchor.strip()
                and block_type in [BlockType.TITLE, BlockType.TEXT, BlockType.INTERLINE_EQUATION]
            ):
                block["anchor"] = anchor.strip()
            if block_type == BlockType.TITLE:
                block["is_numbered_style"] = block_info.get("is_numbered_style", False)
                block["level"] = block_info.get("level", 1)
            blocks.append(block)

        self.image_blocks = []
        self.table_blocks = []
        self.chart_blocks = []
        self.interline_equation_blocks = []
        self.text_blocks = []
        self.title_blocks = []
        self.discarded_blocks = []
        self.list_blocks = []
        self.index_blocks = []
        for block in blocks:
            if block["type"] in [BlockType.IMAGE_BODY, BlockType.IMAGE_CAPTION, BlockType.IMAGE_FOOTNOTE]:
                self.image_blocks.append(block)
            elif block["type"] in [BlockType.TABLE_BODY, BlockType.TABLE_CAPTION, BlockType.TABLE_FOOTNOTE]:
                self.table_blocks.append(block)
            elif block["type"] in [BlockType.CHART_BODY, BlockType.CHART_CAPTION]:
                self.chart_blocks.append(block)
            elif block["type"] == BlockType.INTERLINE_EQUATION:
                self.interline_equation_blocks.append(block)
            elif block["type"] == BlockType.TEXT:
                self.text_blocks.append(block)
            elif block["type"] == BlockType.TITLE:
                self.title_blocks.append(block)
            elif block["type"] in [BlockType.REF_TEXT]:
                self.ref_text_blocks.append(block)
            elif block["type"] in [BlockType.PHONETIC]:
                self.phonetic_blocks.append(block)
            elif block["type"] in [BlockType.HEADER, BlockType.FOOTER, BlockType.PAGE_NUMBER, BlockType.ASIDE_TEXT, BlockType.PAGE_FOOTNOTE]:
                self.discarded_blocks.append(block)
            elif block["type"] == BlockType.LIST:
                self.list_blocks.append(block)
            elif block["type"] == BlockType.INDEX:
                self.index_blocks.append(block)
            else:
                continue

        self.image_blocks, not_include_image_blocks = fix_two_layer_blocks(self.image_blocks, BlockType.IMAGE)
        self.table_blocks, not_include_table_blocks = fix_two_layer_blocks(self.table_blocks, BlockType.TABLE)
        self.chart_blocks, not_include_chart_blocks = fix_two_layer_blocks(self.chart_blocks, BlockType.CHART)

        for block in not_include_image_blocks + not_include_table_blocks + not_include_chart_blocks:
            block["type"] = BlockType.TEXT
            self.text_blocks.append(block)


    def get_list_blocks(self):
        return self.list_blocks

    def get_index_blocks(self):
        return self.index_blocks

    def get_image_blocks(self):
        return self.image_blocks

    def get_table_blocks(self):
        return self.table_blocks

    def get_chart_blocks(self):
        return self.chart_blocks

    def get_title_blocks(self):
        return self.title_blocks

    def get_text_blocks(self):
        return self.text_blocks

    def get_interline_equation_blocks(self):
        return self.interline_equation_blocks

    def get_discarded_blocks(self):
        return self.discarded_blocks


def _parse_style_list(style_str: str | None) -> list:
    """Parse the input data."""
    if not style_str:
        return []
    return [style.strip() for style in style_str.split(',') if style.strip()]


def _parse_hyperlink_text_children(hyperlink_content: str, text_tag_re) -> tuple:
    """Parse the input data."""
    url_start = hyperlink_content.find('<url>')
    url_end = hyperlink_content.find('</url>')
    if url_start == -1 or url_end == -1 or url_end < url_start:
        return [], ''

    children = []
    pos = 0
    while pos < url_start:
        text_match = text_tag_re.search(hyperlink_content, pos)
        if text_match is None or text_match.start() >= url_start:
            break

        text_end = hyperlink_content.find('</text>', text_match.end())
        if text_end == -1 or text_end > url_start:
            return [], ''

        child = {
            "type": ContentType.TEXT,
            "content": hyperlink_content[text_match.end():text_end],
        }
        style = _parse_style_list(text_match.group(1))
        if style:
            child["style"] = style
        children.append(child)
        pos = text_end + 7

    return children, hyperlink_content[url_start + 5:url_end]


def parse_text_block_spans(content: str) -> list:
    """
    Parse the input data.

    Implementation detail.
    Process formula content.
    Implementation detail.
    Process text content.

    Implementation detail.

    Args:
        Process text content.

    Returns:
        Implementation detail.
        Process text content.
    """
    if not content:
        return []

    # Match the expected pattern.
    _text_tag_re = re.compile(r'<text(?:\s+style="([^"]*)")?>')

    spans = []
    last_end = 0
    pos = 0

    while pos < len(content):
        # Match the expected pattern.
        eq_start = content.find('<eq>', pos)
        # Match the expected pattern.
        hyperlink_start = content.find('<hyperlink>', pos)
        # Match the expected pattern.
        text_tag_match = _text_tag_re.search(content, pos)
        text_tag_start = text_tag_match.start() if text_tag_match else -1

        # Implementation detail.
        candidates = []
        if eq_start != -1:
            candidates.append((eq_start, 'eq'))
        if hyperlink_start != -1:
            candidates.append((hyperlink_start, 'hyperlink'))
        if text_tag_start != -1:
            candidates.append((text_tag_start, 'text'))

        # Process text content.
        if not candidates:
            remaining_text = content[last_end:]
            if remaining_text:
                spans.append({
                    "type": ContentType.TEXT,
                    "content": remaining_text
                })
            break

        # Implementation detail.
        next_tag_pos, next_tag_type = min(candidates, key=lambda x: x[0])

        # Process text content.
        if next_tag_pos > last_end:
            text_before = content[last_end:next_tag_pos]
            if text_before:
                spans.append({
                    "type": ContentType.TEXT,
                    "content": text_before
                })

        # Process formula content.
        if next_tag_type == 'eq':
            eq_end = content.find('</eq>', next_tag_pos)
            if eq_end != -1:
                formula_content = content[next_tag_pos + 4:eq_end]
                spans.append({
                    "type": ContentType.INLINE_EQUATION,
                    "content": formula_content
                })
                pos = eq_end + 5  # Remove invalid or unnecessary data.
                last_end = pos
            else:
                # Process text content.
                spans.append({
                    "type": ContentType.TEXT,
                    "content": content[last_end:]
                })
                break

        # Process text content.
        elif next_tag_type == 'text':
            text_end = content.find('</text>', next_tag_pos)
            if text_end != -1:
                # Match the expected pattern.
                # Match the expected pattern.
                tag_open_end = content.find('>', next_tag_pos) + 1
                text_content = content[tag_open_end:text_end]
                style_str = text_tag_match.group(1) if text_tag_match and text_tag_match.start() == next_tag_pos else None
                span = {
                    "type": ContentType.TEXT,
                    "content": text_content
                }
                if style_str:
                    span["style"] = [s.strip() for s in style_str.split(',') if s.strip()]
                spans.append(span)
                pos = text_end + 7  # Remove invalid or unnecessary data.
                last_end = pos
            else:
                # Process text content.
                spans.append({
                    "type": ContentType.TEXT,
                    "content": content[last_end:]
                })
                break

        # Process the current item.
        elif next_tag_type == 'hyperlink':
            hyperlink_end = content.find('</hyperlink>', next_tag_pos)
            if hyperlink_end != -1:
                # Extract the required value.
                hyperlink_content = content[next_tag_pos + 11:hyperlink_end]

                # Parse the input data.
                children, link_url = _parse_hyperlink_text_children(
                    hyperlink_content,
                    _text_tag_re,
                )

                if children and link_url:
                    if len(children) == 1:
                        child = children[0]
                        span = {
                            "type": ContentType.HYPERLINK,
                            "content": child["content"],
                            "url": link_url,
                        }
                        if child.get("style"):
                            span["style"] = child["style"]
                    else:
                        span = {
                            "type": ContentType.HYPERLINK,
                            "content": ''.join(
                                child["content"] for child in children
                            ),
                            "url": link_url,
                            "children": children,
                        }
                    spans.append(span)
                    pos = hyperlink_end + 12  # Remove invalid or unnecessary data.
                    last_end = pos
                else:
                    # Process text content.
                    spans.append({
                        "type": ContentType.TEXT,
                        "content": content[last_end:]
                    })
                    break
            else:
                # Process text content.
                spans.append({
                    "type": ContentType.TEXT,
                    "content": content[last_end:]
                })
                break

    return spans


def parse_list_block(list_block: dict):
    """
    Parse the input data.

    Args:
        Implementation detail.

    Returns:
        Parse the input data.
    """
    content = list_block.get("content", [])
    if not content:
        return None

    blocks = []

    for item in content:
        item_type = item.get("type", "")

        if item_type == "text":
            # Parse the input data.
            text_content = item.get("content", "")
            spans = parse_text_block_spans(text_content)
            text_block = {
                "type": BlockType.TEXT,
                "lines": [{"spans": spans}]
            }
            blocks.append(text_block)

        elif item_type == "list":
            # Parse the input data.
            nested_list = parse_list_block(item)
            if nested_list:
                blocks.append(nested_list)

    # Build the required output.
    result = {
        "type": BlockType.LIST,
        "attribute": list_block.get("attribute", "unordered"),
        "ilevel": list_block.get("ilevel", 0),
        "blocks": blocks
    }
    if "start" in list_block:
        result["start"] = list_block["start"]

    return result


def parse_index_block(index_block: dict):
    """
    Parse the input data.

    Args:
        Implementation detail.

    Returns:
        Parse the input data.
    """
    content = index_block.get("content", [])
    if not content:
        return None

    blocks = []

    for item in content:
        item_type = item.get("type", "")

        if item_type == "text":
            text_content = item.get("content", "")
            spans = parse_text_block_spans(text_content)
            text_block = {
                "type": BlockType.TEXT,
                "lines": [{"spans": spans}]
            }
            anchor = item.get("anchor")
            if isinstance(anchor, str) and anchor.strip():
                text_block["anchor"] = anchor.strip()
            blocks.append(text_block)

        elif item_type == "index":
            nested_index = parse_index_block(item)
            if nested_index:
                blocks.append(nested_index)

    result = {
        "type": BlockType.INDEX,
        "ilevel": index_block.get("ilevel", 0),
        "blocks": blocks
    }

    return result


def _sanitize_table_hyperlink_href(href: str) -> str:
    """Process table content."""
    normalized_href = html_lib.unescape(href).strip()
    if not normalized_href:
        return ""

    if normalized_href.lower().startswith(("javascript:", "data:", "vbscript:")):
        return ""

    parsed = urlparse(normalized_href)
    scheme = parsed.scheme.lower() if parsed.scheme else ""
    if scheme and scheme not in {"http", "https", "mailto", "ftp"}:
        return ""

    return html_lib.escape(normalized_href, quote=True)


def clean_table_html(html: str) -> str:
    """
    Process table content.

    Implementation detail.
    Merge the related values.
    Merge the related values.
    Process table content.
    Process image content.

    Implementation detail.
    Remove invalid or unnecessary data.
    Remove invalid or unnecessary data.
    Remove invalid or unnecessary data.
    Process table content.

    Args:
        Process table content.

    Returns:
        Implementation detail.
    """
    if not html:
        return ""

    # Process table content.
    preserved_attrs = {'colspan', 'rowspan'}
    # Process image content.
    img_preserved_attrs = {'src', 'alt', 'width', 'height'}
    # Process table content.
    anchor_preserved_attrs = {'href'}

    def clean_tag(match):
        """Implementation detail."""
        full_tag = match.group(0)
        tag_name = match.group(1).lower()

        # Process the current item.
        is_self_closing = full_tag.rstrip().endswith('/>')

        # Process image content.
        current_preserved = preserved_attrs | (img_preserved_attrs if tag_name == 'img' else set())
        current_preserved |= anchor_preserved_attrs if tag_name == 'a' else set()

        # Extract the required value.
        kept_attrs = []

        # Match the expected pattern.
        attr_pattern = r'(\w+)\s*=\s*(?:"([^"]*)"|\'([^\']*)\'|(\S+))|(\w+)(?=\s|>|/>)'
        for attr_match in re.finditer(attr_pattern, full_tag):
            if attr_match.group(5):
                # Remove invalid or unnecessary data.
                continue

            attr_name = attr_match.group(1)
            if attr_name is None:
                continue
            attr_name = attr_name.lower()
            attr_value = attr_match.group(2) or attr_match.group(3) or attr_match.group(4) or ""

            # Process image content.
            if tag_name == "a" and attr_name == "href":
                attr_value = _sanitize_table_hyperlink_href(attr_value)
                if not attr_value:
                    continue

            if attr_name in current_preserved:
                kept_attrs.append(f'{attr_name}="{attr_value}"')

        # Build the required output.
        if kept_attrs:
            attrs_str = ' ' + ' '.join(kept_attrs)
        else:
            attrs_str = ''

        if is_self_closing:
            return f'<{tag_name}{attrs_str}/>'
        else:
            return f'<{tag_name}{attrs_str}>'

    # Match the expected pattern.
    # Match the expected pattern.
    tag_pattern = r'<(\w+)(?:\s+[^>]*)?\s*/?>'

    result = re.sub(tag_pattern, clean_tag, html)

    return result


def isolated_formula_clean(txt):
    latex = txt[:]
    if latex.startswith("\\["): latex = latex[2:]
    if latex.endswith("\\]"): latex = latex[:-2]
    latex = latex.strip()
    return latex


def code_content_clean(content):
    """Remove invalid or unnecessary data."""
    if not content:
        return ""

    lines = content.splitlines()
    start_idx = 0
    end_idx = len(lines)

    # Process the current item.
    if lines and lines[0].startswith("```"):
        start_idx = 1

    # Process the current item.
    if lines and end_idx > start_idx and lines[end_idx - 1].strip() == "```":
        end_idx -= 1

    # Implementation detail.
    if start_idx < end_idx:
        return "\n".join(lines[start_idx:end_idx]).strip()
    return ""


def __tie_up_category_by_index(blocks, subject_block_type, object_block_type):
    """Implementation detail."""
    # Extract the required value.
    def get_subjects():
        return list(
            map(
                lambda x: {"lines": x["lines"], "index": x["index"]},
                filter(
                    lambda x: x["type"] == subject_block_type,
                    blocks,
                ),
            )
        )

    def get_objects():
        return list(
            map(
                lambda x: {"lines": x["lines"], "index": x["index"]},
                filter(
                    lambda x: x["type"] == object_block_type,
                    blocks,
                ),
            )
        )

    # Implementation detail.
    return tie_up_category_by_index(
        get_subjects,
        get_objects,
        include_bbox=False,
    )


def get_type_blocks(blocks, block_type: Literal["image", "table", "chart"]):
    with_captions = __tie_up_category_by_index(blocks, f"{block_type}_body", f"{block_type}_caption")
    ret = []
    for v in with_captions:
        record = {
            f"{block_type}_body": v["sub_bbox"],
            f"{block_type}_caption_list": v["obj_bboxes"],
        }
        ret.append(record)
    return ret


def fix_two_layer_blocks(blocks, fix_type: Literal["image", "table", "chart"]):
    need_fix_blocks = get_type_blocks(blocks, fix_type)
    fixed_blocks = []
    not_include_blocks = []
    processed_indices = set()

    # Process the current item.
    for block in need_fix_blocks:
        caption_list = block[f"{fix_type}_caption_list"]
        body_index = block[f"{fix_type}_body"]["index"]

        # Process the current item.
        if caption_list:
            # Validate the current value.
            caption_list.sort(key=lambda x: x["index"], reverse=True)
            filtered_captions = [caption_list[0]]
            for i in range(1, len(caption_list)):
                prev_index = caption_list[i - 1]["index"]
                curr_index = caption_list[i]["index"]

                # Validate the current value.
                if curr_index == prev_index - 1:
                    filtered_captions.append(caption_list[i])
                else:
                    # Validate the current value.
                    gap_indices = set(range(curr_index + 1, prev_index))
                    if gap_indices == {body_index}:
                        # Implementation detail.
                        filtered_captions.append(caption_list[i])
                    else:
                        # Implementation detail.
                        not_include_blocks.extend(caption_list[i:])
                        break
            # Implementation detail.
            filtered_captions.reverse()
            block[f"{fix_type}_caption_list"] = filtered_captions

    # Build the required output.
    for block in need_fix_blocks:
        body = block[f"{fix_type}_body"]
        caption_list = block[f"{fix_type}_caption_list"]

        body["type"] = f"{fix_type}_body"
        for caption in caption_list:
            caption["type"] = f"{fix_type}_caption"
            processed_indices.add(caption["index"])

        processed_indices.add(body["index"])

        two_layer_block = {
            "type": fix_type,
            "blocks": [body],
            "index": body["index"],
        }
        two_layer_block["blocks"].extend([*caption_list])
        # Sort items into the required order.
        two_layer_block["blocks"].sort(key=lambda x: x["index"])

        fixed_blocks.append(two_layer_block)

    # Add the value to the result.
    for block in blocks:
        block.pop("type", None)
        if block["index"] not in processed_indices and block not in not_include_blocks:
            not_include_blocks.append(block)

    return fixed_blocks, not_include_blocks


def classify_caption_blocks(page_blocks: list) -> list:
    """
    Implementation detail.

    Implementation detail.
    Implementation detail.
    Implementation detail.
    Validate the current value.
    Implementation detail.
    Implementation detail.
       Implementation detail.
       Implementation detail.
       Implementation detail.
    """
    if not page_blocks:
        return page_blocks

    available_types = ["table", "image", "chart"]

    # Match the expected pattern.
    table_caption_prefixes = ["\u8868", "table"]
    image_caption_prefixes = ["\u56fe", "fig"]
    chart_caption_prefixes = ["\u56fe", "fig", "chart"]

    # Process the current item.
    preprocessed_blocks = []
    n = len(page_blocks)

    for i, block in enumerate(page_blocks):
        block_type = block.get("type")

        # Validate the current value.
        if block_type in available_types:
            preprocessed_blocks.append(block)

            # Match the expected pattern.
            if i + 1 < n:
                next_block = page_blocks[i + 1]
                next_block_type = next_block.get("type")

                if next_block_type == "text":
                    content = next_block.get("content", "").strip().lower()

                    # Validate the current value.
                    if block_type == "table":
                        if any(content.startswith(prefix.lower()) for prefix in table_caption_prefixes):
                            # Process the current item.
                            next_block = next_block.copy()
                            next_block["type"] = "caption"
                            page_blocks[i + 1] = next_block
                    elif block_type == "image":
                        if any(content.startswith(prefix.lower()) for prefix in image_caption_prefixes):
                            # Process the current item.
                            next_block = next_block.copy()
                            next_block["type"] = "caption"
                            page_blocks[i + 1] = next_block
                    elif block_type == "chart":
                        if any(content.startswith(prefix.lower()) for prefix in chart_caption_prefixes):
                            # Process the current item.
                            next_block = next_block.copy()
                            next_block["type"] = "caption"
                            page_blocks[i + 1] = next_block
        else:
            preprocessed_blocks.append(block)

    # Process the current item.
    result_blocks = []

    for i, block in enumerate(page_blocks):
        if block.get("type") != "caption":
            result_blocks.append(block)
            continue

        # Match the expected pattern.
        # Remove invalid or unnecessary data.
        prev_parent_type = None
        j = i - 1
        while j >= 0:
            prev_block_type = page_blocks[j].get("type")
            if prev_block_type in available_types:
                prev_parent_type = prev_block_type
                break
            elif prev_block_type == "caption":
                # Match the expected pattern.
                j -= 1
            else:
                # Match the expected pattern.
                break

        # Match the expected pattern.
        # Remove invalid or unnecessary data.
        next_parent_type = None
        k = i + 1
        while k < n:
            next_block_type = page_blocks[k].get("type")
            if next_block_type in available_types:
                next_parent_type = next_block_type
                break
            elif next_block_type == "caption":
                # Match the expected pattern.
                k += 1
            else:
                # Match the expected pattern.
                break

        # Implementation detail.
        new_block = block.copy()
        if prev_parent_type:
            # Implementation detail.
            new_block["type"] = f"{prev_parent_type}_caption"
        elif next_parent_type:
            # Implementation detail.
            new_block["type"] = f"{next_parent_type}_caption"
        else:
            # Implementation detail.
            new_block["type"] = "text"

        result_blocks.append(new_block)

    return result_blocks
