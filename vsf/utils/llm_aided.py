# Copyright (c) Opendatalab. All rights reserved.
from concurrent.futures import ThreadPoolExecutor

import json_repair
from loguru import logger
from openai import OpenAI

from vsf.backend.pipeline.pipeline_middle_json_mkcontent import merge_para_with_text
from vsf.utils.enum_class import BlockType


TITLE_BLOCK_TYPES = {
    BlockType.TITLE,
    BlockType.DOC_TITLE,
    BlockType.PARAGRAPH_TITLE,
}
MAX_TITLE_GROUP_WORKERS = 4


def _get_title_line_avg_height(block):
    line_avg_height = block.get("line_avg_height")
    if isinstance(line_avg_height, (int, float)) and line_avg_height > 0:
        return line_avg_height

    title_block_line_height_list = []
    for line in block.get("lines", []):
        # Implementation detail.
        bbox = line.get("bbox")
        if not bbox or len(bbox) < 4:
            continue
        line_height = bbox[3] - bbox[1]
        if line_height > 0:
            title_block_line_height_list.append(int(line_height))

    if len(title_block_line_height_list) > 0:
        return sum(title_block_line_height_list) / len(title_block_line_height_list)

    bbox = block.get("bbox")
    if bbox and len(bbox) >= 4:
        return max(0, int(bbox[3] - bbox[1]))
    return 0


def _collect_title_block_refs(page_info_list):
    title_block_refs = []
    title_types = set()

    for page_info in page_info_list:
        for block in page_info.get("para_blocks", []):
            block_type = block.get("type")
            if block_type in TITLE_BLOCK_TYPES:
                title_block_refs.append((page_info, block))
                title_types.add(block_type)

    return title_block_refs, title_types


def _build_title_dict(title_block_refs):
    title_dict = {}

    for i, (page_info, block) in enumerate(title_block_refs):
        title_dict[str(i)] = [
            merge_para_with_text(block),
            _get_title_line_avg_height(block),
            int(page_info["page_idx"]) + 1,
        ]

    return title_dict


def _build_title_optimize_prompt(title_dict):
    return f"""\u8f93\u5165\u7684\u5185\u5bb9\u662f\u4e00\u7bc7\u6587\u6863\u4e2d\u6240\u6709\u6807\u9898\u7ec4\u6210\u7684\u5b57\u5178\uff0c\u8bf7\u6839\u636e\u4ee5\u4e0b\u6307\u5357\u4f18\u5316\u6807\u9898\u7684\u7ed3\u679c\uff0c\u4f7f\u7ed3\u679c\u7b26\u5408\u6b63\u5e38\u6587\u6863\u7684\u5c42\u6b21\u7ed3\u6784\uff1a

1. \u5b57\u5178\u4e2d\u6bcf\u4e2avalue\u5747\u4e3a\u4e00\u4e2alist\uff0c\u5305\u542b\u4ee5\u4e0b\u5143\u7d20\uff1a
    - \u6807\u9898\u6587\u672c
    - \u6587\u672c\u884c\u9ad8\u662f\u6807\u9898\u6240\u5728\u5757\u7684\u5e73\u5747\u884c\u9ad8
    - \u6807\u9898\u6240\u5728\u7684\u9875\u7801

2. \u4fdd\u7559\u539f\u59cb\u5185\u5bb9\uff1a
    - \u8f93\u5165\u7684\u5b57\u5178\u4e2d\u6240\u6709\u5143\u7d20\u90fd\u662f\u6709\u6548\u7684\uff0c\u4e0d\u80fd\u5220\u9664\u5b57\u5178\u4e2d\u7684\u4efb\u4f55\u5143\u7d20
    - \u8bf7\u52a1\u5fc5\u4fdd\u8bc1\u8f93\u51fa\u7684\u5b57\u5178\u4e2d\u5143\u7d20\u7684\u6570\u91cf\u548c\u8f93\u5165\u7684\u6570\u91cf\u4e00\u81f4

3. \u4fdd\u6301\u5b57\u5178\u5185key-value\u7684\u5bf9\u5e94\u5173\u7cfb\u4e0d\u53d8

4. \u4f18\u5316\u5c42\u6b21\u7ed3\u6784\uff1a
    - \u6839\u636e\u6807\u9898\u5185\u5bb9\u7684\u8bed\u4e49\u4e3a\u6bcf\u4e2a\u6807\u9898\u5143\u7d20\u6dfb\u52a0\u9002\u5f53\u7684\u5c42\u6b21\u7ed3\u6784
    - \u884c\u9ad8\u8f83\u5927\u7684\u6807\u9898\u4e00\u822c\u662f\u66f4\u9ad8\u7ea7\u522b\u7684\u6807\u9898
    - \u6807\u9898\u4ece\u524d\u81f3\u540e\u7684\u5c42\u7ea7\u5fc5\u987b\u662f\u8fde\u7eed\u7684\uff0c\u4e0d\u80fd\u8df3\u8fc7\u5c42\u7ea7
    - \u6807\u9898\u5c42\u7ea7\u6700\u591a\u4e3a4\u7ea7\uff0c\u4e0d\u8981\u6dfb\u52a0\u8fc7\u591a\u7684\u5c42\u7ea7
    - \u4f18\u5316\u540e\u7684\u6807\u9898\u53ea\u4fdd\u7559\u4ee3\u8868\u8be5\u6807\u9898\u7684\u5c42\u7ea7\u7684\u6574\u6570\uff0c\u4e0d\u8981\u4fdd\u7559\u5176\u4ed6\u4fe1\u606f

5. \u5408\u7406\u6027\u68c0\u67e5\u4e0e\u5fae\u8c03\uff1a
    - \u5728\u5b8c\u6210\u521d\u6b65\u5206\u7ea7\u540e\uff0c\u4ed4\u7ec6\u68c0\u67e5\u5206\u7ea7\u7ed3\u679c\u7684\u5408\u7406\u6027
    - \u6839\u636e\u4e0a\u4e0b\u6587\u5173\u7cfb\u548c\u903b\u8f91\u987a\u5e8f\uff0c\u5bf9\u4e0d\u5408\u7406\u7684\u5206\u7ea7\u8fdb\u884c\u5fae\u8c03
    - \u786e\u4fdd\u6700\u7ec8\u7684\u5206\u7ea7\u7ed3\u679c\u7b26\u5408\u6587\u6863\u7684\u5b9e\u9645\u7ed3\u6784\u548c\u903b\u8f91

IMPORTANT:
\u8bf7\u76f4\u63a5\u8fd4\u56de\u4f18\u5316\u8fc7\u7684\u7531\u6807\u9898\u5c42\u7ea7\u7ec4\u6210\u7684\u5b57\u5178\uff0c\u683c\u5f0f\u4e3a{{\u6807\u9898id:\u6807\u9898\u5c42\u7ea7}}\uff0c\u5982\u4e0b\uff1a
{{
  0:1,
  1:2,
  2:2,
  3:3
}}
\u4e0d\u9700\u8981\u5bf9\u5b57\u5178\u683c\u5f0f\u5316\uff0c\u4e0d\u9700\u8981\u8fd4\u56de\u4efb\u4f55\u5176\u4ed6\u4fe1\u606f\u3002

Input title list:
{title_dict}

Corrected title list:
"""


def _build_relative_title_optimize_prompt(title_dict):
    return f"""\u8f93\u5165\u5185\u5bb9\u662f\u67d0\u4e00\u7bc7\u6587\u6863\u4e2d\u9664\u6587\u7ae0\u6807\u9898\u5916\u7684\u5168\u90e8\u7ae0\u8282/\u6bb5\u843d\u6807\u9898\u7ec4\u6210\u7684\u5b57\u5178\u3002

\u8bf7\u6ce8\u610f\uff1a
- \u6587\u7ae0\u6807\u9898\u4e0d\u5728\u672c\u6b21\u8f93\u5165\u4e2d\uff0c\u5df2\u7ecf\u7531\u7cfb\u7edf\u5355\u72ec\u8bc6\u522b\u5e76\u8bbe\u7f6e\u4e3a1\u7ea7\u6807\u9898

1. \u5b57\u5178\u4e2d\u6bcf\u4e2avalue\u5747\u4e3a\u4e00\u4e2alist\uff0c\u5305\u542b\u4ee5\u4e0b\u5143\u7d20\uff1a
    - \u6807\u9898\u6587\u672c
    - \u6587\u672c\u884c\u9ad8\u662f\u6807\u9898\u6240\u5728\u5757\u7684\u5e73\u5747\u884c\u9ad8
    - \u6807\u9898\u6240\u5728\u7684\u9875\u7801

2. \u4fdd\u7559\u539f\u59cb\u5185\u5bb9\uff1a
    - \u8f93\u5165\u7684\u5b57\u5178\u4e2d\u6240\u6709\u5143\u7d20\u90fd\u662f\u6709\u6548\u7684\uff0c\u4e0d\u80fd\u5220\u9664\u5b57\u5178\u4e2d\u7684\u4efb\u4f55\u5143\u7d20
    - \u8bf7\u52a1\u5fc5\u4fdd\u8bc1\u8f93\u51fa\u7684\u5b57\u5178\u4e2d\u5143\u7d20\u7684\u6570\u91cf\u548c\u8f93\u5165\u7684\u6570\u91cf\u4e00\u81f4

3. \u4fdd\u6301\u5b57\u5178\u5185key-value\u7684\u5bf9\u5e94\u5173\u7cfb\u4e0d\u53d8

4. \u4f18\u5316\u5c42\u6b21\u7ed3\u6784\uff1a
    - \u6839\u636e\u6807\u9898\u5185\u5bb9\u7684\u8bed\u4e49\u4e3a\u6bcf\u4e2a\u6807\u9898\u5143\u7d20\u6dfb\u52a0\u9002\u5f53\u7684\u5c42\u6b21\u7ed3\u6784
    - \u884c\u9ad8\u8f83\u5927\u7684\u6807\u9898\u4e00\u822c\u662f\u66f4\u9ad8\u7ea7\u522b\u7684\u6807\u9898
    - \u6807\u9898\u4ece\u524d\u81f3\u540e\u7684\u5c42\u7ea7\u5fc5\u987b\u662f\u8fde\u7eed\u7684\uff0c\u4e0d\u80fd\u8df3\u8fc7\u5c42\u7ea7
    - \u6807\u9898\u5c42\u7ea7\u6700\u591a\u4e3a4\u7ea7\uff0c\u4e0d\u8981\u6dfb\u52a0\u8fc7\u591a\u7684\u5c42\u7ea7
    - \u4f18\u5316\u540e\u7684\u6807\u9898\u53ea\u4fdd\u7559\u4ee3\u8868\u8be5\u6807\u9898\u7684\u5c42\u7ea7\u7684\u6574\u6570\uff0c\u4e0d\u8981\u4fdd\u7559\u5176\u4ed6\u4fe1\u606f

5. \u5408\u7406\u6027\u68c0\u67e5\u4e0e\u5fae\u8c03\uff1a
    - \u5728\u5b8c\u6210\u521d\u6b65\u5206\u7ea7\u540e\uff0c\u4ed4\u7ec6\u68c0\u67e5\u5206\u7ea7\u7ed3\u679c\u7684\u5408\u7406\u6027
    - \u6839\u636e\u4e0a\u4e0b\u6587\u5173\u7cfb\u548c\u903b\u8f91\u987a\u5e8f\uff0c\u5bf9\u4e0d\u5408\u7406\u7684\u5206\u7ea7\u8fdb\u884c\u5fae\u8c03
    - \u786e\u4fdd\u6700\u7ec8\u7684\u5206\u7ea7\u7ed3\u679c\u7b26\u5408\u6587\u6863\u7684\u5b9e\u9645\u7ed3\u6784\u548c\u903b\u8f91

IMPORTANT:
\u8bf7\u76f4\u63a5\u8fd4\u56de\u4f18\u5316\u540e\u7684\u6807\u9898\u5c42\u7ea7\u5b57\u5178\uff0c\u683c\u5f0f\u4e3a{{\u6807\u9898id:\u6807\u9898\u5c42\u7ea7}}\uff0c\u5982\u4e0b\uff1a
{{
  0:1,
  1:2,
  2:2,
  3:3
}}
\u4e0d\u8981\u8fd4\u56de Markdown\uff0c\u4e0d\u8981\u8fd4\u56de\u4ee3\u7801\u5757\uff0c\u4e0d\u8981\u8fd4\u56de\u4efb\u4f55\u89e3\u91ca\u6587\u5b57\u3002

Input title list:
{title_dict}

Corrected title list:
"""


def _request_title_levels(title_aided_config, title_dict, prompt_builder=None):
    if len(title_dict) == 0:
        return {}

    client = OpenAI(
        api_key=title_aided_config["api_key"],
        base_url=title_aided_config["base_url"],
    )

    retry_count = 0
    max_retries = 3
    expected_keys = set(range(len(title_dict)))
    if prompt_builder is None:
        prompt_builder = _build_title_optimize_prompt
    title_optimize_prompt = prompt_builder(title_dict)

    logger.debug(f"Requesting LLM for title optimization with prompt: {title_optimize_prompt}")

    api_params = {
        "model": title_aided_config["model"],
        "messages": [{"role": "user", "content": title_optimize_prompt}],
        "temperature": 0.7,
        "stream": True,
    }
    if "enable_thinking" in title_aided_config:
        api_params["extra_body"] = {
            "enable_thinking": title_aided_config["enable_thinking"]
        }

    while retry_count < max_retries:
        try:
            completion = client.chat.completions.create(**api_params)
            content_pieces = []
            for chunk in completion:
                if chunk.choices and chunk.choices[0].delta.content is not None:
                    content_pieces.append(chunk.choices[0].delta.content)

            content = "".join(content_pieces).strip()
            if "</think>" in content:
                idx = content.index("</think>") + len("</think>")
                content = content[idx:].strip()

            logger.debug(f"Raw LLM output for title levels: {content}")
            dict_completion = json_repair.loads(content)
            dict_completion = {int(k): int(v) for k, v in dict_completion.items()}

            if set(dict_completion.keys()) == expected_keys:
                return dict_completion

            logger.warning(
                "The keys in the optimized title result do not match the input titles."
            )
        except Exception as e:
            logger.exception(e)

        retry_count += 1

    logger.error("Failed to decode dict after maximum retries.")
    return None


def _apply_levels_to_blocks(title_block_refs, levels_by_index):
    if levels_by_index is None:
        return

    for i, (_, block) in enumerate(title_block_refs):
        block["level"] = int(levels_by_index[i])


def _normalize_title_types(title_block_refs):
    for _, block in title_block_refs:
        if block.get("type") in [BlockType.DOC_TITLE, BlockType.PARAGRAPH_TITLE]:
            block["type"] = BlockType.TITLE


def _get_title_block_identity(block):
    block_index = block.get("index")
    if block_index is not None:
        return ("index", block_index)

    return (
        "bbox_text",
        tuple(block.get("bbox", [])),
        merge_para_with_text(block),
    )


def _sync_para_titles_to_preproc(page_info_list):
    for page_info in page_info_list:
        para_title_map = {}
        for block in page_info.get("para_blocks", []):
            if block.get("type") in TITLE_BLOCK_TYPES:
                para_title_map[_get_title_block_identity(block)] = block

        if len(para_title_map) == 0:
            continue

        for block in page_info.get("preproc_blocks", []):
            if block.get("type") not in TITLE_BLOCK_TYPES:
                continue

            para_block = para_title_map.get(_get_title_block_identity(block))
            if para_block is None:
                continue

            block["type"] = para_block.get("type", block.get("type"))
            if "level" in para_block:
                block["level"] = para_block["level"]


def _run_single_pass_title_leveling(title_block_refs, title_aided_config):
    title_dict = _build_title_dict(title_block_refs)
    levels_by_index = _request_title_levels(title_aided_config, title_dict)
    _apply_levels_to_blocks(title_block_refs, levels_by_index)


def _split_paragraph_title_groups(title_block_refs):
    groups = []
    current_group = []

    for title_ref in title_block_refs:
        _, block = title_ref
        if block.get("type") == BlockType.DOC_TITLE:
            if current_group:
                groups.append(current_group)
                current_group = []
        elif block.get("type") == BlockType.PARAGRAPH_TITLE:
            current_group.append(title_ref)

    if current_group:
        groups.append(current_group)

    return groups


def _offset_paragraph_title_levels(levels_by_index):
    if not levels_by_index:
        return levels_by_index

    return {
        index: 2 if level == 1 else level
        for index, level in levels_by_index.items()
    }


def _request_paragraph_group_levels(title_block_refs, title_aided_config):
    title_dict = _build_title_dict(title_block_refs)
    levels_by_index = _request_title_levels(
        title_aided_config,
        title_dict,
        prompt_builder=_build_relative_title_optimize_prompt,
    )
    return _offset_paragraph_title_levels(levels_by_index)


def _run_grouped_title_leveling(title_block_refs, title_aided_config):
    doc_title_refs = []
    for title_ref in title_block_refs:
        _, block = title_ref
        if block.get("type") == BlockType.DOC_TITLE:
            block["level"] = 1
            doc_title_refs.append(title_ref)

    paragraph_title_groups = _split_paragraph_title_groups(title_block_refs)
    group_levels = []

    if len(paragraph_title_groups) > 1:
        max_workers = min(len(paragraph_title_groups), MAX_TITLE_GROUP_WORKERS)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    _request_paragraph_group_levels,
                    title_group,
                    title_aided_config,
                )
                for title_group in paragraph_title_groups
            ]
            group_levels = [future.result() for future in futures]
    else:
        group_levels = [
            _request_paragraph_group_levels(title_group, title_aided_config)
            for title_group in paragraph_title_groups
        ]

    for title_group, levels_by_index in zip(paragraph_title_groups, group_levels):
        _apply_levels_to_blocks(title_group, levels_by_index)

    _normalize_title_types(doc_title_refs)
    for title_group in paragraph_title_groups:
        _normalize_title_types(title_group)


def llm_aided_title(page_info_list, title_aided_config):
    title_block_refs, title_types = _collect_title_block_refs(page_info_list)
    if len(title_block_refs) == 0:
        logger.info("No titles detected, skipping LLM-aided title optimization.")
        return

    has_doc_title = BlockType.DOC_TITLE in title_types
    has_paragraph_title = BlockType.PARAGRAPH_TITLE in title_types
    has_generic_title = BlockType.TITLE in title_types

    if has_doc_title and has_paragraph_title and not has_generic_title:
        _run_grouped_title_leveling(title_block_refs, title_aided_config)
        _sync_para_titles_to_preproc(page_info_list)
        return

    doc_title_refs = []
    title_refs_for_llm = []
    for title_ref in title_block_refs:
        _, block = title_ref
        if block.get("type") == BlockType.DOC_TITLE:
            block["level"] = 1
            doc_title_refs.append(title_ref)
        else:
            title_refs_for_llm.append(title_ref)

    if len(title_refs_for_llm) > 0:
        _run_single_pass_title_leveling(title_refs_for_llm, title_aided_config)

    _normalize_title_types(doc_title_refs)
    _normalize_title_types(title_refs_for_llm)
    _sync_para_titles_to_preproc(page_info_list)
