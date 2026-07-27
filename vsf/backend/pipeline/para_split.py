# Copyright (c) Opendatalab. All rights reserved.
import copy
from vsf.utils.enum_class import ContentType, BlockType, SplitFlag
from vsf.utils.language import detect_lang


LINE_STOP_FLAG = ('.', '!', '?', '\u3002', '\uff01', '\uff1f', ')', '\uff09', '"', '\u201d', ':', '\uff1a', ';', '\uff1b')
LIST_END_FLAG = ('.', '\u3002', ';', '\uff1b')


class ListLineTag:
    IS_LIST_START_LINE = 'is_list_start_line'
    IS_LIST_END_LINE = 'is_list_end_line'


def __process_blocks(blocks):
    # Process the current item.
    # Implementation detail.
    # Implementation detail.

    result = []
    current_group = []

    for i in range(len(blocks)):
        current_block = blocks[i]

        # Implementation detail.
        if current_block['type'] in [BlockType.TEXT, BlockType.INDEX, BlockType.VERTICAL_TEXT]:
            current_block['bbox_fs'] = copy.deepcopy(current_block['bbox'])
            if 'lines' in current_block and len(current_block['lines']) > 0:
                current_block['bbox_fs'] = [
                    min([line['bbox'][0] for line in current_block['lines']]),
                    min([line['bbox'][1] for line in current_block['lines']]),
                    max([line['bbox'][2] for line in current_block['lines']]),
                    max([line['bbox'][3] for line in current_block['lines']]),
                ]
            current_group.append(current_block)

        # Validate the current value.
        if i + 1 < len(blocks):
            next_block = blocks[i + 1]
            # Implementation detail.
            if next_block['type'] in [
                BlockType.ABSTRACT,
                BlockType.INTERLINE_EQUATION,
                BlockType.DOC_TITLE,
                BlockType.PARAGRAPH_TITLE,
            ]:
                result.append(current_group)
                current_group = []

    # Process the current item.
    if current_group:
        result.append(current_group)

    return result


def __is_list_or_index_block(block):
    if block['type'] == BlockType.VERTICAL_TEXT:
        return BlockType.VERTICAL_TEXT
    if block['type'] == BlockType.INDEX:
        for line in block['lines']:
            line[ListLineTag.IS_LIST_START_LINE] = True
        return BlockType.INDEX
    # Implementation detail.
    # Implementation detail.
    # Implementation detail.
    # Implementation detail.

    # Implementation detail.
    # Implementation detail.
    # Implementation detail.
    if len(block['lines']) >= 2:
        first_line = block['lines'][0]
        line_height = first_line['bbox'][3] - first_line['bbox'][1]
        block_weight = block['bbox_fs'][2] - block['bbox_fs'][0]
        block_height = block['bbox_fs'][3] - block['bbox_fs'][1]
        page_weight, page_height = block['page_size']

        left_close_num = 0
        left_not_close_num = 0
        right_not_close_num = 0
        right_close_num = 0
        lines_text_list = []
        center_close_num = 0
        external_sides_not_close_num = 0
        multiple_para_flag = False
        last_line = block['lines'][-1]

        if page_weight == 0:
            block_weight_ratio = 0
        else:
            block_weight_ratio = block_weight / page_weight
        # logger.info(f"block_weight_ratio: {block_weight_ratio}")

        # Implementation detail.
        if (
            first_line['bbox'][0] - block['bbox_fs'][0] > line_height / 2
            and abs(last_line['bbox'][0] - block['bbox_fs'][0]) < line_height / 2
            and block['bbox_fs'][2] - last_line['bbox'][2] > line_height
        ):
            multiple_para_flag = True

        block_text = ''

        for line in block['lines']:
            line_text = ''

            for span in line['spans']:
                span_type = span['type']
                if span_type == ContentType.TEXT:
                    line_text += span['content'].strip()
            # Add the value to the result.
            lines_text_list.append(line_text)
            block_text = ''.join(lines_text_list)

        block_lang = detect_lang(block_text)
        # logger.info(f"block_lang: {block_lang}")

        for line in block['lines']:
            line_mid_x = (line['bbox'][0] + line['bbox'][2]) / 2
            block_mid_x = (block['bbox_fs'][0] + block['bbox_fs'][2]) / 2
            if (
                line['bbox'][0] - block['bbox_fs'][0] > 0.7 * line_height
                and block['bbox_fs'][2] - line['bbox'][2] > 0.7 * line_height
            ):
                external_sides_not_close_num += 1
            if abs(line_mid_x - block_mid_x) < line_height / 2:
                center_close_num += 1

            # Validate the current value.
            if abs(block['bbox_fs'][0] - line['bbox'][0]) < line_height / 2:
                left_close_num += 1
            elif line['bbox'][0] - block['bbox_fs'][0] > line_height:
                left_not_close_num += 1

            # Calculate the result.
            if abs(block['bbox_fs'][2] - line['bbox'][2]) < line_height:
                right_close_num += 1
            else:
                # Implementation detail.
                if block_lang in ['zh', 'ja', 'ko']:
                    closed_area = 0.26 * block_weight
                else:
                    # Implementation detail.
                    # Implementation detail.
                    if block_weight_ratio >= 0.5:
                        closed_area = 0.26 * block_weight
                    else:
                        closed_area = 0.36 * block_weight
                if block['bbox_fs'][2] - line['bbox'][2] > closed_area:
                    right_not_close_num += 1

        # Validate the current value.
        line_end_flag = False
        # Validate the current value.
        line_num_flag = False
        num_start_count = 0
        num_end_count = 0
        flag_end_count = 0

        if len(lines_text_list) > 0:
            for line_text in lines_text_list:
                if len(line_text) > 0:
                    if line_text[-1] in LIST_END_FLAG:
                        flag_end_count += 1
                    if line_text[0].isdigit():
                        num_start_count += 1
                    if line_text[-1].isdigit():
                        num_end_count += 1

            if (
                num_start_count / len(lines_text_list) >= 0.8
                or num_end_count / len(lines_text_list) >= 0.8
            ):
                line_num_flag = True
            if flag_end_count / len(lines_text_list) >= 0.8:
                line_end_flag = True

        # Process the file path.
        if (
            left_close_num / len(block['lines']) >= 0.8
            or right_close_num / len(block['lines']) >= 0.8
        ) and line_num_flag:
            for line in block['lines']:
                line[ListLineTag.IS_LIST_START_LINE] = True
            return BlockType.INDEX

        # Implementation detail.
        # Implementation detail.
        elif (
            external_sides_not_close_num >= 2
            and center_close_num == len(block['lines'])
            and external_sides_not_close_num / len(block['lines']) >= 0.5
            and block_height / block_weight > 0.4
        ):
            for line in block['lines']:
                line[ListLineTag.IS_LIST_START_LINE] = True
            return BlockType.LIST

        elif (
            left_close_num >= 2
            and (right_not_close_num >= 2 or line_end_flag or left_not_close_num >= 2)
            and not multiple_para_flag
            # and block_weight_ratio > 0.27
        ):
            # Validate the current value.
            if left_close_num / len(block['lines']) > 0.8:
                # Implementation detail.
                if flag_end_count == 0 and right_close_num / len(block['lines']) < 0.5:
                    for line in block['lines']:
                        if abs(block['bbox_fs'][0] - line['bbox'][0]) < line_height / 2:
                            line[ListLineTag.IS_LIST_START_LINE] = True
                # Implementation detail.
                elif line_end_flag:
                    for i, line in enumerate(block['lines']):
                        if (
                            len(lines_text_list[i]) > 0
                            and lines_text_list[i][-1] in LIST_END_FLAG
                        ):
                            line[ListLineTag.IS_LIST_END_LINE] = True
                            if i + 1 < len(block['lines']):
                                block['lines'][i + 1][
                                    ListLineTag.IS_LIST_START_LINE
                                ] = True
                # Validate the current value.
                else:
                    line_start_flag = False
                    for i, line in enumerate(block['lines']):
                        if line_start_flag:
                            line[ListLineTag.IS_LIST_START_LINE] = True
                            line_start_flag = False

                        if (
                            abs(block['bbox_fs'][2] - line['bbox'][2])
                            > 0.1 * block_weight
                        ):
                            line[ListLineTag.IS_LIST_END_LINE] = True
                            line_start_flag = True
            # Implementation detail.
            # Implementation detail.
            elif num_start_count >= 2 and num_start_count == flag_end_count:
                for i, line in enumerate(block['lines']):
                    if len(lines_text_list[i]) > 0:
                        if lines_text_list[i][0].isdigit():
                            line[ListLineTag.IS_LIST_START_LINE] = True
                        if lines_text_list[i][-1] in LIST_END_FLAG:
                            line[ListLineTag.IS_LIST_END_LINE] = True
            else:
                # Process the current item.
                for line in block['lines']:
                    if abs(block['bbox_fs'][0] - line['bbox'][0]) < line_height / 2:
                        line[ListLineTag.IS_LIST_START_LINE] = True
                    if abs(block['bbox_fs'][2] - line['bbox'][2]) > line_height:
                        line[ListLineTag.IS_LIST_END_LINE] = True

            return BlockType.LIST
        else:
            return BlockType.TEXT
    else:
        return BlockType.TEXT


def __merge_2_text_blocks(block1, block2):
    if len(block1['lines']) > 0 and len(block2['lines']) > 0:
        first_line = block1['lines'][0]
        line_height = first_line['bbox'][3] - first_line['bbox'][1]
        block1_weight = block1['bbox'][2] - block1['bbox'][0]
        block2_weight = block2['bbox'][2] - block2['bbox'][0]
        min_block_weight = min(block1_weight, block2_weight)
        if abs(block1['bbox_fs'][0] - first_line['bbox'][0]) < line_height / 2:
            last_line = block2['lines'][-1]
            if len(last_line['spans']) > 0:
                last_span = last_line['spans'][-1]
                line_height = last_line['bbox'][3] - last_line['bbox'][1]
                if len(first_line['spans']) > 0:
                    first_span = first_line['spans'][0]
                    if len(first_span['content']) > 0:
                        span_start_with_num = first_span['content'][0].isdigit()
                        span_start_with_big_char = first_span['content'][0].isupper()
                        if (
                            # Implementation detail.
                            abs(block2['bbox_fs'][2] - last_line['bbox'][2]) < line_height
                            # Implementation detail.
                            and not last_span['content'].endswith(LINE_STOP_FLAG)
                            # Merge the related values.
                            and abs(block1_weight - block2_weight) < min_block_weight
                            # Implementation detail.
                            and not span_start_with_num
                            # Implementation detail.
                            and not span_start_with_big_char
                            # Implementation detail.
                            and block1['bbox'][1] < block2['bbox'][3]
                            # Implementation detail.
                            and (len(block1['lines']) > 1 or len(block2['lines']) > 1)
                        ):
                            if block1['page_num'] != block2['page_num']:
                                for line in block1['lines']:
                                    for span in line['spans']:
                                        span[SplitFlag.CROSS_PAGE] = True
                            block2['lines'].extend(block1['lines'])
                            block1['lines'] = []
                            block1[SplitFlag.LINES_DELETED] = True

    return block1, block2


def __merge_2_vertical_text_blocks(block1, block2):
    if len(block1['lines']) > 0 and len(block2['lines']) > 0:
        first_line = block1['lines'][0]
        line_width = first_line['bbox'][2] - first_line['bbox'][0]
        block1_height = block1['bbox'][3] - block1['bbox'][1]
        block2_height = block2['bbox'][3] - block2['bbox'][1]
        min_block_height = min(block1_height, block2_height)
        if line_width > 0 and abs(block1['bbox_fs'][1] - first_line['bbox'][1]) < line_width / 2:
            last_line = block2['lines'][-1]
            if len(last_line['spans']) > 0:
                last_span = last_line['spans'][-1]
                line_width = last_line['bbox'][2] - last_line['bbox'][0]
                if line_width > 0 and len(first_line['spans']) > 0:
                    first_span = first_line['spans'][0]
                    first_content = first_span.get('content', '')
                    last_content = last_span.get('content', '')
                    if len(first_content) > 0:
                        span_start_with_num = first_content[0].isdigit()
                        span_start_with_big_char = first_content[0].isupper()
                        if (
                            abs(block2['bbox_fs'][3] - last_line['bbox'][3]) < line_width
                            and not last_content.endswith(LINE_STOP_FLAG)
                            and abs(block1_height - block2_height) < min_block_height
                            and not span_start_with_num
                            and not span_start_with_big_char
                            # Implementation detail.
                            and block1['bbox'][2] > block2['bbox'][0]
                        ):
                            if block1['page_num'] != block2['page_num']:
                                for line in block1['lines']:
                                    for span in line['spans']:
                                        span[SplitFlag.CROSS_PAGE] = True
                            block2['lines'].extend(block1['lines'])
                            block1['lines'] = []
                            block1[SplitFlag.LINES_DELETED] = True

    return block1, block2


def __merge_2_list_blocks(block1, block2):
    if block1['page_num'] != block2['page_num']:
        for line in block1['lines']:
            for span in line['spans']:
                span[SplitFlag.CROSS_PAGE] = True
    block2['lines'].extend(block1['lines'])
    block1['lines'] = []
    block1[SplitFlag.LINES_DELETED] = True

    return block1, block2


def __is_list_group(text_blocks_group):
    # Implementation detail.
    # Implementation detail.
    for block in text_blocks_group:
        if block['type'] == BlockType.VERTICAL_TEXT:
            return False
        if len(block['lines']) > 3:
            return False
    return True


def __para_merge_page(blocks):
    page_text_blocks_groups = __process_blocks(blocks)
    for text_blocks_group in page_text_blocks_groups:
        if len(text_blocks_group) > 0:
            # Validate the current value.
            for block in text_blocks_group:
                block_type = __is_list_or_index_block(block)
                block['type'] = block_type
                # logger.info(f"{block['type']}:{block}")

        if len(text_blocks_group) > 1:
            # Validate the current value.
            is_list_group = __is_list_group(text_blocks_group)

            # Iterate over the available items.
            for i in range(len(text_blocks_group) - 1, -1, -1):
                current_block = text_blocks_group[i]

                # Validate the current value.
                if i - 1 >= 0:
                    prev_block = text_blocks_group[i - 1]

                    if (
                        current_block['type'] == BlockType.VERTICAL_TEXT
                        and prev_block['type'] == BlockType.VERTICAL_TEXT
                    ):
                        __merge_2_vertical_text_blocks(current_block, prev_block)
                    elif (
                        current_block['type'] == BlockType.TEXT
                        and prev_block['type'] == BlockType.TEXT
                        and not is_list_group
                    ):
                        __merge_2_text_blocks(current_block, prev_block)
                    elif (
                        current_block['type'] == BlockType.LIST
                        and prev_block['type'] == BlockType.LIST
                    ) or (
                        current_block['type'] == BlockType.INDEX
                        and prev_block['type'] == BlockType.INDEX
                    ):
                        __merge_2_list_blocks(current_block, prev_block)

        else:
            continue


def para_split(page_info_list):
    all_blocks = []
    for page_info in page_info_list:
        blocks = copy.deepcopy(page_info['preproc_blocks'])
        for block in blocks:
            block['page_num'] = page_info['page_idx']
            block['page_size'] = page_info['page_size']
        all_blocks.extend(blocks)

    __para_merge_page(all_blocks)
    for page_info in page_info_list:
        page_info['para_blocks'] = []
        for block in all_blocks:
            if 'page_num' in block:
                if block['page_num'] == page_info['page_idx']:
                    if block['type'] == BlockType.VERTICAL_TEXT:
                        block['type'] = BlockType.TEXT
                    page_info['para_blocks'].append(block)
                    # Remove invalid or unnecessary data.
                    del block['page_num']
                    del block['page_size']


if __name__ == '__main__':
    input_blocks = []
    # Implementation detail.
    groups = __process_blocks(input_blocks)
    for group_index, group in enumerate(groups):
        print(f'Group {group_index}: {group}')
