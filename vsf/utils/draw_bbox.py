# Copyright (c) Opendatalab. All rights reserved.
import json
from io import BytesIO

from loguru import logger
from pypdf import PdfReader, PdfWriter, PageObject
from reportlab.pdfgen import canvas

from .enum_class import BlockType, ContentType, SplitFlag


# Process text content.
TEXT_LIKE_BLOCK_TYPES_FOR_BBOX = {
    BlockType.TEXT,
    BlockType.REF_TEXT,
    BlockType.ABSTRACT,
    BlockType.PHONETIC,
}

# Implementation detail.
DIRECT_LAYOUT_BBOX_BLOCK_TYPES = TEXT_LIKE_BLOCK_TYPES_FOR_BBOX | {
    BlockType.TITLE,
    BlockType.INTERLINE_EQUATION,
    BlockType.LIST,
    BlockType.INDEX,
}

# Implementation detail.
SPAN_SOURCE_BLOCK_TYPES = DIRECT_LAYOUT_BBOX_BLOCK_TYPES


def _get_layout_source_blocks(page):
    """Extract the required value."""
    return page.get("preproc_blocks", [])


def cal_canvas_rect(page, bbox):
    """
    Calculate the rectangle coordinates on the canvas based on the original PDF page and bounding box.

    Args:
        page: A PyPDF2 Page object representing a single page in the PDF.
        bbox: [x0, y0, x1, y1] representing the bounding box coordinates.

    Returns:
        rect: [x0, y0, width, height] representing the rectangle coordinates on the canvas.
    """
    page_width, page_height = float(page.cropbox[2]), float(page.cropbox[3])
    
    actual_width = page_width    # The width of the final PDF display
    actual_height = page_height  # The height of the final PDF display
    
    rotation_obj = page.get("/Rotate", 0)
    try:
        rotation = int(rotation_obj) % 360  # cast rotation to int to handle IndirectObject
    except (ValueError, TypeError) as e:
        logger.warning(f"Invalid /Rotate value {rotation_obj!r} on page; defaulting to 0. Error: {e}")
        rotation = 0
    
    if rotation in [90, 270]:
        # PDF is rotated 90 degrees or 270 degrees, and the width and height need to be swapped
        actual_width, actual_height = actual_height, actual_width
        
    x0, y0, x1, y1 = bbox
    rect_w = abs(x1 - x0)
    rect_h = abs(y1 - y0)
    
    if rotation == 270:
        rect_w, rect_h = rect_h, rect_w
        x0 = actual_height - y1
        y0 = actual_width - x1
    elif rotation == 180:
        x0 = page_width - x1
        # y0 stays the same
    elif rotation == 90:
        rect_w, rect_h = rect_h, rect_w
        x0, y0 = y0, x0 
    else:
        # rotation == 0
        y0 = page_height - y1
    
    rect = [x0, y0, rect_w, rect_h]        
    return rect


def draw_bbox_without_number(i, bbox_list, page, c, rgb_config, fill_config):
    new_rgb = [float(color) / 255 for color in rgb_config]
    page_data = bbox_list[i]

    for bbox in page_data:
        rect = cal_canvas_rect(page, bbox)  # Define the rectangle  

        if fill_config:  # filled rectangle
            c.setFillColorRGB(new_rgb[0], new_rgb[1], new_rgb[2], 0.3)
            c.rect(rect[0], rect[1], rect[2], rect[3], stroke=0, fill=1)
        else:  # bounding box
            c.setStrokeColorRGB(new_rgb[0], new_rgb[1], new_rgb[2])
            c.rect(rect[0], rect[1], rect[2], rect[3], stroke=1, fill=0)
    return c


def draw_bbox_with_number(i, bbox_list, page, c, rgb_config, fill_config, draw_bbox=True):
    new_rgb = [float(color) / 255 for color in rgb_config]
    page_data = bbox_list[i]
    # Convert the value to the required format.
    page_width, page_height = float(page.cropbox[2]), float(page.cropbox[3])

    for j, bbox in enumerate(page_data):
        # Implementation detail.
        rect = cal_canvas_rect(page, bbox)  # Define the rectangle  
        
        if draw_bbox:
            if fill_config:
                c.setFillColorRGB(*new_rgb, 0.3)
                c.rect(rect[0], rect[1], rect[2], rect[3], stroke=0, fill=1)
            else:
                c.setStrokeColorRGB(*new_rgb)
                c.rect(rect[0], rect[1], rect[2], rect[3], stroke=1, fill=0)
        c.setFillColorRGB(*new_rgb, 1.0)
        c.setFontSize(size=10)
        
        c.saveState()
        rotation_obj = page.get("/Rotate", 0)
        try:
            rotation = int(rotation_obj) % 360  # cast rotation to int to handle IndirectObject
        except (ValueError, TypeError):
            logger.warning(f"Invalid /Rotate value: {rotation_obj!r}, defaulting to 0")
            rotation = 0

        if rotation == 0:
            c.translate(rect[0] + rect[2] + 2, rect[1] + rect[3] - 10)
        elif rotation == 90:
            c.translate(rect[0] + 10, rect[1] + rect[3] + 2)
        elif rotation == 180:
            c.translate(rect[0] - 2, rect[1] + 10)
        elif rotation == 270:
            c.translate(rect[0] + rect[2] - 10, rect[1] - 2)
            
        c.rotate(rotation)
        c.drawString(0, 0, str(j + 1))
        c.restoreState()

    return c


def draw_layout_bbox(pdf_info, pdf_bytes, out_path, filename):
    dropped_bbox_list = []
    tables_body_list, tables_caption_list, tables_footnote_list = [], [], []
    imgs_body_list, imgs_caption_list, imgs_footnote_list = [], [], []
    codes_body_list, codes_caption_list, codes_footnote_list = [], [], []
    titles_list = []
    texts_list = []
    interline_equations_list = []
    lists_list = []
    list_items_list = []
    indexs_list = []

    for page in pdf_info:
        page_dropped_list = []
        tables_body, tables_caption, tables_footnote = [], [], []
        imgs_body, imgs_caption, imgs_footnote = [], [], []
        codes_body, codes_caption, codes_footnote = [], [], []
        titles = []
        texts = []
        interline_equations = []
        lists = []
        list_items = []
        indices = []

        for dropped_bbox in page['discarded_blocks']:
            page_dropped_list.append(dropped_bbox['bbox'])
        dropped_bbox_list.append(page_dropped_list)
        for block in _get_layout_source_blocks(page):
            bbox = block["bbox"]
            if block["type"] == BlockType.TABLE:
                for nested_block in block["blocks"]:
                    bbox = nested_block["bbox"]
                    if nested_block["type"] == BlockType.TABLE_BODY:
                        tables_body.append(bbox)
                    elif nested_block["type"] == BlockType.TABLE_CAPTION:
                        tables_caption.append(bbox)
                    elif nested_block["type"] == BlockType.TABLE_FOOTNOTE:
                        if nested_block.get(SplitFlag.CROSS_PAGE, False):
                            continue
                        tables_footnote.append(bbox)
            elif block["type"] == BlockType.IMAGE:
                for nested_block in block["blocks"]:
                    bbox = nested_block["bbox"]
                    if nested_block["type"] == BlockType.IMAGE_BODY:
                        imgs_body.append(bbox)
                    elif nested_block["type"] == BlockType.IMAGE_CAPTION:
                        imgs_caption.append(bbox)
                    elif nested_block["type"] == BlockType.IMAGE_FOOTNOTE:
                        imgs_footnote.append(bbox)
            elif block["type"] == BlockType.CODE:
                for nested_block in block["blocks"]:
                    if nested_block["type"] == BlockType.CODE_BODY:
                        bbox = nested_block["bbox"]
                        codes_body.append(bbox)
                    elif nested_block["type"] == BlockType.CODE_CAPTION:
                        bbox = nested_block["bbox"]
                        codes_caption.append(bbox)
                    elif nested_block["type"] == BlockType.CODE_FOOTNOTE:
                        bbox = nested_block["bbox"]
                        codes_footnote.append(bbox)
            elif block["type"] == BlockType.CHART:
                for nested_block in block["blocks"]:
                    if nested_block["type"] == BlockType.CHART_BODY:
                        bbox = nested_block["bbox"]
                        imgs_body.append(bbox)
                    elif nested_block["type"] == BlockType.CHART_CAPTION:
                        bbox = nested_block["bbox"]
                        imgs_caption.append(bbox)
                    elif nested_block["type"] == BlockType.CHART_FOOTNOTE:
                        bbox = nested_block["bbox"]
                        imgs_footnote.append(bbox)
            elif block["type"] == BlockType.TITLE:
                titles.append(bbox)
            elif block["type"] in TEXT_LIKE_BLOCK_TYPES_FOR_BBOX:
                texts.append(bbox)
            elif block["type"] == BlockType.INTERLINE_EQUATION:
                interline_equations.append(bbox)
            elif block["type"] == BlockType.LIST:
                lists.append(bbox)
                if "blocks" in block:
                    for sub_block in block["blocks"]:
                        list_items.append(sub_block["bbox"])
            elif block["type"] == BlockType.INDEX:
                indices.append(bbox)

        tables_body_list.append(tables_body)
        tables_caption_list.append(tables_caption)
        tables_footnote_list.append(tables_footnote)
        imgs_body_list.append(imgs_body)
        imgs_caption_list.append(imgs_caption)
        imgs_footnote_list.append(imgs_footnote)
        titles_list.append(titles)
        texts_list.append(texts)
        interline_equations_list.append(interline_equations)
        lists_list.append(lists)
        list_items_list.append(list_items)
        indexs_list.append(indices)
        codes_body_list.append(codes_body)
        codes_caption_list.append(codes_caption)
        codes_footnote_list.append(codes_footnote)

    layout_bbox_list = []

    for page in pdf_info:
        page_block_list = []
        for block in _get_layout_source_blocks(page):
            if block["type"] in DIRECT_LAYOUT_BBOX_BLOCK_TYPES:
                bbox = block["bbox"]
                page_block_list.append(bbox)
            elif block["type"] in [BlockType.IMAGE, BlockType.CHART, BlockType.CODE, BlockType.TABLE]:
                for sub_block in block["blocks"]:
                    if sub_block.get(SplitFlag.CROSS_PAGE, False):
                        continue
                    bbox = sub_block["bbox"]
                    page_block_list.append(bbox)

        layout_bbox_list.append(page_block_list)

    pdf_bytes_io = BytesIO(pdf_bytes)
    pdf_docs = PdfReader(pdf_bytes_io)
    output_pdf = PdfWriter()

    for i, page in enumerate(pdf_docs.pages):
        # Extract the required value.
        page_width, page_height = float(page.cropbox[2]), float(page.cropbox[3])
        custom_page_size = (page_width, page_height)

        packet = BytesIO()
        # Build the required output.
        c = canvas.Canvas(packet, pagesize=custom_page_size)

        c = draw_bbox_without_number(i, codes_body_list, page, c, [102, 0, 204], True)
        c = draw_bbox_without_number(i, codes_caption_list, page, c, [204, 153, 255], True)
        c = draw_bbox_without_number(i, codes_footnote_list, page, c, [229, 204, 255], True)
        c = draw_bbox_without_number(i, dropped_bbox_list, page, c, [158, 158, 158], True)
        c = draw_bbox_without_number(i, tables_body_list, page, c, [204, 204, 0], True)
        c = draw_bbox_without_number(i, tables_caption_list, page, c, [255, 255, 102], True)
        c = draw_bbox_without_number(i, tables_footnote_list, page, c, [229, 255, 204], True)
        c = draw_bbox_without_number(i, imgs_body_list, page, c, [153, 255, 51], True)
        c = draw_bbox_without_number(i, imgs_caption_list, page, c, [102, 178, 255], True)
        c = draw_bbox_without_number(i, imgs_footnote_list, page, c, [255, 178, 102], True)
        c = draw_bbox_without_number(i, titles_list, page, c, [102, 102, 255], True)
        c = draw_bbox_without_number(i, texts_list, page, c, [153, 0, 76], True)
        c = draw_bbox_without_number(i, interline_equations_list, page, c, [0, 255, 0], True)
        c = draw_bbox_without_number(i, lists_list, page, c, [40, 169, 92], True)
        c = draw_bbox_without_number(i, list_items_list, page, c, [40, 169, 92], False)
        c = draw_bbox_without_number(i, indexs_list, page, c, [40, 169, 92], True)
        c = draw_bbox_with_number(i, layout_bbox_list, page, c, [255, 0, 0], False, draw_bbox=False)

        c.save()
        packet.seek(0)
        overlay_pdf = PdfReader(packet)

        # Validate the current value.
        if len(overlay_pdf.pages) > 0:
            new_page = PageObject(pdf=None)
            new_page.update(page)
            page = new_page
            page.merge_page(overlay_pdf.pages[0])
        else:
            # Process the current item.
            # Build the required output.
            pass

        output_pdf.add_page(page)

    # Add the value to the result.
    with open(f"{out_path}/{filename}", "wb") as f:
        output_pdf.write(f)


def draw_span_bbox(pdf_info, pdf_bytes, out_path, filename):
    text_list = []
    inline_equation_list = []
    interline_equation_list = []
    image_list = []
    table_list = []
    dropped_list = []

    def get_span_info(span):
        if span['type'] == ContentType.TEXT:
            page_text_list.append(span['bbox'])
        elif span['type'] == ContentType.INLINE_EQUATION:
            page_inline_equation_list.append(span['bbox'])
        elif span['type'] == ContentType.INTERLINE_EQUATION:
            page_interline_equation_list.append(span['bbox'])
        elif span['type'] in [ContentType.IMAGE, ContentType.CHART]:
            page_image_list.append(span['bbox'])
        elif span['type'] == ContentType.TABLE:
            page_table_list.append(span['bbox'])

    for page in pdf_info:
        page_text_list = []
        page_inline_equation_list = []
        page_interline_equation_list = []
        page_image_list = []
        page_table_list = []
        page_dropped_list = []


        # Implementation detail.
        for block in page['discarded_blocks']:
            for line in block['lines']:
                for span in line['spans']:
                    page_dropped_list.append(span['bbox'])
        dropped_list.append(page_dropped_list)
        # Implementation detail.
        # Merge the related values.
        for block in page['preproc_blocks']:
            if block['type'] in SPAN_SOURCE_BLOCK_TYPES:
                for line in block['lines']:
                    for span in line['spans']:
                        get_span_info(span)
            elif block['type'] in [BlockType.IMAGE, BlockType.TABLE, BlockType.CHART, BlockType.CODE]:
                for sub_block in block['blocks']:
                    for line in sub_block['lines']:
                        for span in line['spans']:
                            get_span_info(span)
        text_list.append(page_text_list)
        inline_equation_list.append(page_inline_equation_list)
        interline_equation_list.append(page_interline_equation_list)
        image_list.append(page_image_list)
        table_list.append(page_table_list)

    pdf_bytes_io = BytesIO(pdf_bytes)
    pdf_docs = PdfReader(pdf_bytes_io)
    output_pdf = PdfWriter()

    for i, page in enumerate(pdf_docs.pages):
        # Extract the required value.
        page_width, page_height = float(page.cropbox[2]), float(page.cropbox[3])
        custom_page_size = (page_width, page_height)

        packet = BytesIO()
        # Build the required output.
        c = canvas.Canvas(packet, pagesize=custom_page_size)

        # Extract the required value.
        draw_bbox_without_number(i, text_list, page, c,[255, 0, 0], False)
        draw_bbox_without_number(i, inline_equation_list, page, c, [0, 255, 0], False)
        draw_bbox_without_number(i, interline_equation_list, page, c, [0, 0, 255], False)
        draw_bbox_without_number(i, image_list, page, c, [255, 204, 0], False)
        draw_bbox_without_number(i, table_list, page, c, [204, 0, 255], False)
        draw_bbox_without_number(i, dropped_list, page, c, [158, 158, 158], False)

        c.save()
        packet.seek(0)
        overlay_pdf = PdfReader(packet)

        # Validate the current value.
        if len(overlay_pdf.pages) > 0:
            new_page = PageObject(pdf=None)
            new_page.update(page)
            page = new_page
            page.merge_page(overlay_pdf.pages[0])
        else:
            # Process the current item.
            # Build the required output.
            pass

        output_pdf.add_page(page)

    # Save the PDF
    with open(f"{out_path}/{filename}", "wb") as f:
        output_pdf.write(f)


if __name__ == "__main__":
    # Extract the required value.
    pdf_path = "examples/demo1.pdf"
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    # Extract the required value.

    json_path = "examples/demo1_1746005777.0863056_middle.json"
    with open(json_path, "r", encoding="utf-8") as f:
        pdf_ann = json.load(f)
    pdf_info = pdf_ann["pdf_info"]
    # Process the file path.
    draw_layout_bbox(pdf_info, pdf_bytes, "examples", "output_with_layout.pdf")
