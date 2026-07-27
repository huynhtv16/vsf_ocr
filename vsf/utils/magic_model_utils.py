# Copyright (c) Opendatalab. All rights reserved.
"""
Implementation detail.
"""
from typing import List, Dict, Any, Callable

from loguru import logger
from mineru.utils.boxbase import bbox_distance, bbox_center_distance, is_in


def reduct_overlap(bboxes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Implementation detail.

    Args:
        Implementation detail.

    Returns:
        Implementation detail.
    """
    N = len(bboxes)
    keep = [True] * N
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if is_in(bboxes[i]['bbox'], bboxes[j]['bbox']):
                keep[i] = False
    return [bboxes[i] for i in range(N) if keep[i]]


def tie_up_category_by_index(
        get_subjects_func: Callable,
        get_objects_func: Callable,
        extract_subject_func: Callable = None,
        extract_object_func: Callable = None,
        object_block_type: str = "object",
        include_bbox: bool = True,
):
    """
    Implementation detail.
    Match the expected pattern.
    Implementation detail.
    Implementation detail.
    Implementation detail.

    Implementation detail.
        Extract the required value.
        Extract the required value.
        Extract the required value.
        Extract the required value.

    Prepare the output value.
        Implementation detail.
    """
    subjects = get_subjects_func()
    objects = get_objects_func()

    # Extract the required value.
    if extract_subject_func is None:
        extract_subject_func = lambda x: x
    if extract_object_func is None:
        extract_object_func = lambda x: x

    # Prepare the output value.
    result_dict = {}

    # Initialize the component.
    for i, subject in enumerate(subjects):
        result_dict[i] = {
            "sub_bbox": extract_subject_func(subject),
            "obj_bboxes": [],
            "sub_idx": i,
        }

    # Extract the required value.
    object_indices = set(obj["index"] for obj in objects)

    def calc_effective_index_diff(obj_index: int, sub_index: int) -> int:
        """
        Calculate the result.
        Implementation detail.
        Implementation detail.
        """
        if obj_index == sub_index:
            return 0

        start, end = min(obj_index, sub_index), max(obj_index, sub_index)
        abs_diff = end - start

        # Calculate the result.
        other_objects_count = 0
        for idx in range(start + 1, end):
            if idx in object_indices:
                other_objects_count += 1

        return abs_diff - other_objects_count

    # Match the expected pattern.
    for obj in objects:
        if len(subjects) == 0:
            # Remove invalid or unnecessary data.
            continue

        obj_index = obj["index"]
        min_index_diff = float("inf")
        best_subject_indices = []

        # Implementation detail.
        for i, subject in enumerate(subjects):
            sub_index = subject["index"]
            index_diff = calc_effective_index_diff(obj_index, sub_index)

            if index_diff < min_index_diff:
                min_index_diff = index_diff
                best_subject_indices = [i]
            elif index_diff == min_index_diff:
                best_subject_indices.append(i)

        if len(best_subject_indices) == 1:
            best_subject_idx = best_subject_indices[0]
        # Implementation detail.
        elif len(best_subject_indices) == 2:
            # Match the expected pattern.
            if include_bbox:
                # Calculate the result.
                edge_distances = [(idx, bbox_distance(obj["bbox"], subjects[idx]["bbox"])) for idx in best_subject_indices]
                edge_dist_diff = abs(edge_distances[0][1] - edge_distances[1][1])

                for idx, edge_dist in edge_distances:
                    logger.debug(f"Obj index: {obj_index}, Sub index: {subjects[idx]['index']}, Edge distance: {edge_dist}")

                if edge_dist_diff > 2:
                    # Match the expected pattern.
                    best_subject_idx = min(edge_distances, key=lambda x: x[1])[0]
                    logger.debug(f"Obj index: {obj_index}, edge_dist_diff > 2, matching to subject with min edge distance, index: {subjects[best_subject_idx]['index']}")
                elif object_block_type == "table_caption":
                    # Match the expected pattern.
                    best_subject_idx = max(best_subject_indices, key=lambda idx: subjects[idx]["index"])
                    logger.debug(f"Obj index: {obj_index}, edge_dist_diff <= 2 and table_caption, matching to later subject with index: {subjects[best_subject_idx]['index']}")
                elif object_block_type.endswith("footnote"):
                    # Match the expected pattern.
                    best_subject_idx = min(best_subject_indices, key=lambda idx: subjects[idx]["index"])
                    logger.debug(f"Obj index: {obj_index}, edge_dist_diff <= 2 and footnote, matching to earlier subject with index: {subjects[best_subject_idx]['index']}")
                else:
                    # Match the expected pattern.
                    center_distances = [(idx, bbox_center_distance(obj["bbox"], subjects[idx]["bbox"])) for idx in best_subject_indices]
                    for idx, center_dist in center_distances:
                        logger.debug(f"Obj index: {obj_index}, Sub index: {subjects[idx]['index']}, Center distance: {center_dist}")
                    best_subject_idx = min(center_distances, key=lambda x: x[1])[0]
            else:
                best_subject_idx = best_subject_indices[0]
        else:
            raise ValueError("More than two subjects have the same minimal index difference, which is unexpected.")

        # Add the value to the result.
        result_dict[best_subject_idx]["obj_bboxes"].append(extract_object_func(obj))

    # Sort items into the required order.
    ret = list(result_dict.values())
    ret.sort(key=lambda x: x["sub_idx"])

    return ret
