# Copyright (c) Opendatalab. All rights reserved.
from typing import Dict, List, Tuple

import numpy as np


class TableRecover:
    def __init__(
        self,
    ):
        pass

    def __call__(
        self, polygons: np.ndarray, rows_thresh=10, col_thresh=15
    ) -> Dict[int, Dict]:
        rows = self.get_rows(polygons, rows_thresh)
        longest_col, each_col_widths, col_nums = self.get_benchmark_cols(
            rows, polygons, col_thresh
        )
        each_row_heights, row_nums = self.get_benchmark_rows(rows, polygons)
        table_res, logic_points_dict = self.get_merge_cells(
            polygons,
            rows,
            row_nums,
            col_nums,
            longest_col,
            each_col_widths,
            each_row_heights,
        )
        logic_points = np.array(
            [logic_points_dict[i] for i in range(len(polygons))]
        ).astype(np.int32)
        return table_res, logic_points

    @staticmethod
    def get_rows(polygons: np.array, rows_thresh=10) -> Dict[int, List[int]]:
        """Implementation detail."""
        y_axis = polygons[:, 0, 1]
        if y_axis.size == 1:
            return {0: [0]}

        concat_y = np.array(list(zip(y_axis, y_axis[1:])))
        minus_res = concat_y[:, 1] - concat_y[:, 0]

        result = {}
        split_idxs = np.argwhere(abs(minus_res) > rows_thresh).squeeze()
        # Implementation detail.
        if split_idxs.size == 0:
            return {0: [i for i in range(len(y_axis))]}
        if split_idxs.ndim == 0:
            split_idxs = split_idxs[None, ...]

        if max(split_idxs) != len(minus_res):
            split_idxs = np.append(split_idxs, len(minus_res))

        start_idx = 0
        for row_num, idx in enumerate(split_idxs):
            if row_num != 0:
                start_idx = split_idxs[row_num - 1] + 1
            result.setdefault(row_num, []).extend(range(start_idx, idx + 1))

        # Calculate the result.
        return result

    def get_benchmark_cols(
        self, rows: Dict[int, List], polygons: np.ndarray, col_thresh=15
    ) -> Tuple[np.ndarray, List[float], int]:
        longest_col = max(rows.values(), key=lambda x: len(x))
        longest_col_points = polygons[longest_col]
        longest_x_start = list(longest_col_points[:, 0, 0])
        longest_x_end = list(longest_col_points[:, 2, 0])
        min_x = longest_x_start[0]
        max_x = longest_x_end[-1]

        # Implementation detail.
        # Implementation detail.
        def update_longest_col(col_x_list, cur_v, min_x_, max_x_, insert_last):
            for i, v in enumerate(col_x_list):
                if cur_v - col_thresh <= v <= cur_v + col_thresh:
                    break
                if cur_v < min_x_:
                    col_x_list.insert(0, cur_v)
                    min_x_ = cur_v
                    break
                if cur_v > max_x_:
                    if insert_last:
                        col_x_list.append(cur_v)
                    max_x_ = cur_v
                    break
                if cur_v < v:
                    col_x_list.insert(i, cur_v)
                    break
            return min_x_, max_x_

        for row_value in rows.values():
            cur_row_start = list(polygons[row_value][:, 0, 0])
            cur_row_end = list(polygons[row_value][:, 2, 0])
            for idx, (cur_v_start, cur_v_end) in enumerate(
                zip(cur_row_start, cur_row_end)
            ):
                min_x, max_x = update_longest_col(
                    longest_x_start, cur_v_start, min_x, max_x, True
                )
                min_x, max_x = update_longest_col(
                    longest_x_start, cur_v_end, min_x, max_x, False
                )

        longest_x_start = np.array(longest_x_start)
        each_col_widths = (longest_x_start[1:] - longest_x_start[:-1]).tolist()
        each_col_widths.append(max_x - longest_x_start[-1])
        col_nums = longest_x_start.shape[0]
        return longest_x_start, each_col_widths, col_nums

    def get_benchmark_rows(
        self, rows: Dict[int, List], polygons: np.ndarray
    ) -> Tuple[np.ndarray, List[float], int]:
        leftmost_cell_idxs = [v[0] for v in rows.values()]
        benchmark_x = polygons[leftmost_cell_idxs][:, 0, 1]

        each_row_widths = (benchmark_x[1:] - benchmark_x[:-1]).tolist()

        # Implementation detail.
        bottommost_idxs = list(rows.values())[-1]
        bottommost_boxes = polygons[bottommost_idxs]
        # Implementation detail.
        max_height = max([self.compute_L2(v[1, :], v[0, :]) for v in bottommost_boxes])
        each_row_widths.append(max_height)

        row_nums = benchmark_x.shape[0]
        return each_row_widths, row_nums

    @staticmethod
    def compute_L2(a1: np.ndarray, a2: np.ndarray) -> float:
        return np.linalg.norm(a2 - a1)

    def get_merge_cells(
        self,
        polygons: np.ndarray,
        rows: Dict,
        row_nums: int,
        col_nums: int,
        longest_col: np.ndarray,
        each_col_widths: List[float],
        each_row_heights: List[float],
    ) -> Dict[int, Dict[int, int]]:
        col_res_merge, row_res_merge = {}, {}
        logic_points = {}
        merge_thresh = 10
        for cur_row, col_list in rows.items():
            one_col_result, one_row_result = {}, {}
            for one_col in col_list:
                box = polygons[one_col]
                box_width = self.compute_L2(box[3, :], box[0, :])

                # Implementation detail.
                loc_col_idx = np.argmin(np.abs(longest_col - box[0, 0]))
                col_start = max(sum(one_col_result.values()), loc_col_idx)

                # Calculate the result.
                for i in range(col_start, col_nums):
                    col_cum_sum = sum(each_col_widths[col_start : i + 1])
                    if i == col_start and col_cum_sum > box_width:
                        one_col_result[one_col] = 1
                        break
                    elif abs(col_cum_sum - box_width) <= merge_thresh:
                        one_col_result[one_col] = i + 1 - col_start
                        break
                    # Implementation detail.
                    elif col_cum_sum > box_width:
                        idx = (
                            i
                            if abs(col_cum_sum - box_width)
                            < abs(col_cum_sum - each_col_widths[i] - box_width)
                            else i - 1
                        )
                        one_col_result[one_col] = idx + 1 - col_start
                        break
                else:
                    one_col_result[one_col] = col_nums - col_start
                col_end = one_col_result[one_col] + col_start - 1
                box_height = self.compute_L2(box[1, :], box[0, :])
                row_start = cur_row
                for j in range(row_start, row_nums):
                    row_cum_sum = sum(each_row_heights[row_start : j + 1])
                    # Implementation detail.
                    # Implementation detail.
                    if j == row_start and row_cum_sum > box_height:
                        one_row_result[one_col] = 1
                        break
                    elif abs(box_height - row_cum_sum) <= merge_thresh:
                        one_row_result[one_col] = j + 1 - row_start
                        break
                    # Implementation detail.
                    elif row_cum_sum > box_height:
                        idx = (
                            j
                            if abs(row_cum_sum - box_height)
                            < abs(row_cum_sum - each_row_heights[j] - box_height)
                            else j - 1
                        )
                        one_row_result[one_col] = idx + 1 - row_start
                        break
                else:
                    one_row_result[one_col] = row_nums - row_start
                row_end = one_row_result[one_col] + row_start - 1
                logic_points[one_col] = np.array(
                    [row_start, row_end, col_start, col_end]
                )
            col_res_merge[cur_row] = one_col_result
            row_res_merge[cur_row] = one_row_result

        res = {}
        for i, (c, r) in enumerate(zip(col_res_merge.values(), row_res_merge.values())):
            res[i] = {k: [cc, r[k]] for k, cc in c.items()}
        return res, logic_points
