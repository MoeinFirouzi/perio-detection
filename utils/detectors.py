from utils.polygons import (
    get_minor_axis_from_point,
    find_higher_intersections_for_shape,
    find_lower_intersections_for_shape,
    get_lowest_coordinate,
    get_parallel_major_axes,
    find_lower_intersection,
)


# def extract_tooth_root(major_axes, polygons):
#     return find_lower_intersection(major_axes, polygons)


def extract_cej(polygon, axes):
    return find_lower_intersections_for_shape(polygon, axes)


def extract_bone_level(polygon, axes):
    return find_higher_intersections_for_shape(polygon, axes)


# def extract_tooth_root(major_axes, polygons):
#     roots = []

#     for polygon in polygons:
#         lowest_point = get_lowest_coordinate(polygon)
#         minor_axis = get_minor_axis_from_point(polygon, lowest_point)
#         left_axis, right_axis = get_parallel_major_axes(polygon)


def extract_tooth_root(major_axes, polygons):
    return [get_lowest_coordinate(polygon) for polygon in polygons]
