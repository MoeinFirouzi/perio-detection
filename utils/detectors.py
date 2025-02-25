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


# def extract_cej(polygon, axes):
#     return find_lower_intersections_for_shape(polygon, axes)


# def extract_bone_level(polygon, axes):
#     return find_higher_intersections_for_shape(polygon, axes)


# def extract_tooth_root(major_axes, polygons):
#     return [get_lowest_coordinate(polygon) for polygon in polygons]


def extract_tooth_root(polygon, parallel_axes):
    # roots = []

    lowest_point = get_lowest_coordinate(polygon)
    minor_axis = get_minor_axis_from_point(polygon, lowest_point)
    # left_axis, right_axis = get_parallel_major_axes(polygon)
    left_axis, right_axis = parallel_axes
    left_root = minor_axis.intersection(left_axis)
    right_root = minor_axis.intersection(right_axis)
        # roots.append((left_root, right_root))

    return left_root, right_root


def extract_cej(polygon, parallel_axes):
    # cej_levels = []
    # for pair in parallel_axes:
    cej_points = find_lower_intersections_for_shape(polygon, parallel_axes)
        # cej_levels.append((cej_points[0], cej_points[1]))

    return cej_points[0], cej_points[1]


def extract_bone_level(polygon, parallel_axes):
    # bone_levels = []
    # for pair in parallel_axes:
    bone_points = find_higher_intersections_for_shape(polygon, parallel_axes)
        # bone_levels.append((bone_points[0], bone_points[1]))

    return bone_points[0], bone_points[1]
