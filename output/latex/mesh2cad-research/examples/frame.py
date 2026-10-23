import cadquery as cq
outer_x = 0.58333331
outer_y = 0.062500022
outer_z = 1.0
inner_x = 0.5
inner_z = 0.9375
r = cq.Workplane("XY").box(
    outer_x, outer_y, outer_z)
opening = cq.Workplane("XY").box(
    inner_x, outer_y + 0.02, inner_z)
r = r.cut(opening)
