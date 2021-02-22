from PIL import Image, ImageDraw
import itertools

xcs = [0, 10, 20]
ycs = [0, 10, 20]
COLORS = ['blue', 'green', 'red']
SHAPES = ["circle", "square", "triangle"]


def make_img_one_shape(x, y, col, shape, size=10, picture_size=32):
    img = Image.new('RGB', (picture_size, picture_size), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)
    if shape == "circle":
        draw.ellipse((x, y, x + size, y + size), fill=col, outline=col)
    if shape == "square":
        draw.rectangle([(x, y), (x + size, y + size)], fill=col, outline=col)
    if shape == "triangle":
        draw.polygon([(x, y), (x + size / 2, y + size / 2), (x + size, y)], fill=col, outline=col)
    return img


def make_img_two_shapes(x, y, x2, y2, col, shape, col2, shape2, picture_size=32):
    img = Image.new('RGB', (picture_size, picture_size), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)
    if shape == "circle":
        draw.ellipse((x, y, x + 10, y + 10), fill=col, outline=col)
    if shape == "square":
        draw.rectangle([(x, y), (x + 10, y + 10)], fill=col, outline=col)
    if shape == "triangle":
        draw.polygon([(x, y), (x + 5, y + 5), (x + 10, y)], fill=col, outline=col)
    if shape2 == "circle":
        draw.ellipse((x2, y2, x2 + 10, y2 + 10), fill=col2, outline=col2)
    if shape2 == "square":
        draw.rectangle([(x2, y2), (x2 + 10, y2 + 10)], fill=col2, outline=col2)
    if shape2 == "triangle":
        draw.polygon([(x2, y2), (x2 + 5, y2 + 5), (x2 + 10, y2)], fill=col2, outline=col2)
    return img


### Each image has a

def make_img_shape_from_center(x_center, y_center, shape, size, col):
    x_1 = x_center - size / 2
    y_1 = y_center - size / 2
    return make_img_one_shape(x_1, y_1, col, shape, size=size)


# # one shape#####################################
# exampletype = "single_allcolors_allshapes"
# for x in xcs:
#     for y in ycs:
#         for col in colors:
#             for shape in shapes:
#                 make_img_one_shape(x, y, col, shape, exampletype)
#
# # two shapes#####################################
# exampletype2 = "double_allcolors_allshapes"
# twoshapes = list(itertools.product(xcs, ycs, xcs, ycs, colors, colors, shapes, shapes))
#
# for x, x2, y, y2, color, color2, shape, shape2 in twoshapes:
#     if not (x == x2 and y == y2):
#         make_img_two_shapes(x, x2, y, y2, color, shape, color2, shape2, exampletype2)
