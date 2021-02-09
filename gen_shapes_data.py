from PIL import Image, ImageDraw
import itertools
    
def makeImgOneShape(x,y,col,shape,exampletype):
    img = Image.new('RGB', (32, 32), color = (0,0,0))
    draw = ImageDraw.Draw(img)
    if shape=="circle":
        draw.ellipse((x, y, x+10, y+10), fill=col, outline=col)
    if shape=="square":
        draw.rectangle([(x, y), (x+10, y+10)], fill=col, outline=col)
    if shape=="triangle":
        draw.polygon([(x, y), (x+5, y+5),(x+10, y)], fill=col, outline=col)
    img.save(exampletype+"_"+str(x)+"_"+str(y)+"_"+col+"_"+shape+'.png')

def makeImgTwoShapes(x,y,x2,y2,col,shape,col2,shape2,exampletype):
    img = Image.new('RGB', (32, 32), color = (0,0,0))
    draw = ImageDraw.Draw(img)
    if shape=="circle":
        draw.ellipse((x, y, x+10, y+10), fill=col, outline=col)
    if shape=="square":
        draw.rectangle([(x, y), (x+10, y+10)], fill=col, outline=col)
    if shape=="triangle":
        draw.polygon([(x, y), (x+5, y+5),(x+10, y)], fill=col, outline=col)
    if shape2=="circle":
        draw.ellipse((x2, y2, x2+10, y2+10), fill=col2, outline=col2)
    if shape2=="square":
        draw.rectangle([(x2, y2), (x2+10, y2+10)], fill=col2, outline=col2)
    if shape2=="triangle":
        draw.polygon([(x2, y2), (x2+5, y2+5),(x2+10, y2)], fill=col2, outline=col2)
    img.save(exampletype+"_"+str(x)+"_"+str(y)+"_"+col+"_"+shape+"_"+str(x2)+"_"+str(y2)+"_"+col2+"_"+shape2+'.png')

xcs=[0,10,20]
ycs=[0,10,20]
colors=['blue','green','red']
shapes=["circle","square","triangle"]

#one shape#####################################
exampletype = "single_allcolors_allshapes"
for x in xcs:
    for y in ycs:
        for col in colors:
            for shape in shapes:
                makeImgOneShape(x,y,col,shape,exampletype)

#two shapes#####################################
exampletype2 = "double_allcolors_allshapes"
twoshapes = list(itertools.product(xcs,ycs,xcs,ycs,colors,colors,shapes,shapes))

for x,x2,y,y2,color,color2,shape,shape2 in twoshapes:
    if not (x==x2 and y==y2):
        makeImgTwoShapes(x,x2,y,y2,color,shape,color2,shape2,exampletype2)