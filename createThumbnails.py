from PIL import Image
import glob, os
import math

size = 64, 64

#path = "../Pictures/skitour+site:gipfelbuch.ch/"
picturePath = "../Pictures/"
folders = os.listdir(picturePath)
print(folders)

for fo in folders:
	fopath = picturePath + fo + "/" #"../Pictures/Schneebrettlawine/"

	for infile in glob.glob(fopath+"*.jpg"):
	    file, ext = os.path.splitext(infile)
	    im = Image.open(infile)
	    
	    dSize = im.width - im.height
	    if dSize > 0:
	    	im = im.crop([0+math.floor(dSize/2),0,im.width-math.ceil(dSize/2)-1,im.height-1])
	    elif dSize < 0:
	    	im = im.crop([0,0-math.floor(dSize/2),im.width-1,im.height+math.ceil(dSize/2)-1])
	    
	    im.thumbnail(size, Image.ANTIALIAS)
	    if not im.size == size:
	    	print(im.size)
	    im.save(file + ".thumbnail", "JPEG")