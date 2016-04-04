import cv2
import numpy as np
import urllib2
import json
import os
import StringIO
from PIL import Image
from PIL import ImageEnhance

def download():
    return response.download(request,db)

def link():
    return response.download(request,db,attachment=False)

def delete():
    filename = request.args(0)
    db(db.image.image==filename).delete()
    redirect(URL('index'))

def facebook():
    return dict()

def index():
	image_form = FORM(
		INPUT(_name='image_title',_type='text'),
		INPUT(_name='image_file',_type='file'),
		INPUT(_name='color_hue',_type='text'),
		INPUT(_name='color_sat',_type='text'),
		INPUT(_name='color_light',_type='text')
		)

	if image_form.accepts(request.vars,formname='image_form'):
		filename = db.image.image.store(image_form.vars.image_file.file, image_form.vars.image_file.filename)
		path = os.path.join(request.folder, 'uploads', filename)
		hue = 202 if not image_form.vars.color_hue else int(image_form.vars.color_hue)
		sat = 0.15 if not image_form.vars.color_sat else float(image_form.vars.color_sat)
		light = 0.97 if not image_form.vars.color_light else float(image_form.vars.color_light)
		image = generateImage(open(path, 'rb'), hue, sat, light)
		image.save(path)
		id = db.image.insert(image=filename,title=image_form.vars.image_title)

	images = db().select(db.image.ALL)

	return dict(images=images)

def face_detection():
    # Masquerade as Mozilla because some web servers may not like python bots.
    hdr = {'User-Agent': 'Mozilla/5.0'}
    # Set up the request
    req = urllib2.Request(request.vars.url, headers=hdr)
    try:
        # Obtain the content of the url
        con = urllib2.urlopen(req)
        output = generateImage(con)
        output.save('image.png')

        response.headers['Content-Type'] = "image/x-png"
        response.headers['Content-disposition'] = 'attachment; filename=image.png'
        res = response.stream(open('image.png', "rb"), chunk_size=4096)
        return res

    except urllib2.HTTPError, e:
        return e.fp.read()

def rgb_to_hsv(rgb):
    # Translated from source of colorsys.rgb_to_hsv
    # r,g,b should be a numpy arrays with values between 0 and 255
    # rgb_to_hsv returns an array of floats between 0.0 and 1.0.
    rgb = rgb.astype('float')
    hsv = np.zeros_like(rgb)
    # in case an RGBA array was passed, just copy the A channel
    hsv[..., 3:] = rgb[..., 3:]
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = np.max(rgb[..., :3], axis=-1)
    minc = np.min(rgb[..., :3], axis=-1)
    hsv[..., 2] = maxc
    mask = maxc != minc
    hsv[mask, 1] = (maxc - minc)[mask] / maxc[mask]
    rc = np.zeros_like(r)
    gc = np.zeros_like(g)
    bc = np.zeros_like(b)
    rc[mask] = (maxc - r)[mask] / (maxc - minc)[mask]
    gc[mask] = (maxc - g)[mask] / (maxc - minc)[mask]
    bc[mask] = (maxc - b)[mask] / (maxc - minc)[mask]
    hsv[..., 0] = np.select(
        [r == maxc, g == maxc], [bc - gc, 2.0 + rc - bc], default=4.0 + gc - rc)
    hsv[..., 0] = (hsv[..., 0] / 6.0) % 1.0
    return hsv

def hsv_to_rgb(hsv):
    # Translated from source of colorsys.hsv_to_rgb
    # h,s should be a numpy arrays with values between 0.0 and 1.0
    # v should be a numpy array with values between 0.0 and 255.0
    # hsv_to_rgb returns an array of uints between 0 and 255.
    rgb = np.empty_like(hsv)
    rgb[..., 3:] = hsv[..., 3:]
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    i = (h * 6.0).astype('uint8')
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6
    conditions = [s == 0.0, i == 1, i == 2, i == 3, i == 4, i == 5]
    rgb[..., 0] = np.select(conditions, [v, q, p, p, t, v], default=v)
    rgb[..., 1] = np.select(conditions, [v, v, v, q, p, p], default=t)
    rgb[..., 2] = np.select(conditions, [v, p, t, v, v, q], default=p)
    return rgb.astype('uint8')

def hueShift(img, amount):
    arr = np.array(img)
    hsv = rgb_to_hsv(arr)
    hsv[..., 0] = (hsv[..., 0]+amount) % 1.0
    rgb = hsv_to_rgb(hsv)
    return Image.fromarray(rgb, 'RGB')

def generateImage(con, hue, sat, light):
    # Module folder
    moduledir = os.path.dirname(os.path.abspath('__file__'))
    try:
        # Read the content and convert it into an numpy array
        im_array = np.asarray(bytearray(con.read()), dtype=np.uint8)
        # Convert the numpy array into an image.
        image = cv2.imdecode(im_array, cv2.IMREAD_COLOR)
        # Convert to gray scale and equalize
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(8,8))
        gray = clahe.apply(gray)

        # Load the classifier
        faceCascade = cv2.CascadeClassifier(os.path.join(moduledir, "applications", "opencvcomposer", "controllers", "haarcascade_frontalface_default.xml"))
        # Detect faces in the image (the face must be at least %25 of the image)
        (totalW, totalH) = image.shape[:2]
        minSize = (int(totalW * 0.15), int(totalH * 0.15))
        faces = faceCascade.detectMultiScale(gray, 1.3, 5, 0, minSize)

        # Load the mask
        maskPath = os.path.join(moduledir, "applications", "opencvcomposer", "controllers", "GOT06_MASCARA-v2.png")
        pilMaskImage = Image.open(maskPath)
        (maskW, maskH) = pilMaskImage.size

        # Setting the size and position of the target area
        faceSize = (170, 200)
        facePosition = ((maskW - faceSize[0]) / 2, (maskH - faceSize[1]) / 2 - 5)

        # Verify that this image as a single face
        if len(faces) == 1:
            # Get the single face
            (x, y, w, h) = faces[0]
            extraTop = h * 0.045
            extraBottom = h * 0.075
            face = image[y-extraTop:y+h+extraTop+extraBottom, x:x+w]
            face = cv2.resize(face, faceSize)
            # Creates a blank image with the face in it
            blankImage = np.zeros((maskH,maskW,3), np.uint8)
            blankImage[facePosition[1]:facePosition[1] + faceSize[1], facePosition[0]:facePosition[0] + faceSize[0]] = face
            # Converts to PIL format and pastes the mask on top
            blankImage = cv2.cvtColor(blankImage,cv2.COLOR_BGR2RGB)
            pilBlankImage = Image.fromarray(blankImage)
            pilBlankImage = ImageEnhance.Color(pilBlankImage).enhance(sat)
            pilBlankImage = ImageEnhance.Brightness(pilBlankImage).enhance(light)
            pilBlankImage = hueShift(pilBlankImage, hue / 360.0)
            pilBlankImage.paste(pilMaskImage, (0, 0), pilMaskImage)
            return pilBlankImage
        else:
            # Draw a rectangle around the faces
            output = image.copy()
            for (x, y, w, h) in faces:
                cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 2)
            output = cv2.cvtColor(output,cv2.COLOR_BGR2RGB)
            return Image.fromarray(output)

    except urllib2.HTTPError, e:
        return e.fp.read()
