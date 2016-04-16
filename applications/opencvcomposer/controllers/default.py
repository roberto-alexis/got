import cv2
import numpy as np
import urllib2
import json
import os
import StringIO
import uuid
import datetime
from PIL import Image
from PIL import ImageEnhance

#--------------------------------------
# Endpoints
#--------------------------------------

# Endpoint: GET /
# Response: HTML
# Returns the index view, which presents the user
# with the menu to start uploading images. It connects to FB
# to allow the user pick from a gallery image.
def index():
    return dict()

# Endpoint: GET /image?fileId={file_id}
# Response: file
# Downloads an image that was previously uploaded. It receives the
# id of the image. It will throw a 404 if the image is not found.
def image():
    fileId = request.args(0)
    record = db(db.image.title==fileId).select().first()
    if not record:
        raise HTTP(404, "Unknown fileId")    
    fileName = record.image
    response.headers['Content-Type'] = "image/x-jpg"
    response.headers['Content-disposition'] = 'attachment; filename=image.jpg'
    path = os.path.join(request.folder, 'uploads', fileName)
    res = response.stream(open(path, "rb"), chunk_size=4096)
    return res

# Endpoint: POST /upload
# Body: {
#   image_file: file
# }
# Response: JSON {
#   status: 0 (if everything went right) or 1 (if there was an 
#      error
#   file_id: string (identifier of the file, that can be used
#      at /image?fileId={file_id})
#   message: string (explanation of the error)
# }
# Uploads the given file and runs the face detection algorithm
# over it, saving the final composed image to the database.
# If everything goes right, it returns the file id.
# otherwise it returns an error message.
def upload():
    image_form = FORM(
		INPUT(_name='image_file',_type='file')
		)

    if image_form.accepts(request.vars,formname='image_form'):
        return _storeImage(image_form.vars.image_file.file)

    raise HTTP(400, "Missing file parameter")

# Endpoint GET /loadFromUrl?url={url}
# Response: JSON {
#   status: 0 (if everything went right) or 1 (if there was an 
#      error
#   file_id: string (identifier of the file, that can be used
#      at /image?fileId={file_id})
#   message: string (explanation of the error)
# }
# Uploads a file from the given URL and runs the face 
# detection algorithm over it, saving the final composed image 
# to the database.
# If everything goes right, it returns the file id.
# otherwise it returns an error message.
def loadFromUrl():
    # Masquerade as Mozilla because some web servers may not like python bots.
    hdr = {'User-Agent': 'Mozilla/5.0'}
    req = urllib2.Request(request.vars.url, headers=hdr)
    try:
        file = urllib2.urlopen(req)
        return _storeImage(file)

    except urllib2.HTTPError as err:
        response.json({'status': 1, 'message': str(err)})

# Endpoint: GET or POST /custom
# Body: {
#    image_title: string (title of the image)
#    image_file: file (file to upload)
#    color_hue: string (value of the hue variation between 0 and 360)
#    color_sat: string (color saturation between 0.0 and 1.0)
#    color_light: string (brightness between 0.0 and 1.0)
# Response: HTML
# If used with POST it will upload the given file, run face recogni-
# tion and then apply color variations on the image, to finally store
# it in the database.
# In any case, it return the HTML page with the form to select and
# trigger the upload.
def custom():
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
		image = generateImage(open(path, 'rb'), hue, sat, light, False)
		image.save(path)
		id = db.image.insert(image=filename,title=image_form.vars.image_title)

	images = db().select(db.image.ALL, orderby=~db.image.createat)

	return dict(images=images)

#--------------------------------------
# Private methods
#--------------------------------------

# Runs face recognition and saves the final image, returning
# JSON to the final user.
def _storeImage(file):
    try:
        # Prune any images that are more than 1 hour old
        _pruneDatabase(60)
        # Generate new UUID for this image, based on our own domain
        fileId = str(uuid.uuid5(uuid.NAMESPACE_DNS, 'got06.com'))
        # Save the image as ".jpg" (it doesn't really matter the file
        # extension, as the Image object knows how to convert formats
        # anyways.
        filename = db.image.image.store(file, fileId + ".jpg")
        path = os.path.join(request.folder, 'uploads', filename)
        # Generate and save the final image
        image = _generateImage(open(path, 'rb'), 202, 0.15, 0.97, True)
        image.save(path)
        # Insert record in the database
        db.image.insert(image=filename,title=fileId)
        return response.json({'status': 0, 'file_id': fileId})
    except ValueError as err:
        return response.json({'status': 1, 'message': str(err)})

# Trims the database and removes old images, keeping only the given 
# number of minutes worth of images
def _pruneDatabase(minutes):
    limit = datetime.datetime.now.minusMinutes(minutes)
    recordSet = db(db.image.createat < limit)
    for row in db(db.person.id > 0).select():
        path = os.path.join(request.folder, 'uploads', row.image)
        path.delete
    recordSet.delete()

# Converts RGB encoded images to HSV encoding
def _rgb_to_hsv(rgb):
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

# Converts HSV encoded images to RGB encoding
def _hsv_to_rgb(hsv):
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

# Changes the hue of the image by converting it to HSV, applying an
# offset to all values and then converting it back to RGB
def _hueShift(img, amount):
    arr = np.array(img)
    hsv = _rgb_to_hsv(arr)
    hsv[..., 0] = (hsv[..., 0]+amount) % 1.0
    rgb = _hsv_to_rgb(hsv)
    return Image.fromarray(rgb, 'RGB')

# Generates the final image, taking an image from the given connection
# (con) and applying a hue, saturation and light adjustment.
# It returns an Image object with the given result, or it throws a
# ValueError if no face could be found or if there was any other error.
def _generateImage(con, hue, sat, light, fail):
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
        maskPath = os.path.join(moduledir, "applications", "opencvcomposer", "controllers", "GOT06_MASCARA-v3.png")
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
            pilBlankImage = _hueShift(pilBlankImage, hue / 360.0)
            pilBlankImage.paste(pilMaskImage, (0, 0), pilMaskImage)
            return pilBlankImage
        else:
            if len(faces) > 1 and fail:
                raise ValueError("This picture has more than one face")
            if fail:
                raise ValueError("We couldn't find any faces in this picture")
            # Draw a rectangle around the faces
            output = image.copy()
            for (x, y, w, h) in faces:
                cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 2)
            output = cv2.cvtColor(output,cv2.COLOR_BGR2RGB)
            return Image.fromarray(output)

    except urllib2.HTTPError, e:
        raise ValueError("Internal error processing the image")

