#-------------------------------
# imports
#-------------------------------
import os,sys,time,traceback
from math import hypot
import numpy as np
import cv2
import depthai as dai
import frame_capture
import frame_draw
from datetime import date
import time

today = date.today()

def getFrame(queue):
  # Get frame from queue
  frame = queue.get()
  # Convert frame to OpenCV format and return
  return frame.getCvFrame()


def getMonoCamera(pipeline, isLeft):
  # Configure mono camera
  mono = pipeline.createMonoCamera()

  # Set Camera Resolution
  mono.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

  if isLeft:
      # Get left camera
      mono.setBoardSocket(dai.CameraBoardSocket.LEFT)
  else :
      # Get right camera
      mono.setBoardSocket(dai.CameraBoardSocket.RIGHT)
  return mono


def getStereoPair(pipeline, monoLeft, monoRight):
    # Configure stereo pair for depth estimation
    stereo = pipeline.createStereoDepth()
    # Checks occluded pixels and marks them as invalid
    stereo.setLeftRightCheck(True)
    
    # Configure left and right cameras to work as a stereo pair
    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)

    return stereo

def mouseCallback(event,x,y,flags,param):
    global mouseX, mouseY
    if event == cv2.EVENT_LBUTTONDOWN:
        mouseX = x
        mouseY = y

#-------------------------------
# conversion (pixels to measure)
#-------------------------------
calfile = 'camruler_cal.csv'
# distance units designator
unit_suffix = 'mm'

# calibrate every N pixels
pixel_base = 10

# maximum field of view from center to farthest edge
# should be measured in unit_suffix 
cal_range = 72

# initial calibration values table {pixels:scale}
# this is based on the frame size and the cal_range

# calibration loop values
# inside of main loop below
cal_base = 5
cal_last = None

weightvalue = 0 
clawsize = 0
carapacelength = 0

# convert pixels to units
def conv(x,y):

    d = distance(0,0,x,y)

    scale = cal[baseround(d,pixel_base)]

    return x*scale,y*scale

# round to a given base
def baseround(x,base=1):
    return int(base * round(float(x)/base))

# distance formula 2D
def distance(x1,y1,x2,y2):
    return hypot(x1-x2,y1-y2)

#-------------------------------
# define frames
#-------------------------------

# define display frame
framename = "Camruler"
cv2.namedWindow(framename,flags=cv2.WINDOW_NORMAL|cv2.WINDOW_GUI_NORMAL)

#-------------------------------
# key events
#-------------------------------

key_last = 0
key_flags = {'config':False, # c key
             'auto':False,   # a key
             'thresh':False, # t key
             'percent':False,# p key
             'norms':False,  # n key
             'weight':False, # w Key 
             'clawsize':False, # s Key 
             'carapacelength':False, # l Key 
             'rotate':False, # r key
             'lock':False,   # 
             }

def key_flags_clear():

    global key_flags5

    for key in list(key_flags.keys()):
        if key not in ('rotate',):
            key_flags[key] = False

def key_event(key):

    global key_last
    global key_flags
    global mouse_mark
    global cal_last

    # config mode
    if key == 99:
        if key_flags['config']:
            key_flags['config'] = False
        else:
            key_flags_clear()
            key_flags['config'] = True
            cal_last,mouse_mark = 0,None

    # normilization mode
    elif key == 110:
        if key_flags['norms']:
            key_flags['norms'] = False
        else:
            key_flags['thresh'] = False
            key_flags['percent'] = False
            key_flags['lock'] = False
            key_flags['norms'] = True
            mouse_mark = None

    # rotate
    elif key == 114:
        if key_flags['rotate']:
            key_flags['rotate'] = False
        else:
            key_flags['rotate'] = True
    elif key == 119:
        if key_flags['weight']:
            key_flags['weight'] = False
        else:
            key_flags['weight'] = True
    elif key == 115:
        if key_flags['clawsize']:
            key_flags['clawsize'] = False
        else:
            key_flags['clawsize'] = True
    elif key == 108:
        if key_flags['carapacelength']:
            key_flags['carapacelength'] = False
        else:
            key_flags['carapacelength'] = True

    # auto mode
    elif key == 97:
        if key_flags['auto']:
            key_flags['auto'] = False
        else:
            key_flags_clear()
            key_flags['auto'] = True
            mouse_mark = None

    # auto percent
    elif key == 112 and key_flags['auto']:
        key_flags['percent'] = not key_flags['percent']
        key_flags['thresh'] = False
        key_flags['lock'] = False

    # auto threshold
    elif key == 116 and key_flags['auto']:
        key_flags['thresh'] = not key_flags['thresh']
        key_flags['percent'] = False
        key_flags['lock'] = False

    # log
    print('key:',[key,chr(key)])
    key_last = key
    
#-------------------------------
# mouse events
#-------------------------------

# mouse events
mouse_raw  = (0,0) # pixels from top left
mouse_now  = (0,0) # pixels from center
mouse_mark = None  # last click (from center)

# auto measure mouse events
auto_percent = 0.2 
auto_threshold = 127
auto_blur = 5

# normalization mouse events
norm_alpha = 0
norm_beta = 255

# mouse callback
def mouse_event(event,x,y,flags,parameters):

    # globals
    global mouse_raw
    global mouse_now
    global mouse_mark
    global key_last
    global auto_percent
    global auto_threshold
    global auto_blur
    global norm_alpha
    global norm_beta

    # update percent
    if key_flags['percent']:
        auto_percent = 5*(x/width)*(y/height)

    # update threshold
    elif key_flags['thresh']:
        auto_threshold = int(255*x/width)
        auto_blur = int(20*y/height) | 1 # insure it is odd and at least 1

    # update normalization
    elif key_flags['norms']:
        norm_alpha = int(64*x/width)
        norm_beta  = min(255,int(128+(128*y/height)))

    # update mouse location
    mouse_raw = (x,y)

    # offset from center
    # invert y to standard quadrants
    ox = x - cx
    oy = (y-cy)*-1

    # update mouse location
    mouse_raw = (x,y)
    if not key_flags['lock']:
        mouse_now = (ox,oy)

    # left click event
    if event == 1:

        if key_flags['config']:
            key_flags['lock'] = False
            mouse_mark = (ox,oy)

        elif key_flags['auto']:
            key_flags['lock'] = False
            mouse_mark = (ox,oy)

        if key_flags['percent']:
            key_flags['percent'] = False
            mouse_mark = (ox,oy)
            
        elif key_flags['thresh']:
            key_flags['thresh'] = False
            mouse_mark = (ox,oy)
            
        elif key_flags['norms']:
            key_flags['norms'] = False
            mouse_mark = (ox,oy)

        elif not key_flags['lock']:
            if mouse_mark:
                key_flags['lock'] = True
            else:
                mouse_mark = (ox,oy)
        else:
            key_flags['lock'] = False
            mouse_now = (ox,oy)
            mouse_mark = (ox,oy)

        key_last = 0

    # right click event
    elif event == 2:
        key_flags_clear()
        mouse_mark = None
        key_last = 0

# register mouse callback
cv2.setMouseCallback(framename,mouse_event)

#-------------------------------
# main loop
#-------------------------------

# loop
if __name__ == '__main__':

    mouseX = 0
    mouseY = 640
    # Start defining a pipeline
    pipeline = dai.Pipeline()

    # set up the rgb camera
    cam_rgb = pipeline.createColorCamera()
    # This sets the size of the preview window. Is the resolution always 4k?
    cam_rgb.setPreviewSize(1920, 1080) # (w, h)

    cam_rgb.setInterleaved(False)

    # Set output XLink for the rgb camera
    xout_rgb = pipeline.createXLinkOut()
    xout_rgb.setStreamName("rgb")

    # Attach the camera to the output XLink
    cam_rgb.preview.link(xout_rgb.input)

    # Set up left and right cameras
    monoLeft = getMonoCamera(pipeline, isLeft = True)
    monoRight = getMonoCamera(pipeline, isLeft = False)

    # Combine left and right cameras to form a stereo pair
    stereo = getStereoPair(pipeline, monoLeft, monoRight)

    
    # Set XlinkOut for disparity, rectifiedLeft, and rectifiedRight
    xoutDisp = pipeline.createXLinkOut()
    xoutDisp.setStreamName("disparity")
    
    xoutRectifiedLeft = pipeline.createXLinkOut()
    xoutRectifiedLeft.setStreamName("rectifiedLeft")

    xoutRectifiedRight = pipeline.createXLinkOut()
    xoutRectifiedRight.setStreamName("rectifiedRight")

    stereo.disparity.link(xoutDisp.input)
    
    stereo.rectifiedLeft.link(xoutRectifiedLeft.input)
    stereo.rectifiedRight.link(xoutRectifiedRight.input)

    # Pipe line is defined, now we can connect to the device
    with dai.Device(pipeline) as device:
        
        # get output queue
        rgb_queue = device.getOutputQueue(name="rgb", maxSize=1)
        
        # Output queues will be used to get the rgb frames and nn data from the outputs defined above
        disparityQueue = device.getOutputQueue(name="disparity", maxSize=1, blocking=False)
        rectifiedLeftQueue = device.getOutputQueue(name="rectifiedLeft", maxSize=1, blocking=False)
        rectifiedRightQueue = device.getOutputQueue(name="rectifiedRight", maxSize=1, blocking=False)


        # Calculate a multiplier for colormapping disparity map
        disparityMultiplier = 255 / stereo.getMaxDisparity()
        
        # Variable use to toggle between side by side view and one frame view.
        sideBySide = False

        while 1:
            # Get disparity map
            disparity = getFrame(disparityQueue)
            
            # Colormap disparity for display
            disparity = (disparity * disparityMultiplier).astype(np.uint8)
            disparity = cv2.applyColorMap(disparity, cv2.COLORMAP_JET)
            
            # Get left and right rectified frame
            leftFrame = getFrame(rectifiedLeftQueue)
            rightFrame = getFrame(rectifiedRightQueue)
            
            if sideBySide:
                # Show side by side view
                imOut = np.hstack((leftFrame, rightFrame))
            else :
                # Show overlapping frames
                imOut = np.uint8(leftFrame/2 + rightFrame/2)
            
            
            imOut = cv2.cvtColor(imOut,cv2.COLOR_GRAY2RGB) 
            
            imOut = cv2.line(imOut, (mouseX, mouseY), (1280, mouseY), (0, 0, 255), 2)
            imOut = cv2.circle(imOut, (mouseX, mouseY), 2, (255, 255, 128), 2)
            cv2.imshow("Depth", disparity)
            
            # tryGet() will return the data or None if there isn't any
            rgb_frame = rgb_queue.tryGet()
            
            if rgb_frame is not None:
                        framec = rgb_frame.getCvFrame()
                        hsv_frame = cv2.cvtColor(framec, cv2.COLOR_BGR2HSV)
                        height, width, _ = framec.shape
                        area = width*height
                        cx = int(width / 2)
                        cy = int(height / 2)
                        dm = hypot(cx,cy)
                        cal = dict([(x,cal_range/dm) for x in range(0,int(dm)+1,pixel_base)])
                        draw = frame_draw.DRAW()
                        draw.width = width
                        draw.height = height
                        def cal_update(x,y,unit_distance):
                            # basics
                            pixel_distance = hypot(x,y)
                            scale = abs(unit_distance/pixel_distance)
                            target = baseround(abs(pixel_distance),pixel_base)

                            # low-high values in distance
                            low  = target*scale - (cal_base/2)
                            high = target*scale + (cal_base/2)

                            # get low start point in pixels
                            start = target
                            if unit_distance <= cal_base:
                                start = 0
                            else:
                                while start*scale > low:
                                    start -= pixel_base

                            # get high stop point in pixels
                            stop = target
                            if unit_distance >= baseround(cal_range,pixel_base):
                                high = max(cal.keys())
                            else:
                                while stop*scale < high:
                                    stop += pixel_base

                            # set scale
                            for x in range(start,stop+1,pixel_base):
                                cal[x] = scale
                                print(f'CAL: {x} {scale}')

                            # read local calibration data
                            if os.path.isfile(calfile):
                                with open(calfile) as f:
                                    for line in f:
                                        line = line.strip()
                                        if line and line[0] in ('d',):
                                            axis,pixels,scale = [_.strip() for _ in line.split(',',2)]
                                            if axis == 'd':
                                                print(f'LOAD: {pixels} {scale}')
                                                cal[int(pixels)] = float(scale)

                        # Pick pixel value
                        pixel_center = cv2.mean(hsv_frame)[:3]
                        hue_value = pixel_center[0]

                        color = "Undefined"
                        if hue_value < 5:
                            color = "RED"
                        elif hue_value < 22:
                            color = "ORANGE"
                        elif hue_value < 33:
                            color = "YELLOW"
                        elif hue_value < 78:
                            color = "GREEN"
                        elif hue_value < 131:
                            color = "BLUE"
                        elif hue_value < 170:
                            color = "VIOLET"
                        else:
                            color = "RED"

                        pixel_center_bgr = framec[cy, cx]
                        b, g, r = int(pixel_center_bgr[0]), int(
                            pixel_center_bgr[1]), int(pixel_center_bgr[2])

                        key = cv2.waitKey(1)
                        if key == 27:
                            break
                        # get frame
                        frame0 = framec
                        if frame0 is None:
                            time.sleep(0.1)
                            continue

                        # normalize
                        cv2.normalize(frame0,frame0,norm_alpha,norm_beta,cv2.NORM_MINMAX)

                        # rotate 180
                        if key_flags['rotate']:
                            frame0 = cv2.rotate(frame0,cv2.ROTATE_180)

                        # start top-left text block
                        text = []

                        # mouse text
                        text.append('')
                        if not mouse_mark:
                            text.append(f'LAST CLICK: NONE')
                        else:
                            text.append(f'LAST CLICK: {mouse_mark} PIXELS')
                        text.append(f'CURRENT XY: {mouse_now} PIXELS')

                        #-------------------------------
                        # normalize mode
                        #-------------------------------
                        if key_flags['norms']:

                            # print
                            text.append('')
                            text.append(f'NORMILIZE MODE')
                            text.append(f'ALPHA (min): {norm_alpha}')
                            text.append(f'BETA (max): {norm_beta}')
                            
                        #-------------------------------
                        # config mode
                        #-------------------------------
                        if key_flags['config']:

                            # quadrant crosshairs
                            draw.crosshairs(frame0,5,weight=2,color='red',invert=True)

                            # crosshairs aligned (rotated) to maximum distance 
                            draw.line(frame0,cx,cy, cx+cx, cy+cy,weight=1,color='red')
                            draw.line(frame0,cx,cy, cx+cy, cy-cx,weight=1,color='red')
                            draw.line(frame0,cx,cy,-cx+cx,-cy+cy,weight=1,color='red')
                            draw.line(frame0,cx,cy, cx-cy, cy+cx,weight=1,color='red')

                            # mouse cursor lines (parallel to aligned crosshairs)
                            mx,my = mouse_raw
                            draw.line(frame0,mx,my,mx+dm,my+(dm*( cy/cx)),weight=1,color='green')
                            draw.line(frame0,mx,my,mx-dm,my-(dm*( cy/cx)),weight=1,color='green')
                            draw.line(frame0,mx,my,mx+dm,my+(dm*(-cx/cy)),weight=1,color='green')
                            draw.line(frame0,mx,my,mx-dm,my-(dm*(-cx/cy)),weight=1,color='green')
                        
                            # config text data
                            text.append('')
                            text.append(f'CONFIG MODE')

                            # start cal
                            if not cal_last:
                                cal_last = cal_base
                                caltext = f'CONFIG: Click on D = {cal_last}'

                            # continue cal
                            elif cal_last <= cal_range:
                                if mouse_mark:
                                    cal_update(*mouse_mark,cal_last)
                                    cal_last += cal_base
                                caltext = f'CONFIG: Click on D = {cal_last}'

                            # done
                            else:
                                key_flags_clear()
                                cal_last == None
                                with open(calfile,'w') as f:
                                    data = list(cal.items())
                                    data.sort()
                                    for key,value in data:
                                        f.write(f'd,{key},{value}\n')
                                    f.close()
                                caltext = f'CONFIG: Complete.'

                            # add caltext
                            draw.add_text(frame0,caltext,(cx)+100,(cy)+30,color='red')

                            # clear mouse
                            mouse_mark = None

                        #-------------------------------
                        # WEIGHT MODE
                        #-------------------------------
                        elif key_flags['weight']: 

                            # mouse cursor lines
                            draw.vline(frame0,mouse_raw[0],weight=1,color='green')
                            draw.hline(frame0,mouse_raw[1],weight=1,color='green')
                        
                            # draw
                            if mouse_mark:

                                # locations
                                x1,y1 = mouse_mark
                                x2,y2 = mouse_now

                                # convert to distance
                                x1c,y1c = conv(x1,y1)
                                x2c,y2c = conv(x2,y2)
                                xlen = abs(x1c-x2c)
                                ylen = abs(y1c-y2c)
                                llen = hypot(xlen,ylen)
                                alen = 0
                                if max(xlen,ylen) > 0 and min(xlen,ylen)/max(xlen,ylen) >= 0.95:
                                    alen = (xlen+ylen)/2              
                                carea = xlen*ylen

                                # print distances
                                text.append('')
                                text.append(f'X LEN: {xlen:.2f}{unit_suffix}')
                                text.append(f'Y LEN: {ylen:.2f}{unit_suffix}')
                                text.append(f'L LEN: {llen:.2f}{unit_suffix}')
                                variable1 = llen*llen*llen
                                variable2 = variable1*(0.000483) 
                                weightvalue = variable2
                                text.append(f'Weight: {weightvalue:.2f}g')
                                # convert to plot locations
                                x1 += cx
                                x2 += cx
                                y1 *= -1
                                y2 *= -1
                                y1 += cy
                                y2 += cy
                                x3 = x1+((x2-x1)/2)
                                y3 = max(y1,y2)

                                # line weight
                                weight = 1
                                if key_flags['lock']:
                                    weight = 2

                                # plot
                                draw.rect(frame0,x1,y1,x2,y2,weight=weight,color='red')
                                draw.line(frame0,x1,y1,x2,y2,weight=weight,color='green')

                                # add dimensions
                                if alen:
                                    draw.add_text(frame0,f'Avg: {alen:.2f}',x3,y3+34,center=True,top=True,color='green')           
                                if x2 <= x1:
                                    draw.add_text(frame0,f'{ylen:.2f}',x1+4,(y1+y2)/2,middle=True,color='red')
                                    draw.add_text(frame0,f'{llen:.2f}',x2-4,y2-4,right=True,color='green')
                                else:
                                    draw.add_text(frame0,f'{ylen:.2f}',x1-4,(y1+y2)/2,middle=True,right=True,color='red')
                                    draw.add_text(frame0,f'{llen:.2f}',x2+8,y2-4,color='green')
                        #-------------------------------
                        # Clawsize MODE
                        #-------------------------------
                        elif key_flags['clawsize']:

                            # mouse cursor lines
                            draw.vline(frame0,mouse_raw[0],weight=1,color='green')
                            draw.hline(frame0,mouse_raw[1],weight=1,color='green')
                        
                            # draw
                            if mouse_mark:

                                # locations
                                x1,y1 = mouse_mark
                                x2,y2 = mouse_now

                                # convert to distance
                                x1c,y1c = conv(x1,y1)
                                x2c,y2c = conv(x2,y2)
                                xlen = abs(x1c-x2c)
                                ylen = abs(y1c-y2c)
                                llen = hypot(xlen,ylen)
                                alen = 0
                                if max(xlen,ylen) > 0 and min(xlen,ylen)/max(xlen,ylen) >= 0.95:
                                    alen = (xlen+ylen)/2              
                                carea = xlen*ylen

                                # print distances
                                text.append('')
                                text.append(f'X LEN: {xlen:.2f}{unit_suffix}')
                                text.append(f'Y LEN: {ylen:.2f}{unit_suffix}')
                                text.append(f'L LEN: {llen:.2f}{unit_suffix}')
                                clawsize = llen
                                
                                
                                # convert to plot locations
                                x1 += cx
                                x2 += cx
                                y1 *= -1
                                y2 *= -1
                                y1 += cy
                                y2 += cy
                                x3 = x1+((x2-x1)/2)
                                y3 = max(y1,y2)

                                # line weight
                                weight = 1
                                if key_flags['lock']:
                                    weight = 2

                                # plot
                                draw.rect(frame0,x1,y1,x2,y2,weight=weight,color='red')
                                draw.line(frame0,x1,y1,x2,y2,weight=weight,color='green')

                                # add dimensions
                                if alen:
                                    draw.add_text(frame0,f'Avg: {alen:.2f}',x3,y3+34,center=True,top=True,color='green')           
                                if x2 <= x1:
                                    draw.add_text(frame0,f'{ylen:.2f}',x1+4,(y1+y2)/2,middle=True,color='red')
                                    draw.add_text(frame0,f'{llen:.2f}',x2-4,y2-4,right=True,color='green')
                                else:
                                    draw.add_text(frame0,f'{ylen:.2f}',x1-4,(y1+y2)/2,middle=True,right=True,color='red')
                                    draw.add_text(frame0,f'{llen:.2f}',x2+8,y2-4,color='green')
                        
                        #-------------------------------
                        # Carapace MODE
                        #-------------------------------
                        elif key_flags['carapacelength']:

                            # mouse cursor lines
                            draw.vline(frame0,mouse_raw[0],weight=1,color='green')
                            draw.hline(frame0,mouse_raw[1],weight=1,color='green')
                        
                            # draw
                            if mouse_mark:

                                # locations
                                x1,y1 = mouse_mark
                                x2,y2 = mouse_now

                                # convert to distance
                                x1c,y1c = conv(x1,y1)
                                x2c,y2c = conv(x2,y2)
                                xlen = abs(x1c-x2c)
                                ylen = abs(y1c-y2c)
                                llen = hypot(xlen,ylen)
                                alen = 0
                                if max(xlen,ylen) > 0 and min(xlen,ylen)/max(xlen,ylen) >= 0.95:
                                    alen = (xlen+ylen)/2              
                                carea = xlen*ylen

                                # print distances
                                text.append('')
                                text.append(f'X LEN: {xlen:.2f}{unit_suffix}')
                                text.append(f'Y LEN: {ylen:.2f}{unit_suffix}')
                                text.append(f'L LEN: {llen:.2f}{unit_suffix}')
                                carapacelength = llen
                                
                                
                                # convert to plot locations
                                x1 += cx
                                x2 += cx
                                y1 *= -1
                                y2 *= -1
                                y1 += cy
                                y2 += cy
                                x3 = x1+((x2-x1)/2)
                                y3 = max(y1,y2)

                                # line weight
                                weight = 1
                                if key_flags['lock']:
                                    weight = 2

                                # plot
                                draw.rect(frame0,x1,y1,x2,y2,weight=weight,color='red')
                                draw.line(frame0,x1,y1,x2,y2,weight=weight,color='green')

                                # add dimensions
                                if alen:
                                    draw.add_text(frame0,f'Avg: {alen:.2f}',x3,y3+34,center=True,top=True,color='green')           
                                if x2 <= x1:
                                    draw.add_text(frame0,f'{ylen:.2f}',x1+4,(y1+y2)/2,middle=True,color='red')
                                    draw.add_text(frame0,f'{llen:.2f}',x2-4,y2-4,right=True,color='green')
                                else:
                                    draw.add_text(frame0,f'{ylen:.2f}',x1-4,(y1+y2)/2,middle=True,right=True,color='red')
                                    draw.add_text(frame0,f'{llen:.2f}',x2+8,y2-4,color='green')

                        #-------------------------------
                        # auto mode
                        #-------------------------------
                        elif key_flags['auto']:
                            
                            mouse_mark = None

                            # auto text data
                            text.append('')
                            text.append(f'AUTO MODE')
                            text.append(f'UNITS: {unit_suffix}')
                            text.append(f'MIN PERCENT: {auto_percent:.2f}')
                            text.append(f'THRESHOLD: {auto_threshold}')
                            text.append(f'GAUSS BLUR: {auto_blur}')
                            
                            # gray frame
                            frame1 = cv2.cvtColor(frame0,cv2.COLOR_BGR2GRAY)

                            # blur frame
                            frame1 = cv2.GaussianBlur(frame1,(auto_blur,auto_blur),0)

                            # threshold frame n out of 255 (85 = 33%)
                            frame1 = cv2.threshold(frame1,auto_threshold,255,cv2.THRESH_BINARY)[1]

                            # invert
                            frame1 = ~frame1

                            # find contours on thresholded image
                            contours,nada = cv2.findContours(frame1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                            
                            # small crosshairs (after getting frame1)q
                            draw.crosshairs(frame0,5,weight=2,color='green')    
                        
                            # loop over the contours
                            for c in contours:

                                # contour data (from top left)
                                x1,y1,w,h = cv2.boundingRect(c)
                                x2,y2 = x1+w,y1+h
                                x3,y3 = x1+(w/2),y1+(h/2)

                                # percent area
                                percent = 100*w*h/area
                                
                                # if the contour is too small, ignore it
                                if percent < auto_percent:
                                        continue

                                # if the contour is too large, ignore it
                                elif percent > 60:
                                        continue

                                # convert to center, then distance
                                x1c,y1c = conv(x1-(cx),y1-(cy))
                                x2c,y2c = conv(x2-(cx),y2-(cy))
                                xlen = abs(x1c-x2c)
                                ylen = abs(y1c-y2c)
                                alen = 0
                                if max(xlen,ylen) > 0 and min(xlen,ylen)/max(xlen,ylen) >= 0.95:
                                    alen = (xlen+ylen)/2              
                                carea = xlen*ylen

                                # plot
                                draw.rect(frame0,x1,y1,x2,y2,weight=2,color='red')

                                # add dimensions
                                draw.add_text(frame0,f'{xlen:.2f}{unit_suffix}',x1-((x1-x2)/2),min(y1,y2)-8,center=True,color='red')
                                draw.add_text(frame0,f'Area: {carea:.2f}',x3,y2+8,center=True,top=True,color='red')
                                if alen:
                                    draw.add_text(frame0,f'Avg: {alen:.2f}',x3,y2+34,center=True,top=True,color='green')
                                if x1 < width-x2:
                                    draw.add_text(frame0,f'{ylen:.2f}{unit_suffix}',x2+4,(y1+y2)/2,middle=True,color='red')
                                else:
                                    draw.add_text(frame0,f'{ylen:.2f}{unit_suffix}',x1-4,(y1+y2)/2,middle=True,right=True,color='red')

                        #-------------------------------
                        # dimension mode
                        #-------------------------------
                        else:

                            # small crosshairs
                            draw.crosshairs(frame0,5,weight=2,color='green')    

                            # mouse cursor lines
                            draw.vline(frame0,mouse_raw[0],weight=1,color='green')
                            draw.hline(frame0,mouse_raw[1],weight=1,color='green')
                        
                            # draw
                            if mouse_mark:

                                # locations
                                x1,y1 = mouse_mark
                                x2,y2 = mouse_now

                                # convert to distance
                                x1c,y1c = conv(x1,y1)
                                x2c,y2c = conv(x2,y2)
                                xlen = abs(x1c-x2c)
                                ylen = abs(y1c-y2c)
                                llen = hypot(xlen,ylen)
                                alen = 0
                                if max(xlen,ylen) > 0 and min(xlen,ylen)/max(xlen,ylen) >= 0.95:
                                    alen = (xlen+ylen)/2              
                                carea = xlen*ylen

                                # print distances
                                text.append('')
                                text.append(f'X LEN: {xlen:.2f}{unit_suffix}')
                                text.append(f'Y LEN: {ylen:.2f}{unit_suffix}')
                                text.append(f'L LEN: {llen:.2f}{unit_suffix}')

                                # convert to plot locations
                                x1 += cx
                                x2 += cx
                                y1 *= -1
                                y2 *= -1
                                y1 += cy
                                y2 += cy
                                x3 = x1+((x2-x1)/2)
                                y3 = max(y1,y2)

                                # line weight
                                weight = 1
                                if key_flags['lock']:
                                    weight = 2

                                # plot
                                draw.rect(frame0,x1,y1,x2,y2,weight=weight,color='red')
                                draw.line(frame0,x1,y1,x2,y2,weight=weight,color='green')

                                # add dimensions
                                draw.add_text(frame0,f'{xlen:.2f}',x1-((x1-x2)/2),min(y1,y2)-8,center=True,color='red')
                                draw.add_text(frame0,f'Area: {carea:.2f}',x3,y3+8,center=True,top=True,color='red')
                                if alen:
                                    draw.add_text(frame0,f'Avg: {alen:.2f}',x3,y3+34,center=True,top=True,color='green')           
                                if x2 <= x1:
                                    draw.add_text(frame0,f'{ylen:.2f}',x1+4,(y1+y2)/2,middle=True,color='red')
                                    draw.add_text(frame0,f'{llen:.2f}',x2-4,y2-4,right=True,color='green')
                                else:
                                    draw.add_text(frame0,f'{ylen:.2f}',x1-4,(y1+y2)/2,middle=True,right=True,color='red')
                                    draw.add_text(frame0,f'{llen:.2f}',x2+8,y2-4,color='green')

                        # add usage key
                        text.append('')
                        text.append(f'Q = QUIT')
                        text.append(f'R = ROTATE')
                        text.append(f'N = NORMALIZE')
                        text.append(f'A = AUTO-MODE')
                        text.append(f'W = WEIGHT MODE')
                        text.append(f'S = CLAW SIZE MODE')
                        text.append(f'L = CARAPACE LENGTH')
                        if key_flags['auto']:
                            text.append(f'P = MIN-PERCENT')
                            text.append(f'T = THRESHOLD')
                            text.append(f'T = GAUSS BLUR')
                        text.append(f'C = CONFIG-MODE')
                        
                        # draw top-left text block
                        draw.add_text_top_left(frame0,text)

                        # display
                        cv2.rectangle(frame0, (cx + 555, cy-450), (cx + 940, cy - 175), (0, 0, 0), -1)
                        cv2.putText(frame0, "Date: " + str(today), (cx + 565, cy - 415), 2, 0.9 , (0, 255, 255),2)
                        cv2.putText(frame0, "Time: " + str(time.strftime("%I:%M:%S %p")), (cx + 565, cy - 370), 2, 0.9, (0, 255, 255),2)
                        cv2.putText(frame0, "Color: " + color, (cx + 565, cy - 325), 2, 0.9, (0, 255, 255),2)
                        if (weightvalue == 0):
                            cv2.putText(frame0, f"Weight: Measure", (cx + 565, cy - 280), 2, 0.9, (0, 255, 255),2)
                        else:
                            cv2.putText(frame0, f"Weight: {weightvalue:.2f} g", (cx + 565, cy - 280), 2, 0.9, (0, 255, 255),2)

                        if (clawsize == 0):
                            cv2.putText(frame0, f"Claw Length: Measure", (cx + 565, cy - 235), 2, 0.9, (0, 255, 255),2)
                        else:
                            cv2.putText(frame0, f"Claw Length: {clawsize:.2f} mm", (cx + 565, cy - 235), 2, 0.9, (0, 255, 255),2)
                        
                        if (carapacelength == 0):
                            cv2.putText(frame0, f"Carapace Len: Measure", (cx + 565, cy - 190), 2, 0.9, (0, 255, 255),2)
                        else:
                            cv2.putText(frame0, f"Carapace Len: {carapacelength:.2f} mm", (cx + 565, cy - 190), 2, 0.9, (0, 255, 255),2)

                        cv2.imshow(framename,frame0)

                        # key delay and action
                        key = cv2.waitKey(1) & 0xFF

                        # esc ==  27 == quit
                        # q   == 113 == quit
                        if key in (27,113):
                            break

                        # key data
                        elif key not in (-1,255):
                            key_event(key)

#-------------------------------
# kill sequence
#-------------------------------

# close camera thread
framec.stop()

# close all windows
cv2.destroyAllWindows()

# done
exit()

#-------------------------------
# end
#-------------------------------