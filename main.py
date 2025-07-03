import RPi.GPIO as GPIO
import time
import picamera
import smtplib
from email.mime.text import MIMEText
import MySQLdb
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
import re
try:
    from PIL import Image
except ImportError:
    import Image

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)


TRIG = 24
ECHO = 23

TRIG1 = 15
ECHO1 = 14
print("Distance measurment in progress")

GPIO.setup(TRIG1, GPIO.OUT)
GPIO.setup(ECHO1, GPIO.IN)
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)

GPIO.output(TRIG1, False)
GPIO.output(TRIG, False)
print("Waiting for sensor to settle")
time.sleep(2)


a=0
def photo():
    plt.style.use('dark_background')
    img_ori = cv2.imread('fast.jpg')
    height, width, channel = img_ori.shape
    plt.figure(figsize=(12, 10))
    plt.imshow(img_ori,cmap='gray')
    gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)
    plt.figure(figsize=(12,10))
    plt.imshow(gray, cmap='gray')
    img_blurred = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)

    img_blur_thresh = cv2.adaptiveThreshold(
        img_blurred,
        maxValue=255.0,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=19,
        C=9
    )
    img_thresh = cv2.adaptiveThreshold(
        gray,
        maxValue=255.0,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=19,
        C=9
    )

    plt.figure(figsize=(20,20))
    plt.subplot(1,2,1)
    plt.title('Threshold only')
    plt.imshow(img_thresh, cmap='gray')
    plt.subplot(1,2,2)
    plt.title('Blur and Threshold')
    plt.imshow(img_blur_thresh, cmap='gray')



    contours, _ = cv2.findContours(
        img_blur_thresh,
        mode=cv2.RETR_LIST,
        method=cv2.CHAIN_APPROX_SIMPLE
    )

    temp_result = np.zeros((height, width, channel), dtype=np.uint8)

    cv2.drawContours(temp_result, contours=contours, contourIdx=-1, color=(255,255,255))
    plt.figure(figsize=(12, 10))
    plt.imshow(temp_result)

    temp_result = np.zeros((height, width, channel), dtype=np.uint8)

    contours_dict = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(temp_result, pt1=(x, y), pt2=(x + w, y + h), color=(255, 255, 255), thickness=2)

        contours_dict.append({
            'contour': contour,
            'x': x,
            'y': y,
            'w': w,
            'h': h,
            'cx': x + (w / 2),
            'cy': y + (h / 2)
        })
    plt.figure(figsize=(12,10))
    plt.imshow(temp_result, cmap='gray')

    MIN_AREA = 80
    MIN_WIDTH, MIN_HEIGHT = 2, 8
    MIN_RATIO, MAX_RATIO = 0.25, 1.0

    possible_contours = []

    cnt = 0
    for d in contours_dict:
        area = d['w'] * d['h']
        ratio = d['w'] / d['h']

        if area > MIN_AREA \
                and d['w'] > MIN_WIDTH and d['h'] > MIN_HEIGHT \
                and MIN_RATIO < ratio < MAX_RATIO:
            d['idx'] = cnt
            cnt += 1
            possible_contours.append(d)

    temp_result = np.zeros((height, width, channel), dtype=np.uint8)

    for d in possible_contours:
        cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x'] + d['w'], d['y'] + d['h']), color=(255, 255, 255),
                      thickness=2)

    plt.figure(figsize=(12, 10))
    plt.imshow(temp_result, cmap='gray')

    MAX_DIAG_MULTIPLYER = 5
    MAX_ANGLE_DIFF = 12.0
    MAX_AREA_DIFF = 0.6
    MAX_WIDTH_DIFF = 0.8
    MAX_HEIGHT_DIFF = 0.2
    MIN_N_MATCHED = 3

    def find_chars(contour_list):
        matched_result_idx = []

        for d1 in contour_list:
            matched_contours_idx = []
            for d2 in contour_list:
                if d1['idx'] == d2['idx']:
                    continue

                dx = abs(d1['cx'] - d2['cx'])
                dy = abs(d1['cy'] - d2['cy'])

                diagonal_length1 = np.sqrt(d1['w'] ** 2 + d1['h'] ** 2)

                distance = np.linalg.norm(np.array([d1['cx'], d1['cy']]) - np.array([d2['cx'], d2['cy']]))
                if dx == 0:
                    angle_diff = 90
                else:
                    angle_diff = np.degrees(np.arctan(dy / dx))
                area_diff = abs(d1['w'] * d1['h'] - d2['w'] * d2['h']) / (d1['w'] * d1['h'])
                width_diff = abs(d1['w'] - d2['w']) / d1['w']
                height_diff = abs(d1['h'] - d2['h']) / d1['h']

                if distance < diagonal_length1 * MAX_DIAG_MULTIPLYER \
                        and angle_diff < MAX_ANGLE_DIFF and area_diff < MAX_AREA_DIFF \
                        and width_diff < MAX_WIDTH_DIFF and height_diff < MAX_HEIGHT_DIFF:
                    matched_contours_idx.append(d2['idx'])

            matched_contours_idx.append(d1['idx'])

            if len(matched_contours_idx) < MIN_N_MATCHED:
                continue

            matched_result_idx.append(matched_contours_idx)

            unmatched_contour_idx = []
            for d4 in contour_list:
                if d4['idx'] not in matched_contours_idx:
                    unmatched_contour_idx.append(d4['idx'])

            unmatched_contour = np.take(possible_contours, unmatched_contour_idx)

            recursive_contour_list = find_chars(unmatched_contour)

            for idx in recursive_contour_list:
                matched_result_idx.append(idx)

            break

        return matched_result_idx

    result_idx = find_chars(possible_contours)

    matched_result = []
    for idx_list in result_idx:
        matched_result.append(np.take(possible_contours, idx_list))

    temp_result = np.zeros((height, width, channel), dtype=np.uint8)

    for r in matched_result:
        for d in r:
            cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x'] + d['w'], d['y'] + d['h']), color=(255, 255, 255),
                          thickness=2)
    plt.figure(figsize=(12, 10))
    plt.imshow(temp_result, cmap='gray')

    PLATE_WIDTH_PADDING = 1.3  # 1.3
    PLATE_HEIGHT_PADDING = 1.5  # 1.5
    MIN_PLATE_RATIO = 3
    MAX_PLATE_RATIO = 10

    plate_imgs = []
    plate_infos = []

    for i, matched_chars in enumerate(matched_result):
        sorted_chars = sorted(matched_chars, key=lambda x: x['cx'])

        plate_cx = (sorted_chars[0]['cx'] + sorted_chars[-1]['cx']) / 2
        plate_cy = (sorted_chars[0]['cy'] + sorted_chars[-1]['cy']) / 2

        plate_width = (sorted_chars[-1]['x'] + sorted_chars[-1]['w'] - sorted_chars[0]['x']) * PLATE_WIDTH_PADDING

        sum_height = 0
        for d in sorted_chars:
            sum_height += d['h']

        plate_height = int(sum_height / len(sorted_chars) * PLATE_HEIGHT_PADDING)

        triangle_height = sorted_chars[-1]['cy'] - sorted_chars[0]['cy']
        triangle_hypotenus = np.linalg.norm(
            np.array([sorted_chars[0]['cx'], sorted_chars[0]['cy']]) -
            np.array([sorted_chars[-1]['cx'], sorted_chars[-1]['cy']])
        )

        angle = np.degrees(np.arcsin(triangle_height / triangle_hypotenus))

        rotation_matrix = cv2.getRotationMatrix2D(center=(plate_cx, plate_cy), angle=angle, scale=1.0)

        img_rotated = cv2.warpAffine(img_thresh, M=rotation_matrix, dsize=(width, height))

        img_cropped = cv2.getRectSubPix(
            img_rotated,
            patchSize=(int(plate_width), int(plate_height)),
            center=(int(plate_cx), int(plate_cy))
        )
        if img_cropped.shape[1] / img_cropped.shape[0] < MIN_PLATE_RATIO or img_cropped.shape[1] / img_cropped.shape[
            0] < MIN_PLATE_RATIO > MAX_PLATE_RATIO:
            continue
        plate_imgs.append(img_cropped)
        plate_infos.append({
            'x': int(plate_cx - plate_width / 2),
            'y': int(plate_cy - plate_height / 2),
            'w': int(plate_width),
            'h': int(plate_height)
        })

    #plt.subplot(len(matched_result), 1, i + 1)
        plt.imshow(img_cropped, cmap='gray')
        longest_idx, longest_text = -1, 0
    plate_chars = []

    for i, plate_img in enumerate(plate_imgs):
        plate_img = cv2.resize(plate_img, dsize=(0, 0), fx=1.6, fy=1.6)
        _, plate_img = cv2.threshold(plate_img, thresh=0.0, maxval=255.0, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # find contours again (same as above)
        contours, _ = cv2.findContours(plate_img, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)

        plate_min_x, plate_min_y = plate_img.shape[1], plate_img.shape[0]
        plate_max_x, plate_max_y = 0, 0

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            area = w * h
            ratio = w / h

            if area > MIN_AREA \
                    and w > MIN_WIDTH and h > MIN_HEIGHT \
                    and MIN_RATIO < ratio < MAX_RATIO:
                if x < plate_min_x:
                    plate_min_x = x
                if y < plate_min_y:
                    plate_min_y = y
                if x + w > plate_max_x:
                    plate_max_x = x + w
                if y + h > plate_max_y:
                    plate_max_y = y + h

        img_result = plate_img[plate_min_y:plate_max_y, plate_min_x:plate_max_x]

        img_result = cv2.GaussianBlur(img_result, ksize=(3, 3), sigmaX=0)
        _, img_result = cv2.threshold(img_result, thresh=0.0, maxval=255.0, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        img_result = cv2.copyMakeBorder(img_result, top=10, bottom=10, left=10, right=10, borderType=cv2.BORDER_CONSTANT,
                                        value=(0, 0, 0))
        cv2.imwrite('plate_th.jpg',img_result)
        kernel=np.ones((3,3),np.uint8)
        er_plate=cv2.erode(img_result,kernel,iterations=1)
        er_invplate=er_plate
        cv2.imwrite('er_plate.jpg',er_invplate)
        result33=pytesseract.image_to_string(Image.open('er_plate.jpg'),lang='kor')
        found=re.search('(.+?)\n',result33).group(1)
        print(found)
   
    
        im = Image.fromarray(np.uint8(img_result))
        im.save("plate2.png")
       
        chars = pytesseract.image_to_string(img_result, lang='kor')

        result_chars = ''
        has_digit = False
        for c in chars:
            if ord('가') <= ord(c) <= ord('힣') or c.isdigit():
                if c.isdigit():
                    has_digit = True
                result_chars += c
  
        plate_chars.append(result_chars)

        if has_digit and len(result_chars) > longest_text:
            longest_idx = i

        plt.imshow(img_result, cmap='gray')

    info = plate_infos[longest_idx]
    chars = plate_chars[longest_idx]




    img_out = img_ori.copy()
    cv2.rectangle(img_out, pt1=(info['x'], info['y']), pt2=(info['x']+info['w'], info['y']+info['h']), color=(255,0,0), thickness=2)
    cv2.imwrite(found + '.png', img_out)
    plt.figure(figsize=(12, 10))
    plt.imshow(img_out)
    return found

    

def blog_mail(name):
    db=MySQLdb.connect("localhost","pi","1234","us")
    cur = db.cursor()
    print("위반한 차번호:"+name)
    car=name

    cur.execute("select name,email from user where carnumber=%s",(name,))
    b=cur.fetchone()

    smtp = smtplib.SMTP('smtp.gmail.com',587)
    smtp.starttls()
    smtp.login('dhgkdud115@gmail.com','tqrqhxqfzyvwlico')

    msg = MIMEText(b[0]+'님은 속도 위반 하셨습니다. 차량번호:'+name + '\n 자세한 정보는 ''경찰교통민원24(이파인)'' 으로 접속 바랍니다.')
    msg['Subject']='경찰교통민원24'
    msg['To']=b[1]
    smtp.sendmail('dhgkdud115@gmail.com',b[1],msg.as_string())
    smtp.quit()
    cur.close()
    db.close()
    print("메일을 보냈습니다.")
    
def cho01():
    GPIO.output(TRIG1, True)
    time.sleep(0.0001)
    GPIO.output(TRIG1, False)
    while GPIO.input(ECHO1) == 0:
        start1 = time.time()
    while GPIO.input(ECHO1) == 1:
        stop1 = time.time()
    check_time1 = stop1 - start1
    distance1 = check_time1 * 34300 / 2
    print("Distance1 : %.1f cm" %distance1)
    time.sleep(0.4)
    return distance1
def cho02():
    GPIO.output(TRIG, True)
    time.sleep(0.0001)
    GPIO.output(TRIG, False)
    while GPIO.input(ECHO) == 0:
        start = time.time()
    while GPIO.input(ECHO) == 1:
        stop = time.time()
    check_time = stop - start
    distance = check_time * 34300 / 2
    print("Distance : %.1f cm" %distance)
    time.sleep(0.4)
    return distance
    


try :
    while True:
        if a==0:
            dis1=cho01()
            
            if dis1 < 11 :
                a=1
                tm1=time.time()
            else:
                tm1=0
                time.sleep(0.0001)
                a=1
            
        if a==1:
            dis2 = cho02()
            if dis2 < 12:
                tm2=time.time()
                a=0
            else:
                tm2=0
                time.sleep(0.0001)
                a=0
        tim=tm2-tm1
        print("time:%.1f"%tim)
        if tim < 0.5 and tim > 0:
            
            with picamera.PiCamera() as camera:
                camera.resolution = (640,480)
                camera.start_preview()
                time.sleep(1)
                camera.capture('fast.jpg')
                camera.stop_preview()  
                name=photo()
                print(name)
                blog_mail(name)
            
        else:
            tim=0.0

            
            
            
except KeyboardInterrupt:
    print("Measurement stopped by User")
    GPIO.cleanup()


