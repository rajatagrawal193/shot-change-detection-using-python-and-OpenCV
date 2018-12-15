from imutils import face_utils
import numpy as np
import imutils
import cv2
import os
import sys
import shutil
import timeit
import time

def make_directories():
    if os.path.exists('output'):
        shutil.rmtree('output')
    os.mkdir('output')
    frames_url='output/frames'
    difference_images_url='output/difference_images'
    shots_url='output/shots'
    stacked_matrix_url='output'

    if os.path.exists(frames_url):
        shutil.rmtree(frames_url)
    os.mkdir(frames_url)
    if os.path.exists(difference_images_url):
        shutil.rmtree(difference_images_url)
    os.mkdir(difference_images_url)
    if os.path.exists(shots_url):
        shutil.rmtree(shots_url)
    os.mkdir(shots_url)


def detect_shot_change(video_url, frames_url, difference_images_url, shots_url,stacked_matrix_url):
    start = time.time()
    
    cap= cv2.VideoCapture(video_url)
    chunk_size=50    
    images=[]    
    count1=0 #frame count in a chunk
    count2=0 #frame count overall
    success=True 
    while(success):            
        success, image= cap.read()
        if(success):
            image= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image= imutils.resize(image, width=480)
            images.append(image)
            count1+=1
            count2+=1
        if count1==chunk_size:
            print('Processing image {}'.format(count2))
            detect_shot_change_util(video_url, frames_url, difference_images_url, shots_url,stacked_matrix_url, images, count2, count1)
            images=[]   
            images.append(image)    
            count1=1

    detect_shot_change_util(video_url, frames_url, difference_images_url, shots_url,stacked_matrix_url, images, count2, count1)
    cap.release()
    end = time.time() - start       
    print("Time taken for shot Change Detection", end)

def detect_shot_change_util(video_url, frames_url, difference_images_url, shots_url,stacked_matrix_url, images, count2, count1):
    
    files_length=len(images)    
    rows=images[0].shape[0]
    columns= images[0].shape[1]
    temp_count= count2-count1 # Number of already proccessed images
    
    stacked_matrix= np.zeros((len(images)-1, columns), dtype = np.uint8)
    temp_list= np.zeros(500)
    
    for k in range (files_length-1):
        subtracted_matrix= get_subtracted_matrix(images[k], images[k+1], rows, columns)
        # cv2.imwrite(difference_images_url+'/difference_image{}.png'.format(temp_count+k), subtracted_matrix)
        stacked_matrix[k] = np.sum(subtracted_matrix, axis=0)/rows
    
    temp_list= np.sum(stacked_matrix, axis=1)/columns #Sum of subtracted_matrix along 0 axis
    _max=int(np.max(temp_list))

    i=0
    while i< len(temp_list)-1:

        diff=((abs(int(temp_list[i+1])- int(temp_list[i]))/_max))*100
        if diff>60: 
            image =cv2.cvtColor(images[i+2], cv2.COLOR_GRAY2BGR)
            cv2.imwrite(shots_url+'/frame{}.jpg'.format(temp_count+i+2), image)
            i=i+2
        else: i+=1

    

    # cv2.imshow("output", stacked_matrix)
    # cv2.waitKey(0)
    
def convert_to_frames(video_url):
    # To convert the input video to frames
    # cmd='ffmpeg -i '+ video_url+' -filter:v scale=480:-1 '+frames_url+'/frame%3d.png'
    # os.system(cmd)
    cap= cv2.VideoCapture(video_url)
    # print(cap)
    # success, image = cap.read()
    images=[]
    currentFrame = 0
    print(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    success=True 
    while(success):
        # Capture frame-by-frame
        success, image = cap.read()
        if success:
            image= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            image= imutils.resize(image, width=480)
            
            name = 'output/frames/frame' + str(currentFrame) + '.jpg'
            print ('Creating...' + name)
            # cv2.imwrite(name, image)
            images.append(image)
        
        
        currentFrame += 1

    cap.release()
    return images


def list_the_frames(frames_url):
    #List all the frame images from the directory and store it in files and sort them.
    files = os.listdir(frames_url)
    files= sorted(files, key=lambda x:int(x.split('.')[0][5:]))
    # files.sort()
    print(files)

    return files

def get_subtracted_matrix(image1, image2, rows, columns):
    lLimit= 25 
    rLimit= 100
    subtracted_matrix= np.zeros((rows,columns), dtype = np.uint8)
    subtracted_matrix= abs(image2-image1)
    subtracted_matrix[subtracted_matrix>=rLimit]=255
    subtracted_matrix[subtracted_matrix<=lLimit]=0
    return subtracted_matrix

def read_the_images(files, frames_url):
    images=[]
    files_length=len(files)    
    for i in range(files_length):
        # print('processing image {} of {}'.format(i+1, files_length))  

        images.append(cv2.imread(frames_url+'/'+files[i], 0))
    return images










if __name__ == '__main__':
    if len(sys.argv)< 2:
        print('USAGE...  SOMETHING')
    
    # make_directories() #Make the required directories for output
    video_url= sys.argv[1]
    if os.path.exists('output'):
        shutil.rmtree('output')
    os.mkdir('output')
    frames_url='output/frames'
    difference_images_url='output/difference_images'
    shots_url='output/shots'
    stacked_matrix_url='output'

    if os.path.exists(frames_url):
        shutil.rmtree(frames_url)
    os.mkdir(frames_url)
    if os.path.exists(difference_images_url):
        shutil.rmtree(difference_images_url)
    os.mkdir(difference_images_url)
    if os.path.exists(shots_url):
        shutil.rmtree(shots_url)
    os.mkdir(shots_url)

   


    detect_shot_change(video_url, frames_url, difference_images_url, shots_url,stacked_matrix_url)
    # detect_shot_change_util(video_url, frames_url, difference_images_url, shots_url,stacked_matrix_url)
    