# Matthew Kim
# mtt.kim@mail.utoronto.ca

# Libraries used: OpenCV, Numpy, Scipy

import cv2
import numpy as np
import os
import sys
from scipy.cluster.vq import kmeans2, vq
from scipy.spatial.distance import cdist
from collections import Counter

def drawMatches(img1, kp1, img2, kp2, matches, mask):
    """
    Implementation of cv2.drawMatches as OpenCV 2.4.9
    does not have this function available but it's supported in
    OpenCV 3.0.0

    Modified function from:
    http://stackoverflow.com/questions/20259025/module-object-has-no-attribute-drawmatches-opencv-python

    This function takes in two images with their associated 
    keypoints, as well as a list of DMatch data structure (matches) 
    that contains which keypoints matched in which images.

    An image will be produced where a montage is shown with
    the first image followed by the second image beside it.

    Keypoints are delineated with circles, while lines are connected
    between matching keypoints. Inliers will be blue and outliers red.

    img1,img2 - Grayscale images
    kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint 
              detection algorithms
    matches - A list of matches of corresponding keypoints through any
              OpenCV keypoint matching algorithm
    mask    - Mask indicating each match's outlier status. 
    """

    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    # Place the first image to the left
    out[:rows1,:cols1] = np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    out[:rows2,cols1:] = np.dstack([img2, img2, img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them

    index = 0
    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1
        cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)   
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

        # Draw a line in between the two points
        # thickness = 1
        # colour blue

        if(mask[index][0]):
        	cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)
        else:
        	cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (0, 0, 255), 1)

        index = index + 1


    # Show the image
    cv2.imshow('Matched Features', out)
    cv2.waitKey(0)
    cv2.destroyWindow('Matched Features')

    # Also return the image if you'd like a copy
    return out

# Recursively make a descriptor tree. K = branching factor
def make_tree(k):
	"""
	Constructs a tree for efficient retrieval approach by 
	Nister and Stewenius
	"""

	print("Constructing tree with k = "  + str(k) + "...\n")

	descriptors = list()
	descript_dict = dict()
	allDescript_dict = dict()

	sift = cv2.xfeatures2d.SIFT_create()

	print("Getting descriptors of each image...")
	# Get SIFT descriptors for each image

	for i in os.listdir(os.getcwd() + '/DVDcovers'):
		if(i!='.DS_Store'):
			img = cv2.imread('DVDcovers/' + i)
			try:
				gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
			except: 
				print 'Failed to process file: ' + i
			kp, descriptor = sift.detectAndCompute(gray,None)
			# Add them to a list and dictionary
			j = 0
			allDescript_dict[i] = descriptor

			for each in descriptor:
				descript_dict[tuple(list(each))] = i
				descriptors.append(list(each))
				j = j + 1

	descriptors = np.array(descriptors)

	print('Creating kTree...')
	tree = kTree(k,descriptors,0,descript_dict,allDescript_dict)

	return tree

def searchTree(name, tree):

	vote_dict = dict()
	key_dict = dict()
	descript_dict = tree.descript_dict
	allDescript_dict = tree.allDescript_dict

	sift = cv2.xfeatures2d.SIFT_create()

	print("Constructing key dict...")
	# Get SIFT descriptors for each image

	for i in os.listdir(os.getcwd() + '/DVDcovers'):
		if(i!='.DS_Store'):
			img = cv2.imread('DVDcovers/' + i)
			try:
				gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
			except: 
				print 'Failed to process file: ' + i
			kp = sift.detect(gray,None)
			# Add them to a list and dictionary
			key_dict[i] = kp

	try:
		targetImg = cv2.imread('test/' + name)
		targetImg = cv2.cvtColor(targetImg, cv2.COLOR_BGR2GRAY)
	except:
		print 'Failed to open file at: test/' + name

	# Test image keypoints and descriptors
	targetKp, targetDesc = sift.detectAndCompute(targetImg, None)

	print('Initializing voting dictionary...')
	for i in os.listdir(os.getcwd() + '/DVDcovers'):
		if(i != '.DS_Store'):
			vote_dict[i] = 0

	print("Performing search for file " + name + "...")
	for i in range(0,len(targetDesc)):
		descAnswer = tree.search(targetDesc[i])
		for each in descAnswer:
			vote_dict[descript_dict[tuple(list(each))]] = vote_dict[descript_dict[tuple(list(each))]] + 1

	print('Done! The top ten images are:')
	finalAnswer = Counter(vote_dict)
	finalAnswer.most_common()
	mostInliers = 0.0
	actualAnswer = ''
	finalMatches = None
	finalGood = None
	finalMask = None
	for fname, votes in finalAnswer.most_common(10):
		print '%s: %i' % (fname, votes)

		FLANN_INDEX_KDTREE = 0
		index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
		search_params = dict(checks = 50)

		flann = cv2.FlannBasedMatcher(index_params, search_params)

		matches = flann.knnMatch(targetDesc,allDescript_dict[fname],k=2)

		# store all the good matches as per Lowe's ratio test.
		good = []
		for m,n in matches:
		    if m.distance < 0.7*n.distance:
		        good.append(m)

		src_pts = np.float32([ targetKp[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
		dst_pts = np.float32([ key_dict[fname][m.trainIdx].pt for m in good ]).reshape(-1,1,2)
		
		M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

		inliers = 0

		# Get number of inliers
		for each in mask:
			if(each[0]):
				inliers = inliers + 1

		print('inliers: ' + str(inliers) + ' out of ' + str(len(mask)))

		if (inliers > mostInliers):
			mostInliers = inliers
			actualAnswer = fname
			finalMatches = matches
			finalGood = good
			finalMask = mask
	print('I (ktree.py) have found ' + actualAnswer + ' to be the DVDcover in the given image.')
	cover = cv2.imread('DVDcovers/' + actualAnswer)
	cover = cv2.cvtColor(cover, cv2.COLOR_BGR2GRAY)
	drawMatches(targetImg,targetKp,cover,key_dict[actualAnswer],finalGood,finalMask)


class kTree:

	def __init__(self,k,descriptors,level,descript_dict,allDescript_dict):
		self.head = _kTree(k, descriptors, level)
		self.descript_dict = descript_dict
		self.allDescript_dict = allDescript_dict

	def search(self,descriptor):
		return self.head.search(descriptor)

	def descript_dict(self):
		return self.descript_dict

	def allDescript_dict(self):
		return allDescript_dict

class _kTree:

	def __init__(self,k,descriptors,level):
		self.k = k
		self.descriptors = descriptors
		self.data = []
		# kTree children 
		self.children = []
		# Boolean to indicate if this is a leaf
		self.end = False
		if (len(descriptors) >= k):
			self.data, labels = kmeans2(descriptors,k, minit='points')
			# put the descriptors into an array and then make a child!
			for i in range(0,k):
				recDesc = []
				for j in range(0,len(labels)):
					if i == labels[j]:
						recDesc.append(descriptors[j])
				recDesc = np.array(recDesc)
				if(len(recDesc) > 0):
					self.children.append(_kTree(self.k,recDesc,level+1))
		else:
			self.data = descriptors
			self.end = True
	# search for the top 10 matches in our ktree
	def search(self, descriptor):
		if(self.end):
			return self.data
		least = np.linalg.norm(np.array(descriptor)-np.array(self.data[0]))
		which = 0
		for i in range(1,len(self.data)):
			dist = np.linalg.norm(np.array(descriptor) - np.array(self.data[i]))
			if(dist < least):
				least = dist
				which = i
		return self.children[which].search(descriptor)

if __name__ == "__main__":
	# usage: python ./ktrees [k] [name]

	if (not (len(sys.argv) == 3)):
		print("Usage: python ./dvdcovers [name] [k]")
		sys.exit(1)
	if (int(sys.argv[2]) < 2):
		print("k must be greater than 1")
		sys.exit(1)
	tree = make_tree(int(sys.argv[2]))
	searchTree(sys.argv[1], tree)
	sys.exit(0)