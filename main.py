import cv2 as cv
import numpy as np
import random
import math
from gensim.models import Word2Vec
import easygui as eg


def flatten(l):
	def f(l, r=[]):
		if not isinstance(l, list):
			r.append(l)
		else:
			for i in l:
				f(i, r)
		return r

	return f(l)


def ispointsvalid(edgesize, *points):
	points_ = flatten(list(points))
	for i in points_:
		if i < 0 or i > edgesize:
			return False
	return True


def isareavalid(img, *points_):
	height = img.shape[0]
	width = img.shape[1]

	mask = np.zeros((height, width), dtype=np.uint8)
	points = np.array([list(points_)])
	cv.fillPoly(mask, points, (255))
	res = cv.bitwise_and(img, img, mask=mask)

	rect = cv.boundingRect(points)  # returns (x,y,w,h) of the rect
	cropped = res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]

	# cv.imshow("cropped", cropped)
	# cv.imshow("same size", mask)

	s = np.sum(cropped)
	# print('s:', s)
	# cv.waitKey()
	if s == 0:
		return True
	return False


# with open("words.txt", 'r') as f:
# 	words = set(f.readlines()[:])

model = Word2Vec.load('word2vec.model')
keyword = eg.enterbox("Key Word: ", title="Keyword input")

words = [i[0] for i in model.wv.most_similar(keyword, topn=100)]
words = words[::-1]
print(words)
h, w, c = 2000, 2000, 3
img = np.zeros(shape=[h, w, c], dtype=np.uint8)

angles = [90, -90, 0, 0, 0, 0]
while len(words) != 0:

	word_ = words.pop()
	word_ = word_.split('\n')[0].upper()
	word = ''
	for i in word_:
		if i.isalpha():
			word += i

	# angle = math.radians(random.randint(-90, 90))
	angle = math.radians(angles[random.randint(0, 5)])
	font_size = random.randint(8, 90) / 10

	r_x = random.randint(400, h - 400)
	r_y = random.randint(400, w - 400)

	LETTER_WIDTH = font_size * 23
	LETTER_HEIGHT = font_size * 37
	WORD_WIDTH = (len(word) + 1) * LETTER_WIDTH

	bottom_left_p = [int(r_x - (font_size * 10)), int(r_y + (font_size * 10))]
	top_left_p = [int(bottom_left_p[0] - LETTER_HEIGHT * math.sin(angle)),
		      int(bottom_left_p[1] - LETTER_HEIGHT * math.cos(angle))]
	top_right_p = [int(top_left_p[0] + WORD_WIDTH * math.cos(angle)),
		       int(top_left_p[1] - WORD_WIDTH * math.sin(angle))]
	bottom_right_p = [int(bottom_left_p[0] + WORD_WIDTH * math.cos(angle)),
			  int(bottom_left_p[1] - WORD_WIDTH * math.sin(angle))]

	text_img = np.zeros(shape=[h, w, c], dtype=np.uint8)

	if ispointsvalid(h, top_right_p, top_left_p, bottom_left_p, bottom_right_p) and isareavalid(img, top_right_p,
												    top_left_p,
												    bottom_left_p,
												    bottom_right_p
												    ):

		# img = cv.circle(img, tuple(top_left_p), 1, (255, 0, 0), 3)  # blue
		# img = cv.circle(img, tuple(top_right_p), 1, (0, 255, 0), 3)  # green
		# img = cv.circle(img, tuple(bottom_left_p), 1, (0, 0, 255), 3)  # red
		# img = cv.circle(img, tuple(bottom_right_p), 1, (0, 255, 255), 3)  # yellow

		text_img = cv.putText(text_img, word, (r_x, r_y), cv.FONT_HERSHEY_SIMPLEX,
				      font_size,
				      (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
				      max(1, math.ceil(font_size * 2)), cv.LINE_AA)

		M = cv.getRotationMatrix2D(bottom_left_p, math.degrees(angle), 1)
		text_img = cv.warpAffine(text_img, M, (h, w))

		img = np.add(img, text_img)
		print(len(words), word)
		cv.imshow("img", img)
		cv.waitKey(1)

	else:
		# words.add(word)
		words.append(word)
cv.imshow("img", img)

cv.waitKey()
outputfilename = eg.enterbox(msg="Output file name: ", title="Output File Name")
cv.imwrite(outputfilename + '.png', img)