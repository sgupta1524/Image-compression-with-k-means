import numpy as np
import imageio
from matplotlib import pyplot as plt
import sys
import os
import time

#start_time = time.time()
def mykmeans(pixels, K):
	np.random.seed(0)
	x = pixels.reshape(pixels.shape[0] * pixels.shape[1], 3)
	#x = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]]
	cno = K
	x = np.transpose(x)
#	print(x.shape)
	m = np.shape(x)[1]

	random_indices = np.random.choice(m, size=K, replace=False)
	c = np.array(x[:, random_indices])
	#print(c)
	c0 = c * 0

	while np.linalg.norm(c - c0, ord='fro') > 1e-6:
		c0 = c

		# c = [[2, 3 ,4], [1,2,3]]
		cnorm = np.power((c.tolist()), 2)
		cnorm = np.sum(cnorm, axis=0)
		#print(c)
		#print("*******************************")
		#print(cnorm)
		tmpdiff = 2 * np.dot(x.T, c) - cnorm
		#print(c)
		#print(x.T)
		#print(2 * np.dot(x.T, c))
		#print(tmpdiff)
		labels = np.argmax(tmpdiff, axis=1)
		#print(labels)

		#P = csc_matrix((np.ones(m), (np.arange(0, m, 1), labels)), shape=(m, cno))
		P = np.zeros((m, cno), dtype=int)

		for i,j in zip(labels,range(len(labels))):
			#print(i,j)
			P[j][i] = 1
			#print(j,i, P[j][i])

		#print(P)

		sorted_labels = sorted(labels)
		counts_dict = {}
		for i in labels:
			counts_dict[i] = counts_dict.get(i, 0)+1

		counts = [1]*cno
		for i in list(set(sorted_labels)):
			counts[i] = (counts_dict[i])

		#print(counts)
		#print((P.T.dot(x.T)).T)
		c = np.array(np.divide((P.T.dot(x.T)).T, counts))
		#print("###########################")
		#print(c)
	return labels, c.T
	"""
	Your goal of this assignment is implementing your own K-means.

	Input:
		pixels: data set. Each row contains one data point. For image
		dataset, it contains 3 columns, each column corresponding to Red,
		Green, and Blue component.

		K: the number of desired clusters. Too high value of K may result in
		empty cluster error. Then, you need to reduce it.

	Output:
		class: the class assignment of each data point in pixels. The
		assignment should be 1, 2, 3, etc. For K = 5, for example, each cell
		of class should be either 1, 2, 3, 4, or 5. The output should be a
		column vector with size(pixels, 1) elements.

		centroid: the location of K centroids in your result. With images,
		each centroid corresponds to the representative color of each
		cluster. The output should be a matrix with size(pixels, 1) rows and
		3 columns. The range of values should be [0, 255].
	"""


# raise NotImplementedError

def mykmedoids(pixels, K):
	"""
	Your goal of this assignment is implementing your own K-medoids.
	Please refer to the instructions carefully, and we encourage you to
	consult with other resources about this algorithm on the web.
​
	Input:
		pixels: data set. Each row contains one data point. For image
		dataset, it contains 3 columns, each column corresponding to Red,
		Green, and Blue component.
​
		K: the number of desired clusters. Too high value of K may result in
		empty cluster error. Then, you need to reduce it.
​
	Output:
		class: the class assignment of each data point in pixels. The
		assignment should be 1, 2, 3, etc. For K = 5, for example, each cell
		of class should be either 1, 2, 3, 4, or 5. The output should be a
		column vector with size(pixels, 1) elements.
​
		centroid: the location of K centroids in your result. With images,
		each centroid corresponds to the representative color of each
		cluster. The output should be a matrix with size(pixels, 1) rows and
		3 columns. The range of values should be [0, 255].
	"""

	np.random.seed(0)
	x = pixels.reshape(pixels.shape[0] * pixels.shape[1], 3)
	# x = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]]
	cno = K
	print(x.shape)
	number_of_pixels = np.shape(x)[0]
	classes = np.zeros((number_of_pixels, 1), dtype=float)
	random_indices = np.random.choice(number_of_pixels, size=K, replace=False)
	centroid = x[random_indices, :]
	no_iterations = 10
	Jerror = np.zeros((no_iterations, 1), dtype=float)

	def assign_class(data, centers):
		n = np.shape(data)[0]
		k = np.shape(centers)[0]

		classes = np.zeros((n, 1), dtype=float)
		for j in range(0, n):
			dist = np.zeros((1, k), dtype=float)
			for l in range(0, k):
				# print(data[l, :])
				temp = np.subtract(centers[l, :], data[j, :])
				# print("first*************")
				# print(temp.shape)
				temp = np.power(temp.tolist(), 2)
				temp = np.sum(temp, axis=0)
				# print("second*************")
				# print(temp)
				temp = np.sqrt(temp)
				# print(temp)
				# print(dist)
				dist[0, l] = temp
			t = np.random.choice([0, 1], size=1, replace=False)
			classes[j, 0] = np.where(dist == np.amin(dist))[t[0]][0]
		# print(classes.shape)
		return classes

	def calc_rerror(data, centers, classes):
		J = 0
		n = np.shape(data)[0]
		k = np.shape(centers)[0]
		for jj in range(0, n):
			for ll in range(0, k):
				# print(classes)
				if classes[jj, 0] == ll:
					J = J + np.sqrt(np.sum(np.power(np.subtract(centers[ll, :], data[jj, :]).tolist(), 2), axis=0))
					break
		return J

	def update_centres(data, classes, K):
		print(data.shape)
		(n, f) = data.shape
		centroid = np.zeros((K, f), dtype=float)
		no_cluster = np.zeros((K, 1), dtype=float)

		for j in range(0, n):
			cluster_index = classes[j, 0]
			centroid[int(cluster_index), :] = centroid[int(cluster_index), :] + data[j, :]
			no_cluster[int(cluster_index)] = no_cluster[int(cluster_index)] + 1
			for j in range(0, K):
				if no_cluster[j] != 0:
					centroid[j, :] = (centroid[j, :] / no_cluster[j])

		return centroid

	for i in range(1, no_iterations):
		classes = assign_class(x, centroid)
		Jerror[i, 0] = calc_rerror(x, centroid, classes)
		if (i != 1 and Jerror[i, 0] > Jerror[i - 1, 0]):
			break
		else:
			centroid = update_centres(x, classes, K)
			print(centroid.shape)
			for j in range(0, K):
				distance = np.zeros((number_of_pixels, 1), dtype=float)
				for l in range(0, number_of_pixels):
					if (classes[l, 0] == j):
						distance[l, 0] = np.sqrt(
							np.sum(np.power(np.subtract(centroid[j, :], x[l, :]).tolist(), 2), axis=0))
					else:
						distance[l, 0] = float('inf')
				t = np.random.choice([0, 1], size=1, replace=False)
				# print(np.where(distance == np.amin(distance))[t[0]][0])
				centroid[j, :] = x[np.where(distance == np.amin(distance))[t[0]][0], :]

	print("************end*******************")
	print(classes)
	print(centroid)

	centroid2 = np.zeros((number_of_pixels, 3), dtype=int)
	classes2 = np.zeros((number_of_pixels, 1), dtype=int)
	for i in range(number_of_pixels):
		centroid2[i, :] = centroid[int(classes[i][0]), :]
		classes2[i][0] = int(classes[i][0]) + 1

	#print(classes2)
	return classes2, centroid2


#print(c)
#	raise NotImplementedError


def main():
	if (len(sys.argv) < 2):
		print("Please supply an image file")
		return

	image_file_name = sys.argv[1]

	K = 5 if len(sys.argv) == 2 else int(sys.argv[2])
	print(image_file_name, K)
	im = np.asarray(imageio.imread(image_file_name))

	fig, axs = plt.subplots(1, 2)

	classes, centers = mykmedoids(im, K)
	print(classes, centers)
	new_im = np.asarray(centers[classes].reshape(im.shape), im.dtype)
	imageio.imwrite(os.path.basename(os.path.splitext(image_file_name)[0]) + '_converted_mykmedoids_' + str(K) + os.path.splitext(image_file_name)[1], new_im)
	axs[0].imshow(new_im)
	axs[0].set_title('K-medoids')

	classes, centers = mykmeans(im, K)
	print(set(classes), centers)
	print(centers[classes])

	new_im = np.asarray(centers[classes].reshape(im.shape), im.dtype)
	imageio.imwrite(os.path.basename(os.path.splitext(image_file_name)[0]) + '_converted_mykmeans_' + str(K) + os.path.splitext(image_file_name)[1], new_im)
	axs[1].imshow(new_im)
	axs[1].set_title('K-means')

	#print(time.time()-start_time)
	plt.show()

if __name__ == '__main__':
	main()
