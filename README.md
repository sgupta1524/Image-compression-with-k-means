# Image-compression-with-k-means

Both mykmeans and mykmedoids take input and output format as follows: <br />
Input <br />
•pixels: the input image representation. Each row contains one data point (pixel). <br />
For image dataset, it contains 3 columns, each column corresponding to Red, Green, and Blue component.  Each componenthas an integer value between 0 and 255. <br />

•K: the number of desired clusters.  Too high value ofKmay result in empty cluster error. <br />

Output <br />
•class:  cluster assignment of each data point in pixels.  The assignment should be 1,  2,  3,  etc.  ForK= 5, for example, each cell of class should be either 1, 2, 3, 4, or 5.  The output should be a columnvector withsize(pixels, 1)elements. <br />
•centroid:  location  ofKcentroids  (or  representatives)  in  your  result.   With  images,  each  centroidcorresponds to the representative color of each cluster.  The output should be a matrix withKrowsand 3 columns.  The range of values should be [0, 255], possibly floating point numbers

