# Image Recognition: Andrea Abellera and Andrew Marinic
 ## Introduction
<p>	The goal of our project is to create an algorithm for  image recognising. We will be testing a series of images containing balls and other containing round objects that are not balls. These known objects can be used to help us refine our search. </p>
	 <p>	The algorithm  will use the uncanny edge detection to reduce our test image to its edges. The number of trials can be modified to help refine the search. A single trial would search the image as a whole, and then each subsequent trial would scan the trial image to help identify smaller balls, multiple balls, and off centre balls. The algorithm was calculates the eigen matrix of the image library and compares the difference between the matrix and the test image or slice.  We then perform a similar test comparing the difference between our individual library images and the test image to see if we have a good match to any of our library images. </p> 
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTMzOTkyMTE4M119
-->