#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>

using namespace std;
using namespace cv;

int main()
{
	//Uploadiong images
	Mat I1 = imread("../IMG_0045.JPG", IMREAD_GRAYSCALE);
	Mat I2 = imread("../IMG_0046.JPG", IMREAD_GRAYSCALE);

	vector<KeyPoint> m1, m2, l1, l2;
	Mat desc1, desc2;

	//Description and key points computation
	Ptr<AKAZE> D = AKAZE::create();
	D -> detectAndCompute(I1, noArray(), m1, desc1);
	D -> detectAndCompute(I2, noArray(), m2, desc2);

	//Looking for 2 best matches to compare them after
	BFMatcher M(NORM_HAMMING);
	vector<vector <DMatch>> Knn_matches;
	M.knnMatch(desc2, desc1, Knn_matches, 2);

	//Using these two types of matches to find the best one
	vector< DMatch > good_matches;
	for (size_t j = 0; j < Knn_matches.size(); j++){
		DMatch closest = Knn_matches[j][0];
		//Getting two distances to compare them after
		double d1 = Knn_matches[j][0].distance;
		double d2 = Knn_matches[j][1].distance;
		//Took 0.7 for a matching ratio
		if (d1 < d2 * 0.7){
			int i = static_cast<int>(l1.size());
			l1.push_back(m1[closest.trainIdx]);
			l2.push_back(m2[closest.queryIdx]);
			good_matches.push_back(DMatch(i,i,0));
		}
	}

	//Variable just to save the information
	Mat L;
	drawMatches(I1, l1, I2, l2, good_matches, L);
	imshow("Good matches", L);

	//To calculate homography
	vector<Point2f> obj;
	vector<Point2f> scene;

	for (size_t i=0; i < good_matches.size(); i++){
		obj.push_back(l1[good_matches[i].queryIdx].pt);
		scene.push_back(l2[good_matches[i].trainIdx].pt);
	}

	Mat H = findHomography(scene, obj, RANSAC);

	//Apply the computed homography matrix to warp the second image
  Mat Panorama(I1.rows, 2 * I1.cols,  CV_8U);
  warpPerspective(I2, Panorama, H, Panorama.size());
	imshow("Image before the end", Panorama);

  //Adding the first image on the changed by homography
  for(int i = 0; i < I1.rows; i++){
      for(int j = 0; j < I1.cols; j++)
          Panorama.at<uchar>(i,j) = I1.at<uchar>(i,j);
  }
  imshow("Panorama", Panorama);
  imwrite("Panorama.jpg", Panorama);

  waitKey(0);
  return 0;
}
