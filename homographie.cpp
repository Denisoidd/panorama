#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>

using namespace std;
using namespace cv;

int main()
{
	Mat I1 = imread("../IMG_0045.JPG", IMREAD_GRAYSCALE);
	Mat I2 = imread("../IMG_0046.JPG", IMREAD_GRAYSCALE);

	vector<KeyPoint> m1, m2;
	Mat desc1, desc2;

	Ptr<AKAZE> D = AKAZE::create();
	D -> detectAndCompute(I1, noArray(), m1, desc1);
	D -> detectAndCompute(I2, noArray(), m2, desc2);

	//Mat J;
	//drawKeypoints(

	BFMatcher M(NORM_HAMMING);
	vector<DMatch> nn_matches;
	M.match(desc1, desc2, nn_matches);

	Mat L;
	drawMatches(I1, m1, I2, m2, nn_matches, L);
	//imshow("Matches brute", L);

	double max_dist = 0;
	double min_dist = 100;

	for (int i = 0; i < desc1.rows; i++){
		double dist = nn_matches[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}

	vector< DMatch > good_matches;

	for (int i=0; i < desc1.rows; i++ )
	{
		if (nn_matches[i].distance < 5 * min_dist)
		{
			good_matches.push_back(nn_matches[i]);
		}
	}

	Mat Lgood;
	drawMatches(I1, m1, I2, m2, good_matches, Lgood);
	imshow("Good matches", Lgood);

	vector<Point2f> obj;
	vector<Point2f> scene;

	for (size_t i=0; i < good_matches.size(); i++){
		obj.push_back(m1[good_matches[i].queryIdx].pt);
		scene.push_back(m2[good_matches[i].trainIdx].pt);
	}

	Mat H = findHomography(obj, scene, RANSAC);

	Mat K(I1.rows, 2 * I1.cols, CV_8U);

	//Mat I1_final;
	Mat I2_final;

	imshow("Image 1 before", I1);
	imshow("Image 2 before", I2);

	//warpPerspective(I1,I1_final, H, I1.size());
	warpPerspective(I2,I2_final, H, I2.size());
	//warpPerspective(I1,I1_final, H, I1.size());

	hconcat(I1,I2_final,K);
	//Mat Res(I1_final.rows, I1_final.cols + I2_final.cols, CV_8U);
	//hconcat(I1_final, I2_final,Res);
	//imshow("Resultat de warping", Res);
	//imshow("I1 warped", I1_final);
	//imshow("I2 warped", I2_final);

	//imshow("Image 1 after", I1_final);
	//imshow("Image 2 after", I2_final);
	imshow("Panorame", K);

	waitKey(0);
	return 0;
}
