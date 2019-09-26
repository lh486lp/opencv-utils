package com.ruhr.demo;

import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

/**
 * @author XPY
 * @Description 背景去除 简单案列，只适合背景单一的图像
 * @date 2016年8月30日下午4:14:32
 */
public class demo1 {

    public static void main(String[] args) {
        System.loadLibrary("opencv_java411");
        // 读图像
        Mat img = Imgcodecs.imread("/pencv_img/source/1.jpg");
        Mat newImg = doBackgroundRemoval(img);
        // 写图像
        Imgcodecs.imwrite("/pencv_img/source/1.jpg", newImg);
    }

    private static Mat doBackgroundRemoval(Mat frame) {
        // init
        Mat hsvImg = new Mat();
        List<Mat> hsvPlanes = new ArrayList<>();
        Mat thresholdImg = new Mat();

        int threshType = Imgproc.THRESH_BINARY_INV;

        // threshold the image with the average hue value
        hsvImg.create(frame.size(), CvType.CV_8U);
        Imgproc.cvtColor(frame, hsvImg, Imgproc.COLOR_BGR2HSV);
        Core.split(hsvImg, hsvPlanes);

        // get the average hue value of the image

        Scalar average = Core.mean(hsvPlanes.get(0));
        double threshValue = average.val[0];
        Imgproc.threshold(hsvPlanes.get(0), thresholdImg, threshValue, 179.0, threshType);

        Imgproc.blur(thresholdImg, thresholdImg, new Size(5, 5));

        // dilate to fill gaps, erode to smooth edges
        Imgproc.dilate(thresholdImg, thresholdImg, new Mat(), new Point(-1, -1), 1);
        Imgproc.erode(thresholdImg, thresholdImg, new Mat(), new Point(-1, -1), 3);

        Imgproc.threshold(thresholdImg, thresholdImg, threshValue, 179.0, Imgproc.THRESH_BINARY);

        // create the new image
        Mat foreground = new Mat(frame.size(), CvType.CV_8UC3, new Scalar(255, 255, 255));
        thresholdImg.convertTo(thresholdImg, CvType.CV_8U);
        // 掩膜图像复制
        frame.copyTo(foreground, thresholdImg);
        return foreground;
    }
}

