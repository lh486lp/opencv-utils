package com.ruhr.demo;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

/**
 * @author XPY
 * @Description 边缘检测
 * @date 2016年8月30日下午5:01:01
 */
public class demo2 {

    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        // 读图像
        Mat img = Imgcodecs.imread("E:\\face7.jpg");
        Mat newImg = doCanny(img);
        // 写图像
        Imgcodecs.imwrite("E:\\opencv_img\\target\\2.jpg", newImg);
    }

    private static Mat doCanny(Mat frame) {
        // init
        Mat grayImage = new Mat();
        Mat detectedEdges = new Mat();
        double threshold = 10;
        // convert to grayscale
        Imgproc.cvtColor(frame, grayImage, Imgproc.COLOR_BGR2GRAY);
        // reduce noise with a 3x3 kernel
        Imgproc.blur(grayImage, detectedEdges, new Size(3, 3));
        // canny detector, with ratio of lower:upper threshold of 3:1
        Imgproc.Canny(detectedEdges, detectedEdges, threshold, threshold * 3);
        // using Canny's output as a mask, display the result
        Mat dest = new Mat();
        frame.copyTo(dest, detectedEdges);
        return dest;
    }
}

