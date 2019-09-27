package com.ruhr.match;

import lombok.extern.slf4j.Slf4j;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.Scalar;
import org.opencv.features2d.FastFeatureDetector;
import org.opencv.features2d.Features2d;
import org.opencv.features2d.ORB;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;

/**
 * @author luhui
 * @since 2019-09-27 15:52
 */
@Slf4j
public class ORBMatch {

    public void matchImage(Mat originalImage, Mat templateImage) {
        // Initiate ORB detector
        ORB temOrb = ORB.create();

        // find the keypoints with ORB
        MatOfKeyPoint temKp = new MatOfKeyPoint();
        temOrb.detect(templateImage, temKp);

        // compute the descriptors with ORB
        Mat temDes = new Mat();
        temOrb.compute(templateImage, temKp, temDes);

        // draw only keypoints location,not size and orientation
        Mat outImage = new Mat();
        Features2d.drawKeypoints(templateImage, temKp, outImage, new Scalar(255, 0, 0), 0);

        Imgcodecs.imwrite("src/main/resources/orb/result.png", outImage);

        HighGui.imshow("result", outImage);
        HighGui.waitKey();
        System.exit(0);
    }

    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        String templateFilePath = "src/main/resources/pieces.jpeg";
        String originalFilePath = "src/main/resources/go.jpeg";
        //读取图片文件
        Mat templateImage = Imgcodecs.imread(templateFilePath, Imgcodecs.IMREAD_COLOR);
        Mat originalImage = Imgcodecs.imread(originalFilePath, Imgcodecs.IMREAD_COLOR);

        ORBMatch match = new ORBMatch();
        match.matchImage(templateImage, originalImage);
    }
}
