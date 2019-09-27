package com.ruhr.match;

import lombok.extern.slf4j.Slf4j;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.Scalar;
import org.opencv.features2d.FastFeatureDetector;
import org.opencv.features2d.Features2d;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;

/**
 * @author luhui
 * @since 2019-09-27 15:52
 */
@Slf4j
public class FASTMatch {

    public void matchImage(Mat templateImage, Mat originalImage) {
        // Initiate FAST object with default values
        FastFeatureDetector temFast = FastFeatureDetector.create();

        // find and draw the keypoints
        MatOfKeyPoint temKp = new MatOfKeyPoint();
        Mat outImage1 = new Mat();
        temFast.detect(templateImage, temKp);
        Features2d.drawKeypoints(templateImage, temKp, outImage1, new Scalar(255, 0, 0), 0);

        // Print all default params
        log.info("Threshold: {}", temFast.getThreshold());
        log.info("nonmaxSuppression:{}", temFast.getNonmaxSuppression());
        log.info("neighborhood: {}", temFast.getType());
        log.info("Total Keypoints with nonmaxSuppression: {}", temKp.size());

        Imgcodecs.imwrite("src/main/resources/fast/fast_true.png", outImage1);

        // Disable nonmaxSuppression
        temFast.setNonmaxSuppression(false);
        temFast.detect(templateImage, temKp);
        log.info("Total Keypoints without nonmaxSuppression: {}", temKp.size());

        Mat outImage2 = new Mat();
        Features2d.drawKeypoints(templateImage, temKp, outImage2, new Scalar(255, 0, 0), 0);
        Imgcodecs.imwrite("src/main/resources/fast/fast_false.png", outImage2);

        HighGui.imshow("result", outImage2);
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

        FASTMatch match = new FASTMatch();
        match.matchImage(templateImage, originalImage);
    }
}
