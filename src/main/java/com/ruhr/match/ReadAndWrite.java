package com.ruhr.match;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.File;

/**
 * @author luhui
 * @since 2019-09-26 14:51
 */
public class ReadAndWrite {

    public static void main(String[] args) {
        // 这个必须要写,不写报java.lang.UnsatisfiedLinkError
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        File imgFile = new File("target/code.png");
        if (!imgFile.exists()) {
            System.out.println("未找到图片");
            return;
        }
        Mat src = Imgcodecs.imread(imgFile.toString(), Imgcodecs.IMREAD_GRAYSCALE);

        Mat dst = new Mat();
        Imgproc.adaptiveThreshold(src, dst, 255, Imgproc.ADAPTIVE_THRESH_MEAN_C, Imgproc.THRESH_BINARY, 13, 5);
        Imgcodecs.imwrite("target/AdaptiveThreshold" + imgFile.getName(), dst);
    }
}
