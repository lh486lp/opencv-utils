package com.ruhr.demo;

import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

/**
 * @author XPY
 * @Description 人脸检测技术 （靠边缘的和侧脸检测不准确）
 * @date 2016年9月1日下午4:47:33
 */
public class demo3 {

    public static void main(String[] args) {
        System.out.println("Hello, OpenCV");
        // Load the native library.
        System.loadLibrary("opencv_java246");
        new demo3().run();
    }

    public void run() {
        System.out.println("\nRunning DetectFaceDemo");
        System.out.println(getClass().getResource("/haarcascade_frontalface_alt2.xml").getPath());
        // Create a face detector from the cascade file in the resources
        // directory.
        //CascadeClassifier faceDetector = new CascadeClassifier(getClass().getResource("haarcascade_frontalface_alt2.xml").getPath());
        //Mat image = Imgcodecs.imread(getClass().getResource("lena.png").getPath());
        //注意：源程序的路径会多打印一个‘/’，因此总是出现如下错误
        /*
         * Detected 0 faces Writing faceDetection.png libpng warning: Image
         * width is zero in IHDR libpng warning: Image height is zero in IHDR
         * libpng error: Invalid IHDR data
         */
        //因此，我们将第一个字符去掉
        String xmlfilePath = getClass().getResource("/haarcascade_frontalface_alt2.xml").getPath().substring(1);
        CascadeClassifier faceDetector = new CascadeClassifier(xmlfilePath);
        Mat image = Imgcodecs.imread("E:\\face2.jpg");
        // Detect faces in the image.
        // MatOfRect is a special container class for Rect.
        MatOfRect faceDetections = new MatOfRect();
        faceDetector.detectMultiScale(image, faceDetections);

        System.out.println(String.format("Detected %s faces", faceDetections.toArray().length));

        // Draw a bounding box around each face.
        for (Rect rect : faceDetections.toArray()) {
            Imgproc.rectangle(image, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height), new Scalar(0, 255, 0));
        }

        // Save the visualized detection.
        String filename = "E:\\faceDetection.png";
        System.out.println(String.format("Writing %s", filename));
        System.out.println(filename);
        Imgcodecs.imwrite(filename, image);
    }

}

