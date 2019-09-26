package com.ruhr.controller;

import jdk.nashorn.internal.objects.annotations.Getter;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.*;
import org.opencv.features2d.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.xfeatures2d.SURF;

import java.util.LinkedList;
import java.util.List;
import java.util.logging.Logger;

/**
 * Created by niwei on 2017/4/28.
 */
public class ImageRecognition {
    private static final Logger logger = Logger.getGlobal();
    private static final int MIN_COUNT = 4;

    /**
     * 这里设置既定值为0.7，该值可自行调整
     */
    private float nndrRatio = 0.7f;

    private int matchesPointCount = 0;

    public float getNndrRatio() {
        return nndrRatio;
    }

    public void setNndrRatio(float nndrRatio) {
        this.nndrRatio = nndrRatio;
    }

    public int getMatchesPointCount() {
        return matchesPointCount;
    }

    public void setMatchesPointCount(int matchesPointCount) {
        this.matchesPointCount = matchesPointCount;
    }

    public void matchImage(Mat templateImage, Mat originalImage) {
        MatOfKeyPoint templateKeyPoints = new MatOfKeyPoint();
        //指定特征点算法SURF
        SURF featureDetector = SURF.create();
        //获取模板图的特征点
        featureDetector.detect(templateImage, templateKeyPoints);
        //提取模板图的特征点
        MatOfKeyPoint templateDescriptors = new MatOfKeyPoint();
        SURF descriptorExtractor = SURF.create();
        logger.info("提取模板图的特征点");
        descriptorExtractor.compute(templateImage, templateKeyPoints, templateDescriptors);

        //显示模板图的特征点图片
        Mat outputImage = new Mat(templateImage.rows(), templateImage.cols(), Imgcodecs.IMREAD_COLOR);
        logger.info("在图片上显示提取的特征点");
        Features2d.drawKeypoints(templateImage, templateKeyPoints, outputImage, new Scalar(255, 0, 0), 0);

        //获取原图的特征点
        MatOfKeyPoint originalKeyPoints = new MatOfKeyPoint();
        MatOfKeyPoint originalDescriptors = new MatOfKeyPoint();
        featureDetector.detect(originalImage, originalKeyPoints);
        logger.info("提取原图的特征点");
        descriptorExtractor.compute(originalImage, originalKeyPoints, originalDescriptors);

        List<MatOfDMatch> matches = new LinkedList();
        DescriptorMatcher descriptorMatcher = DescriptorMatcher.create(DescriptorMatcher.FLANNBASED);
        logger.info("寻找最佳匹配");
        /*
         * knnMatch方法的作用就是在给定特征描述集合中寻找最佳匹配
         * 使用KNN-matching算法，令K=2，则每个match得到两个最接近的descriptor，然后计算最接近距离和次接近距离之间的比值，当比值大于既定值时，才作为最终match。
         */
        descriptorMatcher.knnMatch(templateDescriptors, originalDescriptors, matches, 2);

        logger.info("计算匹配结果");
        LinkedList<DMatch> goodMatchesList = new LinkedList();

        //对匹配结果进行筛选，依据distance进行筛选
        matches.forEach(match -> {
            DMatch[] dmatcharray = match.toArray();
            DMatch m1 = dmatcharray[0];
            DMatch m2 = dmatcharray[1];

            if (m1.distance <= m2.distance * nndrRatio) {
                goodMatchesList.addLast(m1);
            }
        });

        matchesPointCount = goodMatchesList.size();
        //当匹配后的特征点大于等于 4 个，则认为模板图在原图中，该值可以自行调整
        if (matchesPointCount >= MIN_COUNT) {
            logger.info("模板图在原图匹配成功！");

            List<KeyPoint> templateKeyPointList = templateKeyPoints.toList();
            List<KeyPoint> originalKeyPointList = originalKeyPoints.toList();
            LinkedList<Point> objectPoints = new LinkedList();
            LinkedList<Point> scenePoints = new LinkedList();
            goodMatchesList.forEach(goodMatch -> {
                objectPoints.addLast(templateKeyPointList.get(goodMatch.queryIdx).pt);
                scenePoints.addLast(originalKeyPointList.get(goodMatch.trainIdx).pt);
            });
            MatOfPoint2f objMatOfPoint2f = new MatOfPoint2f();
            objMatOfPoint2f.fromList(objectPoints);
            MatOfPoint2f scnMatOfPoint2f = new MatOfPoint2f();
            scnMatOfPoint2f.fromList(scenePoints);
            //使用 findHomography 寻找匹配上的关键点的变换
            Mat homography = Calib3d.findHomography(objMatOfPoint2f, scnMatOfPoint2f, Calib3d.RANSAC, 3);

            // 透视变换(Perspective Transformation)是将图片投影到一个新的视平面(Viewing Plane)，也称作投影映射(Projective Mapping)。
            Mat templateCorners = new Mat(4, 1, CvType.CV_32FC2);
            Mat templateTransformResult = new Mat(4, 1, CvType.CV_32FC2);
            templateCorners.put(0, 0, 0, 0);
            templateCorners.put(1, 0, templateImage.cols(), 0);
            templateCorners.put(2, 0, templateImage.cols(), templateImage.rows());
            templateCorners.put(3, 0, 0, templateImage.rows());
            //使用 perspectiveTransform 将模板图进行透视变以矫正图象得到标准图片
            Core.perspectiveTransform(templateCorners, templateTransformResult, homography);

            //矩形四个顶点
            double[] pointa = templateTransformResult.get(0, 0);
            double[] pointb = templateTransformResult.get(1, 0);
            double[] pointc = templateTransformResult.get(2, 0);
            double[] pointd = templateTransformResult.get(3, 0);

            //指定取得数组子集的范围
            int rowStart = (int) pointa[1];
            int rowEnd = (int) pointc[1];
            int colStart = (int) pointd[0];
            int colEnd = (int) pointb[0];
            Mat subMat = originalImage.submat(rowStart, rowEnd, colStart, colEnd);
            Imgcodecs.imwrite("target/原图中的匹配图.jpg", subMat);

            //将匹配的图像用用四条线框出来
            // 上 A->B
            Imgproc.line(originalImage, new Point(pointa), new Point(pointb), new Scalar(0, 255, 0), 4);
            // 右 B->C
            Imgproc.line(originalImage, new Point(pointb), new Point(pointc), new Scalar(0, 255, 0), 4);
            // 下 C->D
            Imgproc.line(originalImage, new Point(pointc), new Point(pointd), new Scalar(0, 255, 0), 4);
            // 左 D->A
            Imgproc.line(originalImage, new Point(pointd), new Point(pointa), new Scalar(0, 255, 0), 4);

            MatOfDMatch goodMatches = new MatOfDMatch();
            goodMatches.fromList(goodMatchesList);
            Mat matchOutput = new Mat(originalImage.rows() * 2, originalImage.cols() * 2, Imgcodecs.IMREAD_COLOR);
            Features2d.drawMatches(templateImage, templateKeyPoints, originalImage, originalKeyPoints, goodMatches, matchOutput, new Scalar(0, 255, 0), new Scalar(255, 0, 0), new MatOfByte(), 2);

            Imgcodecs.imwrite("target/特征点匹配过程.jpg", matchOutput);
            Imgcodecs.imwrite("target/模板图在原图中的位置.jpg", originalImage);
        } else {
            logger.info("模板图不在原图中！");
        }

        Imgcodecs.imwrite("target/模板特征点.jpg", outputImage);
    }

    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        String templateFilePath = "target/pieces.jpeg";
        String originalFilePath = "target/go.jpeg";
        //读取图片文件
        Mat templateImage = Imgcodecs.imread(templateFilePath, Imgcodecs.IMREAD_COLOR);
        Mat originalImage = Imgcodecs.imread(originalFilePath, Imgcodecs.IMREAD_COLOR);

        ImageRecognition imageRecognition = new ImageRecognition();
        imageRecognition.matchImage(templateImage, originalImage);

        logger.info("匹配的像素点总数：" + imageRecognition.getMatchesPointCount());
    }
}