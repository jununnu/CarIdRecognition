package com.example.cardetection.utils;

import org.opencv.core.*;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.net.URL;
import java.util.ArrayList;
import java.util.List;

import static org.opencv.highgui.HighGui.imshow;
import static org.opencv.highgui.HighGui.waitKey;
import static org.opencv.imgcodecs.Imgcodecs.imread;
import static org.opencv.imgcodecs.Imgcodecs.imwrite;
import static org.opencv.imgproc.Imgproc.*;

public class CatDetectionUtil {
    public static void main(String[] args) throws Exception {
//        // 解决awt报错问题
        System.setProperty("java.awt.headless", "false");
        System.out.println(System.getProperty("java.library.path"));
        // 加载动态库
        URL url = ClassLoader.getSystemResource("lib/opencv/opencv_java460.dll");
        System.load(url.getPath());
        // 读取图像
        Mat image = imread("src/main/resources/static/car.png");
        // 设置图片大小
        Mat src = image.clone();
        resize(image, src, new Size(620, 480));
        // 将图片转为灰度图
        Mat gray = new Mat(src.rows(),src.cols(), CvType.CV_8SC1);
        cvtColor(src, gray, COLOR_RGB2GRAY);
        bilateralFilter(gray, src, 13, 15, 15);
        imshow("img", src);
//        if (image.empty()) {
//            throw new Exception("image is empty");
//        }
//        imshow("Original Image", src);
//
//        // 创建输出单通道图像
//        Mat grayImage = new Mat(image.rows(), image.cols(), CvType.CV_8SC1);
//        // 进行图像色彩空间转换
//        cvtColor(src, grayImage, COLOR_RGB2GRAY);
//
//        imshow("Processed Image", grayImage);
        waitKey();

    }
}
