#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <random>
#include <math.h>
#include <filesystem>
#include <algorithm>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>

const float INPUT_WIDTH = 640.0;
const float INPUT_HEIGHT = 640.0;
const float SCORE_THRESHOLD = 0.2;
const float NMS_THRESHOLD = 0.5;
const float CONFIDENCE_THRESHOLD = 0.6;

struct Centroid
{
    int id;
    float conf;
    float area;
    cv::Point center;
    cv::Rect box;
    std::vector<cv::Rect> box_history;
    std::vector<cv::Point> position_history;
    cv::Point next_position;

    std::string name;
    bool under_tracking;
    int lane_num;
    float distance{0.0};
    float speed{0.0};
};


class Lane
{
public:
    cv::Point num1[2] = {cv::Point(250,0), cv::Point(45 ,360)};
    cv::Point num2[2] = {cv::Point(267,0), cv::Point(172,360)};
    cv::Point num3[2] = {cv::Point(282,0), cv::Point(330,360)};
    cv::Point num4[2] = {cv::Point(300,0), cv::Point(475,360)};
    cv::Point num5[2] = {cv::Point(313,0), cv::Point(620,360)};
    cv::Point num6[2] = {cv::Point(330,0), cv::Point(640,270)};

    cv::Point speed_line1[2] = {cv::Point(0,105), cv::Point(640, 105)};
    cv::Point speed_line2[2] = {cv::Point(0,320), cv::Point(640, 320)};

    // void drawSpeedLine(cv::Mat const& image)
    // {   
    //     cv::line(image, speed_line1[0], speed_line1[1], cv::Scalar(0.0, 0.0, 255.0), 2);
    //     cv::line(image, speed_line2[0], speed_line2[1], cv::Scalar(0.0, 0.0, 255.0), 2);
    // }
    void draw(cv::Mat const& image)
    {
        cv::line(image, num1[0], num1[1], cv::Scalar(255.0, 255.0, 255.0), 2);
        cv::line(image, num2[0], num2[1], cv::Scalar(255.0, 255.0, 255.0), 2);
        cv::line(image, num3[0], num3[1], cv::Scalar(255.0, 255.0, 255.0), 2);
        cv::line(image, num4[0], num4[1], cv::Scalar(255.0, 255.0, 255.0), 2);
        cv::line(image, num5[0], num5[1], cv::Scalar(255.0, 255.0, 255.0), 2);
        cv::line(image, num6[0], num6[1], cv::Scalar(255.0, 255.0, 255.0), 2);
    }
};


class Timer
{
public:
    std::chrono::time_point<std::chrono::steady_clock> begin, end;
    std::chrono::duration<float> duration;

    void start() {begin = std::chrono::steady_clock::now();}

    float stop()
    {
        end = std::chrono::steady_clock::now();
        duration = end - begin; 
        float second = duration.count();
        return second;
    }

};

double distanceBetweenPoints(cv::Point &point1, cv::Point &point2) 
{
    double distance = sqrt(pow((point2.x - point1.x), 2) + pow((point2.y - point1.y), 2));
    return distance;
};

bool checkTheLine(cv::Point &center, cv::Point left_line[2], cv::Point right_line[2])
{
    bool left  = false;
    bool right = false;

    cv::LineIterator left_it(cv::Size(640,360),left_line[0], left_line[1], 8);
    cv::LineIterator rigth_it(cv::Size(640,360),right_line[0], right_line[1], 8);

    for(int i = 0; i < left_it.count; i++, ++left_it)
    {
        cv::Point pt= left_it.pos();
        if((center.y == pt.y) && (center.x >= pt.x))
            left = true;
    }

    for(int i = 0; i < rigth_it.count; i++, ++rigth_it)
    {
        cv::Point pt= rigth_it.pos();
        if((center.y == pt.y) && (center.x <= pt.x))
            right = true;
    }

    if(left && right)
        return true;
    else
        return false;

}

void findLineNumber(std::vector<Centroid> &vehicles) 
{
    Lane line;

    for (auto &vehicle : vehicles) 
    {

        if(checkTheLine(vehicle.center, line.num1, line.num2))
        {
            vehicle.lane_num = 1;
        }
        else if(checkTheLine(vehicle.center, line.num2, line.num3))
        {
            vehicle.lane_num = 2;
        }
        else if(checkTheLine(vehicle.center, line.num3, line.num4))
        {
            vehicle.lane_num = 3;
        }
        else if(checkTheLine(vehicle.center, line.num4, line.num5))
        {
            vehicle.lane_num = 4;
        }
        else if(checkTheLine(vehicle.center, line.num5, line.num6))
        {
            vehicle.lane_num = 5;
        }
        else
        {
            vehicle.lane_num = 0;
        }
    }
};

void findVehiclesTrajectory(std::vector<Centroid> &existingVehicles, std::vector<Centroid> &currentFrameVehicles) 
{
    for (auto &existingVehicle : existingVehicles) 
    {
        existingVehicle.under_tracking = true;

        if(existingVehicle.position_history.size() == 0)
            existingVehicle.position_history.push_back(existingVehicle.center);
    }
    findLineNumber(currentFrameVehicles);


    for (auto &currentFrameVehicle : currentFrameVehicles) 
    {
        int least_distance_index = 0;
        double least_distance = 100000.0;

        for (int i = 0; i < existingVehicles.size(); i++) 
        {
            if ((existingVehicles[i].under_tracking == true) && (existingVehicles[i].lane_num == currentFrameVehicle.lane_num) )
            {
                double distance = distanceBetweenPoints(currentFrameVehicle.center, existingVehicles[i].position_history.back());
                if ((distance < least_distance) && (existingVehicles[i].area > currentFrameVehicle.area) ) 
                {
                    least_distance = distance;
                    least_distance_index = i;
                }
            }
        }

        if (least_distance < currentFrameVehicle.box.height * 2.5) 
        {
                existingVehicles[least_distance_index].box_history.push_back(currentFrameVehicle.box);
                existingVehicles[least_distance_index].distance = least_distance;
                existingVehicles[least_distance_index].speed = ((least_distance/3)/1) ; //3 is the real distance scale factor and time is 1sec

                existingVehicles[least_distance_index].position_history.push_back(currentFrameVehicle.center);

                existingVehicles[least_distance_index].under_tracking = true;
        }
        else { 
            existingVehicles.push_back(currentFrameVehicle);
        }

    }

};


int main(int argc, char** argv )
{
    std::cout << "OpenCV version : " << CV_VERSION << std::endl;

    Timer time,total_time;
    total_time.start();
    time.start();

    std::vector<std::vector<Centroid>> vector_of_centeroids; // keeping all vehicles from all input images.
    vector_of_centeroids.reserve(5);

    // Load class names:
    std::vector<std::string> class_names;
    class_names.reserve(100);
    std::ifstream ifs(std::string("../models/classes_name.txt").c_str());
    std::string line;
    while (getline(ifs, line)) {class_names.push_back(line);} 
    
    // load the neural network model:
    cv::dnn::Net model = cv::dnn::readNetFromONNX("../models/best.onnx");
    model.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    model.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    // load input image list:
    std::vector<std::string> input_files= {"../media/traffic/image1.jpg", "../media/traffic/image2.jpg"};

    for (std::string const& file : input_files)
    {
        // Load the image:
        cv::Mat image = cv::imread(file, 1);
        
        time.start();
    
        // create blob from image: 
        cv::Mat input;   
        cv::dnn::blobFromImage(image, input, 1./255., cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true, false);
        std::vector<std::string> output_layer_names = model.getUnconnectedOutLayersNames();
    
        // set the blob to the model:
        model.setInput(input);
    
        // forward pass through the model to carry out the detection:
        cv::Mat output;
        model.forward(output);
    
        float x_factor = image.cols / INPUT_WIDTH;
        float y_factor = image.rows / INPUT_HEIGHT;
        std::vector<int> class_ids;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;
    
        std::vector<Centroid> centroids;
        centroids.reserve(5);

        cv::Mat detectedMat(output.size[1], output.size[2], CV_32F, output.ptr<float>());

        for (int i = 0; i < detectedMat.rows; ++i) 
        {
            float confidence = detectedMat.at<float>(i, 4);
            if (confidence >= CONFIDENCE_THRESHOLD) {
    
                float* classes_scores = &detectedMat.at<float>(i, 5);
                cv::Point class_id;
                double max_class_score;
                cv::Mat scores(1, class_names.size(), CV_32FC1, classes_scores);
                cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
                
                if (max_class_score > SCORE_THRESHOLD) {
    
                    confidences.push_back(confidence);
                    class_ids.push_back(class_id.x);
    
                    int x = static_cast<int>(detectedMat.at<float>(i, 0));
                    int y = static_cast<int>(detectedMat.at<float>(i, 1));
                    int w = static_cast<int>(detectedMat.at<float>(i, 2));
                    int h = static_cast<int>(detectedMat.at<float>(i, 3));
    
                    boxes.push_back(cv::Rect((x - w / 2)*x_factor,
                                             (y - h / 2)*y_factor,
                                             (w*x_factor),
                                             (h*y_factor)));
                }
            }
        }
    
        std::vector<int> nms_result;
        cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, nms_result);
        for (int i = 0; i < nms_result.size(); i++) 
        {
            int idx = nms_result[i];
            Centroid object;
            object.id = class_ids[idx];
            object.conf = confidences[idx];
            object.name = class_names[class_ids[idx]];
            object.box  = boxes[idx];
            object.area = (object.box.width * object.box.height);
            object.center     = cv::Point((object.box.x + (object.box.width/2)), (object.box.y + (object.box.height/2)));
            centroids.push_back(object);
        }
        vector_of_centeroids.push_back(centroids);

    }

    std::cout << "Detection Time: " << time.stop() << "\n";

    // Find Vehicles Trajectory:
    time.start();
    std::vector<Centroid> tracking_vehicles;
    bool first_image = true;
    for(auto & current_vehicles: vector_of_centeroids)
    {
        if (first_image == true) 
        {
            for (auto &vehicle : current_vehicles) 
            {
                tracking_vehicles.push_back(vehicle);
            }

            findLineNumber(tracking_vehicles);
            first_image = false;
        } 
        else 
        {
            findVehiclesTrajectory(tracking_vehicles, current_vehicles);
        }
    }
    std::cout << "Trajectory Time: " << time.stop() << "\n";


    std::map<int, std::string> traffic;
    traffic.insert(std::make_pair(0,"Low"));
    traffic.insert(std::make_pair(1,"Medium"));
    traffic.insert(std::make_pair(2,"High"));

    // create result file:
    std::ofstream result_file ("result.txt");
    if (!result_file.is_open())
    {
        std::cout << "Unable to open file";
    }

    if(tracking_vehicles.size()<10)
        result_file << "Traffic Status: "  << traffic[0] << "\n";
    if( (tracking_vehicles.size()>=10) && (tracking_vehicles.size()<=20) )
        result_file << "Traffic Status: "  << traffic[1] << "\n";
    if(tracking_vehicles.size()>20)
        result_file << "Traffic Status: "  << traffic[2] << "\n";

    result_file << "Number of Cars: "  << tracking_vehicles.size() << "\n";
    result_file << "*****************" << "\n";

    for(int num=0; num < tracking_vehicles.size(); num++)
    {
        result_file << "ID: "    << num << "\n";
        result_file << "Type: "    << tracking_vehicles[num].name << "\n";
        result_file << "Lane Number: "<< tracking_vehicles[num].lane_num << "\n";
        result_file << "Distance: "   << tracking_vehicles[num].distance << "\n";
        result_file << "Speed: "      << tracking_vehicles[num].speed << "\n";
	result_file << "---------------------" << "\n";
    }

    std::cout << "Total Time: " << total_time.stop() << "\n";

    return 0;
}
