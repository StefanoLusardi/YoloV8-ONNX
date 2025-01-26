#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "post_processing.hpp"
#include "pre_processing.hpp"

#include <cmath>
#include <filesystem>
#include <iostream>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <stb_image.h>
#include <stb_image_write.h>
#include <string>
#include <vector>

struct image
{
    std::vector<uint8_t> data;
    int width;
    int height;
    int channels;
};

void draw_bbox(std::vector<uint8_t>& image_data, int width, int height, int channels,
               const rect& box, int class_id)
{
    int x1 = std::max(0, box.x);
    int y1 = std::max(0, box.y);
    int x2 = std::min(width - 1, box.x + box.width);
    int y2 = std::min(height - 1, box.y + box.height);

    // Choose color based on class_id (example)
    uint8_t r = 0, g = 0, b = 0;
    switch (class_id % 3)
    {
    case 0:
        r = 255;
        break;
    case 1:
        g = 255;
        break;
    case 2:
        b = 255;
        break;
    }

    // Draw rectangle borders (simple implementation)
    int thickness = 2;
    for (int t = 0; t < thickness; ++t)
    {
        // Top horizontal line
        for (int x = x1; x <= x2; ++x)
        {
            int y = y1 + t;
            if (y >= 0 && y < height)
            {
                image_data[(y * width + x) * channels + 0] = r;
                image_data[(y * width + x) * channels + 1] = g;
                image_data[(y * width + x) * channels + 2] = b;
            }
        }
        // Bottom horizontal line
        for (int x = x1; x <= x2; ++x)
        {
            int y = y2 - t;
            if (y >= 0 && y < height)
            {
                image_data[(y * width + x) * channels + 0] = r;
                image_data[(y * width + x) * channels + 1] = g;
                image_data[(y * width + x) * channels + 2] = b;
            }
        }
        // Left vertical line
        for (int y = y1; y <= y2; ++y)
        {
            int x = x1 + t;
            if (x >= 0 && x < width)
            {
                image_data[(y * width + x) * channels + 0] = r;
                image_data[(y * width + x) * channels + 1] = g;
                image_data[(y * width + x) * channels + 2] = b;
            }
        }
        // Right vertical line
        for (int y = y1; y <= y2; ++y)
        {
            int x = x2 - t;
            if (x >= 0 && x < width)
            {
                image_data[(y * width + x) * channels + 0] = r;
                image_data[(y * width + x) * channels + 1] = g;
                image_data[(y * width + x) * channels + 2] = b;
            }
        }
    }
}

// Save the image with drawn bounding boxes
void save_image_with_bboxes(const image& img, const output& result,
                            const std::filesystem::path& output_path)
{
    std::vector<uint8_t> image_data_copy = img.data;

    for (size_t i = 0; i < result.boxes.size(); ++i)
    {
        draw_bbox(image_data_copy, img.width, img.height, img.channels, result.boxes[i], result.class_ids[i]);
    }

    if (stbi_write_jpg(output_path.string().c_str(), img.width, img.height, img.channels, image_data_copy.data(), 95))
    {
        std::cout << "Image with bounding boxes saved to: " << output_path << std::endl;
    }
    else
    {
        std::cerr << "Failed to save image: " << output_path << std::endl;
    }
}

image load_image(const std::filesystem::path& image_path)
{
    int width, height, channels;
    uint8_t* image_data = stbi_load(image_path.c_str(), &width, &height, &channels, 3);
    if (!image_data)
    {
        throw std::runtime_error("Failed to load image: " + image_path.string());
    }

    size_t total_pixels = width * height * channels;
    image img{
        .data = std::vector<uint8_t>(total_pixels),
        .width = width,
        .height = height,
        .channels = channels};

    std::memcpy(img.data.data(), image_data, total_pixels);
    stbi_image_free(image_data);
    return img;
}

std::string print_shape(const std::vector<std::int64_t>& v)
{
    std::stringstream ss("");
    for (std::size_t i = 0; i < v.size() - 1; i++)
        ss << v[i] << "x";
    ss << v[v.size() - 1];
    return ss.str();
}

int main()
{
    const std::basic_string<ORTCHAR_T> model_path = "yolov8n.onnx";
    const std::filesystem::path image_path = "images/dog.png";

    std::cout
        << "\nModel: " << model_path
        << "\nImage: " << image_path
        << std::endl;

    // YoloV8 target size
    int target_width = 640;
    int target_height = 640;

    image img = load_image(image_path);
    std::vector<float> img_blob = preprocess<float>(img.data, img.width, img.height, img.channels, target_width, target_height, 1.0 / 255.0, std::vector<float>{0.0, 0.0, 0.0}, true);

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "YOLOv8");
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    Ort::Session session(env, model_path.c_str(), session_options);

    // Inputs
    std::vector<const char*> input_names;
    std::vector<std::string> input_names_str;
    std::vector<std::vector<std::int64_t>> input_shapes;
    for (std::size_t i = 0; i < session.GetInputCount(); i++)
    {
        auto input_name = session.GetInputNameAllocated(i, allocator);
        auto input_shape = session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        auto input_type = session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetElementType();
        auto input_count = session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetElementCount();

        // Adjust only dimensions equal to -1 (dynamic axis case)
        for (size_t idx = 0; idx < input_shape.size(); ++idx)
        {
            if (input_shape[idx] == -1)
            {
                // Replace with actual size
                if (idx == 0)
                    input_shape[idx] = 1; // Batch size
                else if (idx == 1)
                    input_shape[idx] = img.channels; // Number of channels (usually 3)
                else if (idx == 2)
                    input_shape[idx] = target_height; // Image height (e.g., 640)
                else if (idx == 3)
                    input_shape[idx] = target_width; // Image width (e.g., 640)
            }
        }

        input_shapes.emplace_back(input_shape);
        input_names_str.emplace_back(input_name.get());

        std::cout
            << "\nInput: " << i
            << "\n - name: " << input_name
            << "\n - shape: " << print_shape(input_shape)
            << "\n - element type: " << input_type
            << "\n - element count: " << input_count
            << std::endl;
    }

    for (auto&& s : input_names_str)
    {
        input_names.emplace_back(s.c_str());
    }

    // Outputs
    std::vector<const char*> output_names;
    std::vector<std::string> output_names_str;
    std::vector<std::vector<std::int64_t>> output_shapes;
    for (std::size_t i = 0; i < session.GetOutputCount(); i++)
    {
        auto output_name = session.GetOutputNameAllocated(i, allocator);
        auto output_shape = session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        auto output_type = session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetElementType();
        auto output_count = session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetElementCount();

        for (auto& s : output_shape)
        {
            if (s < 0)
                s = 1;
        }
        output_shapes.emplace_back(output_shape);
        output_names_str.emplace_back(output_name.get());

        std::cout
            << "\nOutput: " << i
            << "\n - name: " << output_name
            << "\n - shape: " << print_shape(output_shape)
            << "\n - element type: " << output_type
            << "\n - element count: " << output_count
            << std::endl;
    }

    for (auto&& s : output_names_str)
    {
        output_names.emplace_back(s.c_str());
    }

    // Create inference tensor
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    std::vector<Ort::Value> input_tensors;
    input_tensors.emplace_back(
        Ort::Value::CreateTensor<float>(
            memory_info,
            img_blob.data(),
            img_blob.size(),
            input_shapes.at(0).data(),
            input_shapes.at(0).size()));

    // Run inference
    std::vector<Ort::Value> output_tensors = session.Run(
        Ort::RunOptions{nullptr},
        input_names.data(),
        input_tensors.data(),
        input_names.size(),
        output_names.data(),
        output_names.size());

    // Postprocess output
    float* outputs_raw = output_tensors.front().GetTensorMutableData<float>();
    std::vector<int64_t> output_shape = session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();

    const float confidence = 0.8f;
    const auto output_postprocess = postprocess(outputs_raw, output_shape, img.width, img.height, confidence);

    // Print detections
    std::cout << "\nDetection results:\n";
    for (int i = 0; i < output_postprocess.class_ids.size(); ++i)
    {
        std::cout
            << "Class: " << output_postprocess.class_ids[i]
            << ", Confidence: " << output_postprocess.confs[i]
            << std::endl;
    }

    save_image_with_bboxes(img, output_postprocess, "output.jpg");

    return EXIT_SUCCESS;
}
