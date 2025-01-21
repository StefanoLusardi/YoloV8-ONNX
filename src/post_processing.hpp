#pragma once

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

struct rect
{
    int x, y, width, height;
};

struct output
{
    std::vector<rect> boxes;
    std::vector<float> confs;
    std::vector<int> class_ids;
};

rect get_rect(int frame_width, int frame_height, const std::vector<float>& bbox)
{
    int x = static_cast<int>(bbox[0] * frame_width);
    int y = static_cast<int>(bbox[1] * frame_height);
    int width = static_cast<int>((bbox[2] - bbox[0]) * frame_width);
    int height = static_cast<int>((bbox[3] - bbox[1]) * frame_height);
    return {x, y, width, height};
}

output postprocess(const float* output_data, const std::vector<int64_t>& shape, int frame_width, int frame_height, float confidence_threshold)
{
    std::vector<rect> boxes;
    std::vector<float> confs;
    std::vector<int> class_ids;

    const auto offset = 4;
    const auto num_classes = shape[1] - offset;
    std::vector<std::vector<float>> output_matrix(shape[1], std::vector<float>(shape[2]));

    // Construct output matrix
    for (size_t i = 0; i < shape[1]; ++i)
    {
        for (size_t j = 0; j < shape[2]; ++j)
        {
            output_matrix[i][j] = output_data[i * shape[2] + j];
        }
    }

    std::vector<std::vector<float>> transposed_output(shape[2], std::vector<float>(shape[1]));

    // Transpose output matrix
    for (int i = 0; i < shape[1]; ++i)
    {
        for (int j = 0; j < shape[2]; ++j)
        {
            transposed_output[j][i] = output_matrix[i][j];
        }
    }

    // Get all the YOLO proposals
    for (int i = 0; i < shape[2]; ++i)
    {
        const auto& row = transposed_output[i];
        const float* bboxes_ptr = row.data();
        const float* scores_ptr = bboxes_ptr + 4;
        auto max_score_ptr = std::max_element(scores_ptr, scores_ptr + num_classes);
        float score = *max_score_ptr;
        if (score > confidence_threshold)
        {
            boxes.emplace_back(get_rect(frame_width, frame_height, std::vector<float>(bboxes_ptr, bboxes_ptr + 4)));
            int label = max_score_ptr - scores_ptr;
            confs.emplace_back(score);
            class_ids.emplace_back(label);
        }
    }

    return output{boxes, confs, class_ids};
}
