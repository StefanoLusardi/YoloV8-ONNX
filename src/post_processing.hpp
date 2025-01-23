#pragma once

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <utility>
#include <vector>

constexpr float NETWORK_WIDTH = 640.0f;
constexpr float NETWORK_HEIGHT = 640.0f;
constexpr float IOU_THRESHOLD = 0.5f;

struct rect
{
    int x, y, width, height;
    float area() const
    {
        return static_cast<float>(width * height);
    }
};

struct output
{
    std::vector<rect> boxes;
    std::vector<float> confs;
    std::vector<int> class_ids;
};

float calculate_iou(const rect& box1, const rect& box2)
{
    int x1 = std::max(box1.x, box2.x);
    int y1 = std::max(box1.y, box2.y);
    int x2 = std::min(box1.x + box1.width, box2.x + box2.width);
    int y2 = std::min(box1.y + box1.height, box2.y + box2.height);

    int w = x2 - x1;
    int h = y2 - y1;

    if (w <= 0 || h <= 0)
        return 0.0f;

    float intersection = static_cast<float>(w * h);
    float union_area = box1.area() + box2.area() - intersection;

    return intersection / union_area;
}

std::vector<int> non_maximum_suppression(const std::vector<rect>& boxes,
                                         const std::vector<float>& scores)
{
    std::vector<int> indices(boxes.size());
    std::iota(indices.begin(), indices.end(), 0);

    // Sort indices based on scores
    std::sort(indices.begin(), indices.end(), [&scores](int idx1, int idx2) {
        return scores[idx1] > scores[idx2];
    });

    std::vector<int> final_indices;
    while (!indices.empty())
    {
        int idx = indices.front();
        final_indices.push_back(idx);
        indices.erase(indices.begin());

        auto it = indices.begin();
        while (it != indices.end())
        {
            float iou = calculate_iou(boxes[idx], boxes[*it]);
            if (iou > IOU_THRESHOLD)
            {
                it = indices.erase(it);
            }
            else
            {
                ++it;
            }
        }
    }

    return final_indices;
}

rect get_rect(int frame_width, int frame_height, const std::vector<float>& bbox)
{
    float r_w = NETWORK_WIDTH / static_cast<float>(frame_width);
    float r_h = NETWORK_HEIGHT / static_cast<float>(frame_height);

    int l, r, t, b;
    if (r_h > r_w)
    {
        l = bbox[0] - bbox[2] / 2.f;
        r = bbox[0] + bbox[2] / 2.f;
        t = bbox[1] - bbox[3] / 2.f - (NETWORK_HEIGHT - r_w * frame_height) / 2;
        b = bbox[1] + bbox[3] / 2.f - (NETWORK_HEIGHT - r_w * frame_height) / 2;
        l /= r_w;
        r /= r_w;
        t /= r_w;
        b /= r_w;
    }
    else
    {
        l = bbox[0] - bbox[2] / 2.f - (NETWORK_WIDTH - r_h * frame_width) / 2;
        r = bbox[0] + bbox[2] / 2.f - (NETWORK_WIDTH - r_h * frame_width) / 2;
        t = bbox[1] - bbox[3] / 2.f;
        b = bbox[1] + bbox[3] / 2.f;
        l /= r_h;
        r /= r_h;
        t /= r_h;
        b /= r_h;
    }

    // Clamp the coordinates within the image bounds
    l = std::max(0, std::min(l, frame_width - 1));
    r = std::max(0, std::min(r, frame_width - 1));
    t = std::max(0, std::min(t, frame_height - 1));
    b = std::max(0, std::min(b, frame_height - 1));

    return {l, t, r - l, b - t};
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
    std::vector<int> nms_indices = non_maximum_suppression(boxes, confs);
    output result;
    for (int idx : nms_indices)
    {
        result.boxes.push_back(boxes[idx]);
        result.confs.push_back(confs[idx]);
        result.class_ids.push_back(class_ids[idx]);
    }
    return result;
}
