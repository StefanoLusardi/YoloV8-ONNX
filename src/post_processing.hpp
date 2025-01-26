#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <utility>
#include <vector>

constexpr int NETWORK_WIDTH = 640; // Now integers
constexpr int NETWORK_HEIGHT = 640;
constexpr float IOU_THRESHOLD = 0.5f;
constexpr int BBOX_COORDS_OFFSET = 4;

struct rect
{
    int x, y, width, height;

    rect(int x, int y, int width, int height)
        : x(x)
        , y(y)
        , width(width)
        , height(height)
    {
    }

    int area() const
    {
        return width * height;
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
    float union_area = static_cast<float>(box1.area() + box2.area()) - intersection;

    return intersection / union_area;
}

std::vector<int> non_maximum_suppression(const std::vector<rect>& boxes,
                                         const std::vector<float>& scores)
{
    std::vector<int> indices(boxes.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::sort(indices.begin(), indices.end(), [&scores](int idx1, int idx2) {
        return scores[idx1] > scores[idx2];
    });

    std::vector<bool> keep(boxes.size(), true);
    std::vector<int> final_indices;

    for (size_t i = 0; i < indices.size(); ++i)
    {
        if (!keep[indices[i]])
            continue;

        final_indices.push_back(indices[i]);

        for (size_t j = i + 1; j < indices.size(); ++j)
        {
            if (!keep[indices[j]])
                continue;

            float iou = calculate_iou(boxes[indices[i]], boxes[indices[j]]);
            if (iou > IOU_THRESHOLD)
            {
                keep[indices[j]] = false;
            }
        }
    }

    return final_indices;
}

rect get_rect(int frame_width, int frame_height, const std::vector<float>& bbox)
{
    float r_w = static_cast<float>(NETWORK_WIDTH) / frame_width;
    float r_h = static_cast<float>(NETWORK_HEIGHT) / frame_height;

    float fleft, fright, ftop, fbottom;
    if (r_h > r_w)
    {
        fleft = bbox[0] - bbox[2] / 2.0f;
        fright = bbox[0] + bbox[2] / 2.0f;
        ftop = bbox[1] - bbox[3] / 2.0f - (NETWORK_HEIGHT - r_w * frame_height) / 2.0f;
        fbottom = bbox[1] + bbox[3] / 2.0f - (NETWORK_HEIGHT - r_w * frame_height) / 2.0f;
        fleft /= r_w;
        fright /= r_w;
        ftop /= r_w;
        fbottom /= r_w;
    }
    else
    {
        fleft = bbox[0] - bbox[2] / 2.0f - (NETWORK_WIDTH - r_h * frame_width) / 2.0f;
        fright = bbox[0] + bbox[2] / 2.0f - (NETWORK_WIDTH - r_h * frame_width) / 2.0f;
        ftop = bbox[1] - bbox[3] / 2.0f;
        fbottom = bbox[1] + bbox[3] / 2.0f;
        fleft /= r_h;
        fright /= r_h;
        ftop /= r_h;
        fbottom /= r_h;
    }

    int left = static_cast<int>(std::round(std::max(0.0f, std::min(fleft, static_cast<float>(frame_width - 1)))));
    int right = static_cast<int>(std::round(std::max(0.0f, std::min(fright, static_cast<float>(frame_width - 1)))));
    int top = static_cast<int>(std::round(std::max(0.0f, std::min(ftop, static_cast<float>(frame_height - 1)))));
    int bottom = static_cast<int>(std::round(std::max(0.0f, std::min(fbottom, static_cast<float>(frame_height - 1)))));

    return rect(left, top, right - left, bottom - top);
}

output postprocess(const float* output_data, const std::vector<int64_t>& shape, int frame_width, int frame_height, float confidence_threshold)
{
    std::vector<rect> boxes;
    std::vector<float> confs;
    std::vector<int> class_ids;

    const auto num_classes = shape[1] - BBOX_COORDS_OFFSET;

    std::vector<std::vector<float>> output_matrix(shape[1], std::vector<float>(shape[2]));

    for (size_t i = 0; i < shape[1]; ++i)
    {
        for (size_t j = 0; j < shape[2]; ++j)
        {
            output_matrix[i][j] = output_data[i * shape[2] + j];
        }
    }

    std::vector<std::vector<float>> transposed_output(shape[2], std::vector<float>(shape[1]));

    for (int i = 0; i < shape[1]; ++i)
    {
        for (int j = 0; j < shape[2]; ++j)
        {
            transposed_output[j][i] = output_matrix[i][j];
        }
    }

    for (int i = 0; i < shape[2]; ++i)
    {
        const auto& row = transposed_output[i];
        const float* bboxes_ptr = row.data();
        const float* scores_ptr = bboxes_ptr + BBOX_COORDS_OFFSET;
        auto max_score_ptr = std::max_element(scores_ptr, scores_ptr + num_classes);
        float score = *max_score_ptr;
        if (score > confidence_threshold)
        {
            boxes.emplace_back(get_rect(frame_width, frame_height, std::vector<float>(bboxes_ptr, bboxes_ptr + BBOX_COORDS_OFFSET)));
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
