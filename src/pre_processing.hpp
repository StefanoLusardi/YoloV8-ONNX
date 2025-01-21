#pragma once

#include <array>
#include <cstdint>
#include <vector>

void resize_image_aspect_ratio(
    const std::vector<std::uint8_t>& image,
    int image_width,
    int image_height,
    int image_channels,
    int target_width,
    int target_height,
    std::vector<std::uint8_t>& resized_image)
{
    // Calculate the aspect ratios
    double aspect_ratio_image = static_cast<double>(image_width) / image_height;
    double aspect_ratio_target = static_cast<double>(target_width) / target_height;

    // Determine the scaling factors and new dimensions
    int new_width;
    int new_height;
    if (aspect_ratio_image > aspect_ratio_target)
    {
        new_width = target_width;
        new_height = static_cast<int>(target_width / aspect_ratio_image);
    }
    else
    {
        new_height = target_height;
        new_width = static_cast<int>(target_height * aspect_ratio_image);
    }

    // Calculate padding
    int pad_x = (target_width - new_width) / 2;
    int pad_y = (target_height - new_height) / 2;

    // Scale factors
    double scale_x = static_cast<double>(image_width) / new_width;
    double scale_y = static_cast<double>(image_height) / new_height;

    // Resize with aspect ratio preservation
    for (int y = 0; y < new_height; ++y)
    {
        for (int x = 0; x < new_width; ++x)
        {
            int src_x = std::min(static_cast<int>(x * scale_x), image_width - 1);
            int src_y = std::min(static_cast<int>(y * scale_y), image_height - 1);

            for (int c = 0; c < image_channels; ++c)
            {
                resized_image[((y + pad_y) * target_width + (x + pad_x)) * image_channels + c] =
                    image[(src_y * image_width + src_x) * image_channels + c];
            }
        }
    }
}

std::vector<std::uint8_t> resize_image_aspect_ratio(
    const std::vector<std::uint8_t>& image,
    int image_width,
    int image_height,
    int image_channels,
    int target_width,
    int target_height)
{
    std::vector<std::uint8_t> resized_image(target_width * target_height * image_channels, std::uint8_t(0));
    resize_image_aspect_ratio(image, image_width, image_height, image_channels, target_width, target_height, resized_image);
    return resized_image;
}

template <typename T>
void create_blob(
    const std::vector<std::uint8_t>& image,
    int image_width,
    int image_height,
    int image_channels,
    std::vector<T>& blob,
    T scale_factor = 1.0 / 255.0,
    const std::vector<T>& mean = {0.0, 0.0, 0.0},
    bool swapRB_channels = false)
{
    for (int c = 0; c < image_channels; ++c)
    {
        const int channel_offset = (swapRB_channels ? (2 - c) : c);

        for (int y = 0; y < image_height; ++y)
        {
            for (int x = 0; x < image_width; ++x)
            {
                const int idx_offset = y * image_width + x;
                const int blob_idx = c * image_height * image_width + idx_offset;
                const int image_idx = idx_offset * image_channels + channel_offset;
                blob[blob_idx] = static_cast<T>(image[image_idx]) * scale_factor - mean[c];
            }
        }
    }
}

template <typename T>
std::vector<T> create_blob(
    const std::vector<std::uint8_t>& image,
    int image_width,
    int image_height,
    int image_channels,
    T scale_factor = 1.0 / 255.0,
    const std::vector<T>& mean = {0.0, 0.0, 0.0},
    bool swapRB_channels = false)
{
    std::vector<float> blob(image_channels * image_width * image_height);
    create_blob(image, image_width, image_height, image_channels, blob, scale_factor, mean, swapRB_channels);
    return blob;
}

template <typename T>
std::vector<T> preprocess(
    const std::vector<std::uint8_t>& image,
    int image_width,
    int image_height,
    int image_channels,
    int target_width = 640,
    int target_height = 640,
    T scale_factor = 1.0 / 255.0,
    const std::vector<T>& mean = {0.0f, 0.0f, 0.0f},
    bool swapRB_channels = false)
{
    const std::vector<std::uint8_t> resized_image = resize_image_aspect_ratio(image, image_width, image_height, image_channels, target_width, target_height);
    const std::vector<T> blob = create_blob(resized_image, target_width, target_height, image_channels, scale_factor, mean, swapRB_channels);
    return blob;
}
