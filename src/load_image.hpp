#pragma once

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <numeric>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <sstream>
#include <string>
#include <vector>

void preprocess(const std::string& imageFilepath, const std::vector<int>& inputDims, std::vector<float>& output)
{
    int width, height, channels;
    unsigned char* imageData = stbi_load(imageFilepath.c_str(), &width, &height, &channels, 3);
    if (!imageData)
    {
        throw std::runtime_error("Failed to load image: " + imageFilepath);
    }

    // Target dimensions
    int targetHeight = inputDims[2];
    int targetWidth = inputDims[3];

    // Resized image buffer
    std::vector<unsigned char> resizedImage(targetHeight * targetWidth * 3);

    // Use stb_image_resize2 for resizing
    stbir_resize_uint8_linear(
        imageData, width, height, 0,
        resizedImage.data(), targetWidth, targetHeight, 0,
        stbir_pixel_layout(0));

    stbi_image_free(imageData); // Free the original image buffer

    // Convert to float and normalize
    std::vector<float> normalizedImage(targetHeight * targetWidth * 3);
    for (int i = 0; i < targetHeight * targetWidth * 3; ++i)
    {
        float value = resizedImage[i] / 255.0f; // Scale to [0, 1]
        if (i % 3 == 0)
        { // R channel
            normalizedImage[i] = (value - 0.485f) / 0.229f;
        }
        else if (i % 3 == 1)
        { // G channel
            normalizedImage[i] = (value - 0.456f) / 0.224f;
        }
        else
        { // B channel
            normalizedImage[i] = (value - 0.406f) / 0.225f;
        }
    }

    // Convert HWC to CHW
    output.resize(targetHeight * targetWidth * 3);
    int imageSize = targetHeight * targetWidth;
    for (int h = 0; h < targetHeight; ++h)
    {
        for (int w = 0; w < targetWidth; ++w)
        {
            output[0 * imageSize + h * targetWidth + w] = normalizedImage[(h * targetWidth + w) * 3 + 0]; // R
            output[1 * imageSize + h * targetWidth + w] = normalizedImage[(h * targetWidth + w) * 3 + 1]; // G
            output[2 * imageSize + h * targetWidth + w] = normalizedImage[(h * targetWidth + w) * 3 + 2]; // B
        }
    }
}
