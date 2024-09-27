import numpy as np
import cv2
import math


def create_diamond_mask(mask_shape, y, x, s):
    mask = np.zeros(mask_shape)
    d = int(s*math.sqrt(2)/2)
    points = [np.array([[x-d, y], [x, y-d], [x+d, y], [x, y+d]])]
    cv2.fillPoly(mask, points, color=(1, 1, 1))
    return mask


def create_triangle_mask(mask_shape, y, x, l):
    mask = np.zeros(mask_shape)
    d = l//2
    h = int(d*math.sqrt(3))
    points = [np.array([[x-d, y], [x+d, y], [x, y-h]])]
    cv2.fillPoly(mask, points, color=(1, 1, 1))
    return mask


def adjust_brightness(image, factor):
    return (image * factor).astype(np.uint8)


def screen_blend(image, mask):
    # Normalize image and mask to the range [0, 1]
    # image_norm = image.astype(np.float32) / 255.0
    image_norm = image
    mask_norm = mask.astype(np.float32) / 255.0

    # Apply the screen blending algorithm
    result = 1 - (1 - image_norm) * (1 - mask_norm)
    #
    # # Denormalize the result back to the range [0, 255]
    # result = (result * 255).astype(np.uint8)

    return result


def add_rain(image, rain_type='weak'):
    strengths = {'weak': 0.002, 'heavy': 0.004, 'torrential': 0.006}
    rain_strength = strengths[rain_type.lower()]

    mask = np.zeros_like(image).astype(np.uint8)
    mask = np.ascontiguousarray(mask, np.uint8)

    # Define parameters for the streaks
    num_streaks_max = int((image.shape[0] * image.shape[1]) * rain_strength)
    streak_length_max = image.shape[1] // 15

    num_streaks = np.random.randint(num_streaks_max * 0.8, num_streaks_max)
    streak_thickness = 1
    streak_length = np.random.randint(streak_length_max * 0.8, streak_length_max)

    # Generate noise for mask
    points = np.random.randint(0, max(image.shape[0], image.shape[1]), size=(num_streaks, 2))

    # Add streaks to the mask
    angle = np.random.uniform(np.pi/4, 3*np.pi/4)
    for point in points:
        change = ((streak_length/2) * np.array([np.cos(angle), np.sin(angle)])).astype(int)
        start_point = point + change
        end_point = point - change
        cv2.line(mask, tuple(start_point), tuple(end_point), (255, 255, 255), streak_thickness)

    # Add a gaussian blur
    kernel_size = (9, 9)
    mask = cv2.GaussianBlur(mask, kernel_size, 0)
    mask = (mask * 0.8).astype(np.uint8)

    # Lower the brightness of the image
    darkened_image = adjust_brightness(image, 0.7)

    # Use the screen blend algorithm to apply the mask to the image
    image_rain = screen_blend(darkened_image, mask)

    return image_rain
