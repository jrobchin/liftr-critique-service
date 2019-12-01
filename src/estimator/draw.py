import cv2

from src.estimator import entities


def draw_humans(npimg, humans, imgcopy=False):
    if imgcopy:
        npimg = np.copy(npimg)
    image_h, image_w = npimg.shape[:2]
    centers = {}
    for human in humans:
        for part in entities.Part:
            
        

    for human in humans:
        # draw point
        for i in range(entities.Part.BACKGROUND.value):
            if i not in human.body_parts.keys():
                continue

            body_part = human.body_parts[i]
            center = (int(body_part.x * image_w + 0.5),
                        int(body_part.y * image_h + 0.5))
            centers[i] = center
            cv2.circle(npimg, center, 3,
                        entities.part_colors[i], thickness=3, lineType=8, shift=0)

        # draw line
        for pair_order, pair in enumerate(entities.part_pairs):
            if pair[0] not in human.body_parts.keys() or pair[1] not in human.body_parts.keys():
                continue

            cv2.line(npimg, centers[pair[0]], centers[pair[1]], entities.part_colors[pair_order], 3)

    return npimg