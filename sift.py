import cv2
query_img = cv2.imread('query.png', cv2.IMREAD_GRAYSCALE)   
target_img = cv2.imread('target.jpg', cv2.IMREAD_GRAYSCALE) 

sift = cv2.SIFT_create()

kp1, des1 = sift.detectAndCompute(query_img, None)
kp2, des2 = sift.detectAndCompute(target_img, None)

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1, des2, k=2)
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

target_matched_pts = [kp2[m.trainIdx].pt for m in good_matches]

output_img = cv2.cvtColor(target_img, cv2.COLOR_GRAY2BGR)
for pt in target_matched_pts:
    cv2.circle(output_img, (int(pt[0]), int(pt[1])), 4, (0, 255, 0), -1)

matched_img = cv2.drawMatches(query_img, kp1, target_img, kp2, good_matches, None, flags=2)

cv2.imwrite('matched_points.jpg', matched_img)
cv2.imwrite('target_with_keypoints.jpg', output_img)

print("Saved 'matched_points.jpg' and 'target_with_keypoints.jpg'")
