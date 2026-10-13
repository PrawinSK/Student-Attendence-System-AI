import os, cv2
from face_engine import encode_student_image, compute_similarity

imgs = [
    r'C:/Users/prawi/.gemini/antigravity/brain/f6acf7f7-0f86-4683-9c67-2a337b49234c/.user_uploaded/media_1791823253477_0e9503d0.jpg',
    r'C:/Users/prawi/.gemini/antigravity/brain/f6acf7f7-0f86-4683-9c67-2a337b49234c/.user_uploaded/media_1791823261807_03ab8c04.jpg',
    r'C:/Users/prawi/.gemini/antigravity/brain/f6acf7f7-0f86-4683-9c67-2a337b49234c/.user_uploaded/media_1791823275567_01cca16e.jpg'
]

ref_emb = encode_student_image(imgs[0])
print('Reference embedding generated successfully (128-D vector).\n')

for i in [1, 2]:
    probe_emb = encode_student_image(imgs[i])
    l2_dist, cosine_sim = compute_similarity(ref_emb, probe_emb)
    verdict = "MATCH (PRESENT)" if (l2_dist < 1.128 or cosine_sim > 0.363) else "UNKNOWN"
    print(f"Testing Photo {i+1} against Reference:")
    print(f"  • L2 (Euclidean) Distance: {l2_dist:.4f} (Match threshold < 1.128)")
    print(f"  • Cosine Similarity:       {cosine_sim:.4f} (Match threshold > 0.363)")
    print(f"  • Result:                  {verdict}\n")
