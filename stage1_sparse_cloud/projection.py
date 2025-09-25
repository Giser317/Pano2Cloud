import cv2
import numpy as np
from PIL import Image
import argparse


def equirectangular_to_cubemap(equirectangular_img, face_size=512):
    """Convert equirectangular panorama to cubemap faces."""
    h, w, _ = equirectangular_img.shape
    rot_matrices = {
        "front": np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]]),
        "back": np.array([[0, 0, -1], [0, -1, 0], [-1, 0, 0]]),
        "left": np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]),
        "right": np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),
        "top": np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]),
        "bottom": np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]]),
    }

    cubemap_faces = {}
    for face, rot_matrix in rot_matrices.items():
        face_img = np.zeros((face_size, face_size, 3), dtype=np.uint8)
        for i in range(face_size):
            for j in range(face_size):
                x = 2 * (i / face_size) - 1
                y = 2 * (j / face_size) - 1
                z = 1
                norm = np.sqrt(x**2 + y**2 + z**2)
                direction = np.dot(rot_matrix, np.array([x, y, z]) / norm)
                u = 0.5 + np.arctan2(direction[2], direction[0]) / (2 * np.pi)
                v = 0.5 - np.arcsin(direction[1]) / np.pi
                u = int(u * w)
                v = int(v * h)
                face_img[j, i] = equirectangular_img[v % h, u % w]
        cubemap_faces[face] = face_img
    return cubemap_faces


def cubemap_to_equirectangular(cubemap_faces, equi_size=(2048, 1024)):
    """Convert cubemap faces back to equirectangular panorama."""
    w, h = equi_size
    equirectangular_img = np.zeros((h, w, 3), dtype=np.uint8)

    for y in range(h):
        for x in range(w):
            theta = (x / w) * 2 * np.pi - np.pi
            phi = (y / h) * np.pi - np.pi / 2
            direction = np.array(
                [np.cos(phi) * np.cos(theta), np.sin(phi), np.cos(phi) * np.sin(theta)]
            )
            max_axis = np.argmax(np.abs(direction))
            major_dir = np.sign(direction[max_axis])

            if max_axis == 0:
                face = "front" if major_dir > 0 else "back"
                u = direction[2] if major_dir > 0 else -direction[2]
                v = -direction[1]
            elif max_axis == 1:
                face = "top" if major_dir > 0 else "bottom"
                u = direction[0]
                v = direction[2] if major_dir > 0 else -direction[2]
            else:
                face = "right" if major_dir > 0 else "left"
                u = -direction[0] if major_dir > 0 else direction[0]
                v = -direction[1]

            u = (u / np.abs(direction[max_axis]) + 1) / 2
            v = (v / np.abs(direction[max_axis]) + 1) / 2
            face_img = cubemap_faces[face]
            face_size = face_img.shape[0]
            px, py = int(u * face_size), int(v * face_size)
            equirectangular_img[y, x] = face_img[py % face_size, px % face_size]
    return equirectangular_img


def main():
    parser = argparse.ArgumentParser(description="Panorama projection conversion")
    parser.add_argument("--input", type=str, required=True, help="Input panorama image (jpg/png)")
    parser.add_argument("--output", type=str, required=True, help="Output panorama image (jpg/png)")
    parser.add_argument("--face_size", type=int, default=512, help="Cubemap face size")
    parser.add_argument("--width", type=int, default=2048, help="Output panorama width")
    parser.add_argument("--height", type=int, default=1024, help="Output panorama height")
    args = parser.parse_args()

    img = cv2.imread(args.input)
    cubemap_faces = equirectangular_to_cubemap(img, face_size=args.face_size)
    recon = cubemap_to_equirectangular(cubemap_faces, equi_size=(args.width, args.height))
    cv2.imwrite(args.output, recon)


if __name__ == "__main__":
    main()
