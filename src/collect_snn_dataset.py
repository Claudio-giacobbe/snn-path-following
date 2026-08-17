import csv
import time
from pathlib import Path

import cv2
import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient


# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

WIDTH = 256
HEIGHT = 256

SAVE_WIDTH = 64
SAVE_HEIGHT = 64

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = REPO_ROOT / "dataset_snn"
IMAGE_DIR = DATASET_DIR / "images"
CSV_PATH = DATASET_DIR / "labels.csv"

RED_LOW_1 = np.array([0, 50, 30])
RED_HIGH_1 = np.array([15, 255, 255])

RED_LOW_2 = np.array([165, 50, 30])
RED_HIGH_2 = np.array([179, 255, 255])

MIN_AREA = 20

# 5 classes.
# These thresholds are percentages of image width.
# 0 FAR_LEFT, 1 LEFT, 2 CENTER, 3 RIGHT, 4 FAR_RIGHT
CLASS_THRESHOLDS = [0.20, 0.40, 0.60, 0.80]

TARGET_SAMPLES = 3000

# Collect one image every N control loops.
SAVE_EVERY = 2

# ------------------------------------------------------------
# CoppeliaSim
# ------------------------------------------------------------

def connect():
    client = RemoteAPIClient()
    sim = client.require("sim")
    return client, sim


def get_camera(sim):
    try:
        return sim.getObject("/camera")
    except Exception as exc:
        raise RuntimeError("Non trovo '/camera' nella Scene Hierarchy.") from exc


def get_wheels(sim):
    names = [
        "rollingJoint_fl",
        "rollingJoint_rl",
        "rollingJoint_rr",
        "rollingJoint_fr",
    ]

    wheels = []

    for name in names:
        try:
            wheels.append(sim.getObject("/" + name))
        except Exception as exc:
            raise RuntimeError(
                f"Non trovo '/{name}' nella Scene Hierarchy."
            ) from exc

    return wheels


def set_bot_movement(sim, wheels, forward, lateral, rotation):
    velocities = [
        -forward - lateral - rotation,
        -forward + lateral - rotation,
        -forward - lateral + rotation,
        -forward + lateral + rotation,
    ]

    for joint, velocity in zip(wheels, velocities):
        sim.setJointTargetVelocity(joint, float(velocity))


def get_camera_image(sim, camera):
    image, resolution = sim.getVisionSensorImg(camera)

    if image is None:
        return None

    width, height = resolution
    data = np.frombuffer(image, dtype=np.uint8)

    expected = width * height * 3
    if data.size != expected:
        raise RuntimeError(
            f"Immagine camera non valida: {data.size} valori, "
            f"attesi {expected}."
        )

    # CoppeliaSim returns RGB.
    img = data.reshape((height, width, 3))

    # Same vertical correction used by the original project.
    img = cv2.flip(img, 0)

    return img


# ------------------------------------------------------------
# Teacher: current OpenCV detector
# ------------------------------------------------------------

def red_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    mask1 = cv2.inRange(hsv, RED_LOW_1, RED_HIGH_1)
    mask2 = cv2.inRange(hsv, RED_LOW_2, RED_HIGH_2)

    mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((3, 3), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def find_path(mask):
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )

    contours = [
        c for c in contours
        if cv2.contourArea(c) >= MIN_AREA
    ]

    if not contours:
        return None, None, None, 0.0

    biggest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(biggest)

    x, y, w, h = cv2.boundingRect(biggest)

    cx = x + w // 2
    cy = y + h // 2

    return biggest, cx, cy, area


def position_to_class(cx):
    x_norm = cx / float(WIDTH)

    if x_norm < CLASS_THRESHOLDS[0]:
        return 0
    if x_norm < CLASS_THRESHOLDS[1]:
        return 1
    if x_norm < CLASS_THRESHOLDS[2]:
        return 2
    if x_norm < CLASS_THRESHOLDS[3]:
        return 3
    return 4


# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------

def prepare_dataset():
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    new_csv = not CSV_PATH.exists()

    csv_file = open(
        CSV_PATH,
        "a",
        newline="",
        encoding="utf-8",
    )

    writer = csv.writer(csv_file)

    if new_csv:
        writer.writerow(["filename", "label", "cx", "cy", "area"])

    return csv_file, writer


def main():
    client = None
    sim = None
    csv_file = None

    try:
        csv_file, writer = prepare_dataset()

        client, sim = connect()

        camera = get_camera(sim)
        wheels = get_wheels(sim)

        print("=" * 60)
        print("DATASET COLLECTION PER SNN PATH FOLLOWING")
        print("=" * 60)
        print(f"Output: {DATASET_DIR.resolve()}")
        print(f"Target samples: {TARGET_SAMPLES}")
        print()
        print("Classi:")
        print("  0 FAR_LEFT")
        print("  1 LEFT")
        print("  2 CENTER")
        print("  3 RIGHT")
        print("  4 FAR_RIGHT")
        print()

        sim.startSimulation()
        time.sleep(1.0)

        saved = 0
        frame = 0

        while saved < TARGET_SAMPLES:
            frame += 1

            img = get_camera_image(sim, camera)

            if img is None:
                set_bot_movement(sim, wheels, 0, 0, 0)
                time.sleep(0.02)
                continue

            img = cv2.resize(
                img,
                (WIDTH, HEIGHT),
                interpolation=cv2.INTER_AREA,
            )

            mask = red_mask(img)
            contour, cx, cy, area = find_path(mask)

            # Safety: if teacher cannot see the path, stop and don't save.
            if contour is None:
                set_bot_movement(sim, wheels, 0, 0, 0)

                if frame % 30 == 0:
                    print("Percorso non trovato: raccolta sospesa.")

                time.sleep(0.02)
                continue

            label = position_to_class(cx)

            # Save RGB image. The SNN will learn from the camera image,
            # while OpenCV only supplies the training label.
            filename = f"frame_{saved:06d}.png"
            path = IMAGE_DIR / filename

            image_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            ok = cv2.imwrite(str(path), image_bgr)

            if not ok:
                raise RuntimeError(f"Impossibile salvare {path}")

            writer.writerow([
                filename,
                label,
                cx,
                cy,
                f"{area:.3f}",
            ])
            csv_file.flush()

            saved += 1

            # Keep the robot moving using the original controller logic.
            ref_x = WIDTH // 2
            ref_y = HEIGHT - 1
            denominator = ref_y - cy

            if denominator == 0:
                w_rot = 0.0
            else:
                w_rot = np.arctan(
                    (ref_x - cx) / denominator
                )

            angle = abs(np.degrees(w_rot))

            if angle <= 10:
                n, m = 4.5, 0.8
            elif angle <= 25:
                n, m = 4.2, 1.5
            elif angle <= 40:
                n, m = 3.8, 2.3
            elif angle <= 55:
                n, m = 3.2, 3.2
            elif angle <= 70:
                n, m = 2.6, 4.0
            else:
                n, m = 2.0, 4.5

            rotation = float(np.clip(m * w_rot, -4.5, 4.5))

            set_bot_movement(
                sim,
                wheels,
                -n,
                0,
                rotation,
            )

            if saved % 50 == 0:
                print(
                    f"Salvati {saved}/{TARGET_SAMPLES} | "
                    f"class={label} | cx={cx} | angle={angle:.1f}"
                )

            if frame % SAVE_EVERY != 0:
                continue

            time.sleep(0.02)

        print()
        print("Dataset completato.")

    except KeyboardInterrupt:
        print("\nRaccolta interrotta dall'utente.")

    except Exception as exc:
        print("\nERRORE:")
        print(exc)
        raise

    finally:
        if sim is not None:
            try:
                sim.stopSimulation()
            except Exception:
                pass

        if csv_file is not None:
            csv_file.close()

        if sim is not None:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

        print("Robot fermato.")


if __name__ == "__main__":
    main()
