import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np
import torch

from coppeliasim_zmqremoteapi_client import RemoteAPIClient

from snn_model import PathSNN, CLASS_NAMES, CLASS_POSITION


# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO_ROOT / "models" / "path_snn.pth"

DISPLAY_WIDTH = 256
DISPLAY_HEIGHT = 256

# Number of predictions kept for temporal smoothing.
SMOOTHING_WINDOW = 5

# Controller.
FORWARD_SPEED = 3.5
MAX_ROTATION = 4.5
ROTATION_GAIN = 3.0


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
        raise RuntimeError(
            "Non trovo '/camera' nella Scene Hierarchy."
        ) from exc


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
        sim.setJointTargetVelocity(
            joint,
            float(velocity),
        )


def get_camera_image(sim, camera):
    image, resolution = sim.getVisionSensorImg(camera)

    if image is None:
        return None

    width, height = resolution

    data = np.frombuffer(
        image,
        dtype=np.uint8,
    )

    expected = width * height * 3

    if data.size != expected:
        raise RuntimeError(
            f"Immagine camera non valida: "
            f"{data.size}, attesi {expected}."
        )

    # CoppeliaSim -> RGB.
    img = data.reshape(
        (height, width, 3)
    )

    img = cv2.flip(img, 0)

    return img


# ------------------------------------------------------------
# SNN
# ------------------------------------------------------------

def load_model(device):
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modello non trovato: {MODEL_PATH}\n"
            "Esegui prima train_snn.py."
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
    )

    model = PathSNN(
        input_size=checkpoint["input_size"],
        hidden_size=checkpoint["hidden_size"],
        num_outputs=checkpoint["num_outputs"],
        beta=checkpoint["beta"],
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model, checkpoint


def preprocess_image(img, image_size):
    resized = cv2.resize(
        img,
        (image_size, image_size),
        interpolation=cv2.INTER_AREA,
    )

    x = resized.astype(
        np.float32
    ) / 255.0

    # RGB HWC -> CHW -> flattened.
    x = torch.from_numpy(
        x
    ).permute(2, 0, 1).reshape(1, -1)

    return x


@torch.no_grad()
def predict(model, image, image_size, num_steps, device):
    x = preprocess_image(
        image,
        image_size,
    ).to(device)

    _, mem_rec = model(
        x,
        num_steps=num_steps,
    )

    # Final membrane potential = class score.
    scores = mem_rec[-1][0]

    probabilities = torch.softmax(
        scores,
        dim=0,
    )

    predicted_class = int(
        torch.argmax(probabilities).item()
    )

    # Continuous position obtained from the class probabilities.
    class_position = CLASS_POSITION.to(
        probabilities.device
    )

    position = float(
        torch.sum(
            probabilities * class_position
        ).item()
    )

    confidence = float(
        probabilities[predicted_class].item()
    )

    return (
        predicted_class,
        position,
        confidence,
        probabilities.cpu().numpy(),
    )


# ------------------------------------------------------------
# Robot controller
# ------------------------------------------------------------

def run_robot(sim):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model, checkpoint = load_model(device)

    image_size = int(
        checkpoint["image_size"]
    )

    num_steps = int(
        checkpoint["num_steps"]
    )

    camera = get_camera(sim)
    wheels = get_wheels(sim)

    history = []

    print("=" * 60)
    print("SNN PATH FOLLOWING")
    print("=" * 60)
    print("Device:", device)
    print("Model:", MODEL_PATH)
    print("Classes:", CLASS_NAMES)
    print()

    while True:
        img = get_camera_image(
            sim,
            camera,
        )

        if img is None:
            set_bot_movement(
                sim,
                wheels,
                0,
                0,
                0,
            )
            time.sleep(0.02)
            continue

        img = cv2.resize(
            img,
            (DISPLAY_WIDTH, DISPLAY_HEIGHT),
            interpolation=cv2.INTER_AREA,
        )

        (
            predicted_class,
            position,
            confidence,
            probabilities,
        ) = predict(
            model,
            img,
            image_size,
            num_steps,
            device,
        )

        history.append(position)

        if len(history) > SMOOTHING_WINDOW:
            history.pop(0)

        smoothed_position = float(
            np.mean(history)
        )

        # position:
        # -1 = far left
        #  0 = center
        # +1 = far right
        #
        # Existing wheel convention in the original project:
        # positive rotation corresponds to a path left of center.
        rotation = -ROTATION_GAIN * smoothed_position

        rotation = float(
            np.clip(
                rotation,
                -MAX_ROTATION,
                MAX_ROTATION,
            )
        )

        # Reduce forward speed when the network sees a strong lateral error.
        speed_factor = max(
            0.45,
            1.0 - 0.45 * abs(smoothed_position),
        )

        forward = FORWARD_SPEED * speed_factor

        set_bot_movement(
            sim,
            wheels,
            -forward,
            0,
            rotation,
        )

        # Debug display.
        display = cv2.cvtColor(
            img,
            cv2.COLOR_RGB2BGR,
        )

        text1 = (
            f"SNN: {CLASS_NAMES[predicted_class]} "
            f"conf={confidence:.2f}"
        )

        text2 = (
            f"position={smoothed_position:+.2f} "
            f"rotation={rotation:+.2f}"
        )

        cv2.putText(
            display,
            text1,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            display,
            text2,
            (10, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )

        cv2.imshow(
            "SNN PATH FOLLOWING",
            display,
        )

        if (
            cv2.waitKey(1) & 0xFF
        ) == ord("q"):
            break

        time.sleep(0.02)

    set_bot_movement(
        sim,
        wheels,
        0,
        0,
        0,
    )


def main():
    client = None
    sim = None

    try:
        client, sim = connect()

        print("Connessione a CoppeliaSim OK.")
        print("Avvio simulazione...")

        sim.startSimulation()

        time.sleep(1.0)

        run_robot(sim)

    except KeyboardInterrupt:
        print("\nProgramma interrotto.")

    except Exception:
        print("\nERRORE:")
        traceback.print_exc()

    finally:
        if sim is not None:
            try:
                sim.stopSimulation()
            except Exception:
                pass

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        print("Robot fermato.")


if __name__ == "__main__":
    main()
