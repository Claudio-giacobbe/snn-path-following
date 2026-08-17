"""
evaluate_snn.py

Valutazione quantitativa della SNN per il path following in CoppeliaSim.

Basato sui tre file del progetto:
- train_snn.py: PathSNN, 5 classi e 20 timestep
- snn_model.py: posizione continua [-1, 1] ottenuta dalle probabilità
- run_snn_robot.py: camera, smoothing e controller

Metriche calcolate:
1. durata e FPS
2. confidence media/minima
3. errore laterale SNN medio assoluto e massimo
4. tempo in cui la SNN vede una posizione "molto laterale"
5. errore laterale della pista visto dalla CAMERA (proxy indipendente)
6. percentuale di frame in cui la pista rossa è visibile
7. tempo stimato fuori pista
8. durata massima consecutiva fuori pista
9. numero di recuperi dalla condizione fuori pista
10. intensità media e massima della correzione di sterzo
11. numero di cambi di direzione dello sterzo
12. salvataggio di tutte le misure in CSV

NOTA IMPORTANTE:
"fuori pista" qui è una STIMA basata sull'immagine della pista rossa.
Non è una distanza geometrica reale in metri, perché nei tre file originali
non viene definita la geometria della pista né un riferimento world-space.
"""

import sys
import csv
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np
import torch

from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from snn_model import PathSNN, CLASS_NAMES, CLASS_POSITION


# ============================================================
# CONFIG
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO_ROOT / "models" / "path_snn.pth"
CSV_OUTPUT = REPO_ROOT / "results" / "snn_evaluation.csv"

DISPLAY_WIDTH = 256
DISPLAY_HEIGHT = 256

SMOOTHING_WINDOW = 5

FORWARD_SPEED = 3.5
MAX_ROTATION = 4.5
ROTATION_GAIN = 3.0

# Valori per la valutazione
LOW_CONFIDENCE = 0.50
SNN_LARGE_ERROR = 0.75

# Soglie per la detection della pista rossa.
# Possono essere modificate in base alla scena di CoppeliaSim.
RED_S_MIN = 80
RED_V_MIN = 60
RED_H_LOW_1 = 0
RED_H_HIGH_1 = 15
RED_H_LOW_2 = 165
RED_H_HIGH_2 = 179

# Una deviazione immagine oltre questa soglia viene considerata
# "potenzialmente fuori pista".
# 0.0 = centro immagine, 1.0 = bordo immagine.
CAMERA_OFF_TRACK_THRESHOLD = 0.55

# Se la pista occupa troppo poco spazio nell'immagine,
# consideriamo la detection poco affidabile.
MIN_RED_PIXELS = 30

# Durata della prova. Metti None per usare Q per fermare.
TEST_DURATION_SECONDS = 60.0


# ============================================================
# COPPELIASIM
# ============================================================

def connect():
    client = RemoteAPIClient()
    sim = client.require("sim")
    return client, sim


def get_camera(sim):
    return sim.getObject("/camera")


def get_wheels(sim):
    names = [
        "rollingJoint_fl",
        "rollingJoint_rl",
        "rollingJoint_rr",
        "rollingJoint_fr",
    ]

    return [sim.getObject("/" + name) for name in names]


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
            f"Immagine camera non valida: {data.size}, attesi {expected}"
        )

    img = data.reshape((height, width, 3))
    img = cv2.flip(img, 0)

    return img


# ============================================================
# SNN
# ============================================================

def load_model(device):
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modello non trovato: {MODEL_PATH}. "
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

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint


def preprocess_image(img, image_size):
    resized = cv2.resize(
        img,
        (image_size, image_size),
        interpolation=cv2.INTER_AREA,
    )

    x = resized.astype(np.float32) / 255.0

    x = (
        torch.from_numpy(x)
        .permute(2, 0, 1)
        .reshape(1, -1)
    )

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

    # Come nel run_snn_robot.py:
    # il potenziale finale viene usato come score.
    scores = mem_rec[-1][0]

    probabilities = torch.softmax(
        scores,
        dim=0,
    )

    predicted_class = int(
        torch.argmax(probabilities).item()
    )

    class_position = CLASS_POSITION.to(
        probabilities.device
    )

    # Posizione continua:
    # -1 = FAR_LEFT
    #  0 = CENTER
    # +1 = FAR_RIGHT
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


# ============================================================
# DETECTION INDIPENDENTE DELLA PISTA
# ============================================================

def detect_red_track(image):
    """
    Cerca la pista rossa nell'immagine.

    Restituisce:
        visible: True/False
        center_x: centro della pista normalizzato [-1, +1]
        pixel_count: numero di pixel rossi
        mask: maschera binaria
    """

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    lower1 = np.array(
        [RED_H_LOW_1, RED_S_MIN, RED_V_MIN],
        dtype=np.uint8,
    )

    upper1 = np.array(
        [RED_H_HIGH_1, 255, 255],
        dtype=np.uint8,
    )

    lower2 = np.array(
        [RED_H_LOW_2, RED_S_MIN, RED_V_MIN],
        dtype=np.uint8,
    )

    upper2 = np.array(
        [RED_H_HIGH_2, 255, 255],
        dtype=np.uint8,
    )

    mask1 = cv2.inRange(
        hsv,
        lower1,
        upper1,
    )

    mask2 = cv2.inRange(
        hsv,
        lower2,
        upper2,
    )

    mask = cv2.bitwise_or(mask1, mask2)

    # Pulizia semplice del rumore.
    kernel = np.ones((3, 3), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
    )

    ys, xs = np.where(mask > 0)

    pixel_count = len(xs)

    if pixel_count < MIN_RED_PIXELS:
        return False, None, pixel_count, mask

    # Diamo più peso alla parte bassa dell'immagine,
    # che normalmente è più utile per capire dove passa la pista.
    h = image.shape[0]

    bottom_limit = int(h * 0.75)

    bottom_xs = xs[ys >= bottom_limit]

    if len(bottom_xs) >= max(10, MIN_RED_PIXELS // 3):
        xs_used = bottom_xs
    else:
        xs_used = xs

    center_x = float(np.mean(xs_used))

    image_center = image.shape[1] / 2.0

    # -1 = pista a sinistra
    #  0 = pista al centro
    # +1 = pista a destra
    normalized_error = (
        (center_x - image_center)
        / image_center
    )

    normalized_error = float(
        np.clip(
            normalized_error,
            -1.0,
            1.0,
        )
    )

    return True, normalized_error, pixel_count, mask


# ============================================================
# METRICHE
# ============================================================

class Metrics:
    def __init__(self):
        self.start_time = time.perf_counter()

        self.frames = 0
        self.visible_track_frames = 0
        self.low_confidence_frames = 0
        self.large_snn_error_frames = 0
        self.off_track_frames = 0

        self.sum_abs_snn_error = 0.0
        self.max_abs_snn_error = 0.0

        self.sum_abs_camera_error = 0.0
        self.max_abs_camera_error = 0.0

        self.sum_confidence = 0.0
        self.min_confidence = 1.0

        self.sum_abs_rotation = 0.0
        self.max_abs_rotation = 0.0

        self.steering_sign_changes = 0
        self.previous_rotation_sign = 0

        self.current_off_track_time = 0.0
        self.max_off_track_time = 0.0
        self.off_track_start = None

        self.recovery_events = 0
        self.previous_off_track = False

        self.last_time = time.perf_counter()

    def update(
        self,
        position,
        confidence,
        rotation,
        track_visible,
        camera_error,
        now,
    ):
        self.frames += 1

        abs_snn_error = abs(position)

        self.sum_abs_snn_error += abs_snn_error
        self.max_abs_snn_error = max(
            self.max_abs_snn_error,
            abs_snn_error,
        )

        self.sum_confidence += confidence
        self.min_confidence = min(
            self.min_confidence,
            confidence,
        )

        if confidence < LOW_CONFIDENCE:
            self.low_confidence_frames += 1

        if abs_snn_error > SNN_LARGE_ERROR:
            self.large_snn_error_frames += 1

        abs_rotation = abs(rotation)

        self.sum_abs_rotation += abs_rotation
        self.max_abs_rotation = max(
            self.max_abs_rotation,
            abs_rotation,
        )

        sign = 0

        if rotation > 0.15:
            sign = 1
        elif rotation < -0.15:
            sign = -1

        if (
            sign != 0
            and self.previous_rotation_sign != 0
            and sign != self.previous_rotation_sign
        ):
            self.steering_sign_changes += 1

        if sign != 0:
            self.previous_rotation_sign = sign

        if track_visible and camera_error is not None:
            self.visible_track_frames += 1

            abs_camera_error = abs(camera_error)

            self.sum_abs_camera_error += abs_camera_error
            self.max_abs_camera_error = max(
                self.max_abs_camera_error,
                abs_camera_error,
            )

            off_track = (
                abs_camera_error
                > CAMERA_OFF_TRACK_THRESHOLD
            )
        else:
            # Se la pista non viene vista, NON la contiamo
            # automaticamente come uscita: potrebbe essere
            # semplicemente nascosta dall'inquadratura.
            off_track = False

        if off_track:
            self.off_track_frames += 1

            if not self.previous_off_track:
                self.off_track_start = now

        else:
            if self.previous_off_track:
                self.recovery_events += 1

                if self.off_track_start is not None:
                    duration = now - self.off_track_start

                    self.max_off_track_time = max(
                        self.max_off_track_time,
                        duration,
                    )

                self.off_track_start = None

        self.previous_off_track = off_track

    def finish(self):
        end_time = time.perf_counter()

        duration = end_time - self.start_time

        if self.previous_off_track and self.off_track_start is not None:
            duration_off = end_time - self.off_track_start

            self.max_off_track_time = max(
                self.max_off_track_time,
                duration_off,
            )

        if self.frames == 0:
            return {}

        visible_ratio = (
            self.visible_track_frames
            / self.frames
        )

        return {
            "duration_s": duration,
            "frames": self.frames,
            "fps": self.frames / max(duration, 1e-6),

            "mean_abs_snn_error": (
                self.sum_abs_snn_error / self.frames
            ),
            "max_abs_snn_error": self.max_abs_snn_error,

            "mean_confidence": (
                self.sum_confidence / self.frames
            ),
            "min_confidence": self.min_confidence,

            "low_confidence_percent": (
                100.0
                * self.low_confidence_frames
                / self.frames
            ),

            "large_snn_error_percent": (
                100.0
                * self.large_snn_error_frames
                / self.frames
            ),

            "track_visible_percent": (
                100.0 * visible_ratio
            ),

            "mean_abs_camera_error": (
                self.sum_abs_camera_error
                / max(self.visible_track_frames, 1)
            ),

            "max_abs_camera_error": (
                self.max_abs_camera_error
            ),

            "off_track_percent_of_all_frames": (
                100.0
                * self.off_track_frames
                / self.frames
            ),

            "off_track_percent_when_track_visible": (
                100.0
                * self.off_track_frames
                / max(self.visible_track_frames, 1)
            ),

            "max_continuous_off_track_s": (
                self.max_off_track_time
            ),

            "recovery_events": self.recovery_events,

            "mean_abs_rotation": (
                self.sum_abs_rotation / self.frames
            ),

            "max_abs_rotation": self.max_abs_rotation,

            "steering_direction_changes": (
                self.steering_sign_changes
            ),
        }


# ============================================================
# CSV
# ============================================================

CSV_FIELDS = [
    "time_s",
    "predicted_class",
    "position",
    "confidence",
    "camera_track_visible",
    "camera_error",
    "red_pixels",
    "rotation",
    "forward",
]


def write_csv(rows):
    with open(
        CSV_OUTPUT,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=CSV_FIELDS,
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# OUTPUT
# ============================================================

def print_report(report):
    print()
    print("=" * 70)
    print("        RISULTATI VALUTAZIONE SNN PATH FOLLOWING")
    print("=" * 70)

    print(f"Durata test                         : {report['duration_s']:.2f} s")
    print(f"Frame analizzati                    : {report['frames']}")
    print(f"FPS medi                            : {report['fps']:.2f}")
    print()

    print("--- SNN ---")
    print(
        f"Errore laterale SNN medio assoluto : "
        f"{report['mean_abs_snn_error']:.3f}"
    )
    print(
        f"Errore laterale SNN massimo        : "
        f"{report['max_abs_snn_error']:.3f}"
    )
    print(
        f"Confidence media                   : "
        f"{report['mean_confidence']:.3f}"
    )
    print(
        f"Confidence minima                  : "
        f"{report['min_confidence']:.3f}"
    )
    print(
        f"Frame con confidence < {LOW_CONFIDENCE:.2f}: "
        f"{report['low_confidence_percent']:.2f}%"
    )
    print(
        f"Frame con |errore SNN| > "
        f"{SNN_LARGE_ERROR:.2f}: "
        f"{report['large_snn_error_percent']:.2f}%"
    )
    print()

    print("--- PISTA / USCITA PISTA ---")
    print(
        f"Pista visibile                     : "
        f"{report['track_visible_percent']:.2f}%"
    )
    print(
        f"Errore camera medio assoluto       : "
        f"{report['mean_abs_camera_error']:.3f}"
    )
    print(
        f"Errore camera massimo              : "
        f"{report['max_abs_camera_error']:.3f}"
    )
    print(
        f"Fuori pista stimato                : "
        f"{report['off_track_percent_of_all_frames']:.2f}%"
    )
    print(
        f"Fuori pista quando visibile        : "
        f"{report['off_track_percent_when_track_visible']:.2f}%"
    )
    print(
        f"Massima uscita continua stimata   : "
        f"{report['max_continuous_off_track_s']:.2f} s"
    )
    print(
        f"Recuperi dalla condizione OFF     : "
        f"{report['recovery_events']}"
    )
    print()

    print("--- CONTROLLO ---")
    print(
        f"Correzione sterzo media assoluta  : "
        f"{report['mean_abs_rotation']:.3f}"
    )
    print(
        f"Correzione sterzo massima         : "
        f"{report['max_abs_rotation']:.3f}"
    )
    print(
        f"Cambi direzione sterzo            : "
        f"{report['steering_direction_changes']}"
    )

    print()
    print(f"CSV dettagliato salvato in: {CSV_OUTPUT}")
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    client = None
    sim = None

    rows = []
    metrics = Metrics()

    try:
        print("=" * 70)
        print("SNN ABSOLUTE METRICS EVALUATION")
        print("=" * 70)
        print("Device:", device)

        client, sim = connect()

        print("Connessione a CoppeliaSim OK.")

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

        sim.startSimulation()
        time.sleep(1.0)

        test_start = time.perf_counter()

        while True:
            now = time.perf_counter()

            elapsed = now - test_start

            if (
                TEST_DURATION_SECONDS is not None
                and elapsed >= TEST_DURATION_SECONDS
            ):
                print("\nDurata test raggiunta.")
                break

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

            # Smoothing identico al programma originale.
            history.append(position)

            if len(history) > SMOOTHING_WINDOW:
                history.pop(0)

            smoothed_position = float(
                np.mean(history)
            )

            rotation = (
                -ROTATION_GAIN
                * smoothed_position
            )

            rotation = float(
                np.clip(
                    rotation,
                    -MAX_ROTATION,
                    MAX_ROTATION,
                )
            )

            speed_factor = max(
                0.45,
                1.0
                - 0.45 * abs(smoothed_position),
            )

            forward = (
                FORWARD_SPEED
                * speed_factor
            )

            # Detection indipendente della pista.
            (
                track_visible,
                camera_error,
                red_pixels,
                red_mask,
            ) = detect_red_track(img)

            metrics.update(
                position=smoothed_position,
                confidence=confidence,
                rotation=rotation,
                track_visible=track_visible,
                camera_error=camera_error,
                now=now,
            )

            rows.append(
                {
                    "time_s": elapsed,
                    "predicted_class": CLASS_NAMES[
                        predicted_class
                    ],
                    "position": smoothed_position,
                    "confidence": confidence,
                    "camera_track_visible": int(
                        track_visible
                    ),
                    "camera_error": (
                        ""
                        if camera_error is None
                        else camera_error
                    ),
                    "red_pixels": red_pixels,
                    "rotation": rotation,
                    "forward": forward,
                }
            )

            set_bot_movement(
                sim,
                wheels,
                -forward,
                0,
                rotation,
            )

            # ------------------------------------------------
            # DEBUG DISPLAY
            # ------------------------------------------------

            display = cv2.cvtColor(
                img,
                cv2.COLOR_RGB2BGR,
            )

            if track_visible and camera_error is not None:
                image_center_x = DISPLAY_WIDTH // 2

                track_x = int(
                    image_center_x
                    + camera_error
                    * image_center_x
                )

                cv2.line(
                    display,
                    (image_center_x, 0),
                    (
                        image_center_x,
                        DISPLAY_HEIGHT,
                    ),
                    (255, 255, 255),
                    1,
                )

                cv2.line(
                    display,
                    (track_x, 0),
                    (
                        track_x,
                        DISPLAY_HEIGHT,
                    ),
                    (0, 255, 0),
                    2,
                )

            text1 = (
                f"SNN {CLASS_NAMES[predicted_class]} "
                f"conf={confidence:.2f}"
            )

            text2 = (
                f"SNN pos={smoothed_position:+.2f} "
                f"rot={rotation:+.2f}"
            )

            if track_visible:
                text3 = (
                    f"CAM err={camera_error:+.2f} "
                    f"red={red_pixels}"
                )
            else:
                text3 = "CAM: pista non rilevata"

            cv2.putText(
                display,
                text1,
                (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (255, 255, 255),
                1,
            )

            cv2.putText(
                display,
                text2,
                (10, 43),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (255, 255, 255),
                1,
            )

            cv2.putText(
                display,
                text3,
                (10, 64),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (255, 255, 255),
                1,
            )

            cv2.imshow(
                "SNN EVALUATION",
                display,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("\nTest interrotto con Q.")
                break

            time.sleep(0.02)

        set_bot_movement(
            sim,
            wheels,
            0,
            0,
            0,
        )

        report = metrics.finish()

        write_csv(rows)
        print_report(report)

    except KeyboardInterrupt:
        print("\nTest interrotto.")

    except Exception as exc:
        print("\nERRORE DURANTE LA VALUTAZIONE:")
        print(exc)

        try:
            set_bot_movement(
                sim,
                wheels,
                0,
                0,
                0,
            )
        except Exception:
            pass

        raise

    finally:
        try:
            if sim is not None:
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
