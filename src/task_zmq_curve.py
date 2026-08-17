import cv2
import numpy as np
import time
import traceback
from pathlib import Path

from coppeliasim_zmqremoteapi_client import RemoteAPIClient


# ============================================================
# CoppeliaSim 4.10 - ZeroMQ Remote API
#
# IMPORTANTE:
# - NON usa sim.py
# - NON usa simConst.py
# - NON usa remoteApi.dll
# - NON usa sim.handleVisionSensor()
#
# Struttura:
# C:\Users\Pc\Desktop\progetto\
#     task_zmq.py
#     debug\
# ============================================================


# ============================================================
# CARTELLA DEBUG
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
DEBUG_DIR = PROJECT_DIR / "debug"
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

DEBUG_CAMERA = DEBUG_DIR / "debug_camera.png"
DEBUG_MASK = DEBUG_DIR / "debug_mask.png"


# ============================================================
# CONFIGURAZIONE
# ============================================================

WIDTH = 256
HEIGHT = 256

# Percorso ROSSO in HSV.
# Usiamo entrambe le estremità del rosso:
# H = 0..15 e H = 165..179.
RED_LOW_1 = np.array([0, 50, 30])
RED_HIGH_1 = np.array([15, 255, 255])

RED_LOW_2 = np.array([165, 50, 30])
RED_HIGH_2 = np.array([179, 255, 255])

MIN_AREA = 20


# ============================================================
# CONNESSIONE
# ============================================================

def connect():
    client = RemoteAPIClient()
    sim = client.require("sim")
    return client, sim


# ============================================================
# RUOTE
# ============================================================

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
                f"Non trovo '{name}' nella Scene Hierarchy."
            ) from exc

    return wheels


def set_bot_movement(
    sim,
    wheels,
    forward,
    lateral,
    rotation
):

    velocities = [
        -forward - lateral - rotation,
        -forward + lateral - rotation,
        -forward - lateral + rotation,
        -forward + lateral + rotation,
    ]

    for joint, velocity in zip(wheels, velocities):
        sim.setJointTargetVelocity(
            joint,
            velocity
        )


# ============================================================
# CAMERA
# ============================================================

def get_camera_image(sim, camera):

    # IMPORTANTE:
    # NON chiamare sim.handleVisionSensor(camera).
    # La tua camera NON è configurata per explicit handling.
    #
    # CoppeliaSim aggiorna automaticamente il vision sensor
    # durante la simulazione.

    image, resolution = sim.getVisionSensorImg(
        camera
    )

    if image is None:
        return None

    width, height = resolution

    data = np.frombuffer(
        image,
        dtype=np.uint8
    )

    expected = width * height * 3

    if data.size != expected:
        raise RuntimeError(
            f"Immagine camera non valida: "
            f"{data.size} valori, attesi {expected}."
        )

    # CoppeliaSim restituisce RGB.
    img = data.reshape(
        (height, width, 3)
    )

    # Correzione verticale.
    img = cv2.flip(
        img,
        0
    )

    return img


# ============================================================
# ROSSO
# ============================================================

def red_mask(img):

    hsv = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2HSV
    )

    mask1 = cv2.inRange(
        hsv,
        RED_LOW_1,
        RED_HIGH_1
    )

    mask2 = cv2.inRange(
        hsv,
        RED_LOW_2,
        RED_HIGH_2
    )

    mask = cv2.bitwise_or(
        mask1,
        mask2
    )

    # Pulizia del rumore.
    kernel = np.ones(
        (3, 3),
        np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    return mask


# ============================================================
# PERCORSO
# ============================================================

def find_path(mask):

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    if not contours:
        return None, None, None, 0

    contours = [
        c for c in contours
        if cv2.contourArea(c) >= MIN_AREA
    ]

    if not contours:
        return None, None, None, 0

    biggest = max(
        contours,
        key=cv2.contourArea
    )

    area = cv2.contourArea(
        biggest
    )

    x, y, w, h = cv2.boundingRect(
        biggest
    )

    cx = x + w // 2
    cy = y + h // 2

    return biggest, cx, cy, area


# ============================================================
# DEBUG
# ============================================================

def save_debug(img, mask):

    camera_path = str(DEBUG_CAMERA)
    mask_path = str(DEBUG_MASK)

    try:

        # Salviamo l'immagine camera come PNG.
        camera_bgr = cv2.cvtColor(
            img,
            cv2.COLOR_RGB2BGR
        )

        camera_ok = cv2.imwrite(
            camera_path,
            camera_bgr
        )

        mask_ok = cv2.imwrite(
            mask_path,
            mask
        )

        print(
            f"DEBUG salvato -> "
            f"camera={camera_ok}, mask={mask_ok}"
        )
        print(
            f"  {camera_path}"
        )
        print(
            f"  {mask_path}"
        )

    except Exception:
        print("Errore durante il salvataggio DEBUG:")
        traceback.print_exc()


# ============================================================
# ROBOT
# ============================================================

def run_robot(sim):

    # --------------------------------------------------------
    # CAMERA
    # --------------------------------------------------------

    try:
        camera = sim.getObject(
            "/camera"
        )
    except Exception as exc:
        raise RuntimeError(
            "Non trovo '/camera' nella Scene Hierarchy."
        ) from exc

    # --------------------------------------------------------
    # RUOTE
    # --------------------------------------------------------

    wheels = get_wheels(
        sim
    )

    print()
    print("=" * 55)
    print("CAMERA E RUOTE COLLEGATE")
    print("=" * 55)
    print(
        f"Debug: {DEBUG_DIR}"
    )
    print()

    frame = 0

    while True:

        frame += 1

        # ----------------------------------------------------
        # CAMERA
        # ----------------------------------------------------

        img = get_camera_image(
            sim,
            camera
        )

        if img is None:

            set_bot_movement(
                sim,
                wheels,
                0,
                0,
                0
            )

            time.sleep(0.02)
            continue

        img = cv2.resize(
            img,
            (WIDTH, HEIGHT)
        )

        # ----------------------------------------------------
        # ROSSO
        # ----------------------------------------------------

        mask = red_mask(
            img
        )

        red_pixels = cv2.countNonZero(
            mask
        )

        # ----------------------------------------------------
        # DEBUG
        # ----------------------------------------------------

        if frame == 1 or frame % 30 == 0:
            save_debug(
                img,
                mask
            )

        # ----------------------------------------------------
        # PERCORSO
        # ----------------------------------------------------

        contour, cx, cy, area = find_path(
            mask
        )

        display = cv2.cvtColor(
            img,
            cv2.COLOR_RGB2BGR
        )

        # ----------------------------------------------------
        # NON TROVATO
        # ----------------------------------------------------

        if contour is None:

            # Sicurezza.
            set_bot_movement(
                sim,
                wheels,
                0,
                0,
                0
            )

            if frame % 10 == 0:
                print(
                    f"Percorso NON trovato | "
                    f"pixel rossi={red_pixels}"
                )

            cv2.putText(
                display,
                f"ROSSO: {red_pixels}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

        # ----------------------------------------------------
        # TROVATO
        # ----------------------------------------------------

        else:

            cv2.drawContours(
                display,
                [contour],
                -1,
                (255, 0, 255),
                3
            )

            cv2.circle(
                display,
                (cx, cy),
                8,
                (0, 255, 0),
                -1
            )

            # Punto centrale in basso.
            ref_x = WIDTH // 2
            ref_y = HEIGHT - 1

            cv2.circle(
                display,
                (ref_x, ref_y),
                8,
                (255, 0, 0),
                -1
            )

            # ------------------------------------------------
            # ERRORE
            # ------------------------------------------------

            denominator = ref_y - cy

            if denominator == 0:
                w_rot = 0.0
            else:
                w_rot = np.arctan(
                    (ref_x - cx) / denominator
                )

            angle = abs(
                np.degrees(
                    w_rot
                )
            )

            # ------------------------------------------------
            # VELOCITÀ + STERZATA
            # ------------------------------------------------
            # Il robot va volutamente PIÙ LENTO sul rettilineo
            # rispetto alla versione precedente, così ha più
            # tempo per correggere la traiettoria.
            #
            # In curva invece aumentiamo la velocità di ROTAZIONE
            # (m), cioè quanto rapidamente sterza.
            #
            # La velocità in avanti (n) viene comunque ridotta
            # nelle curve molto strette: andare più veloce in
            # avanti durante una curva stretta farebbe uscire
            # il robot dal percorso.

            if angle <= 10:
                # RETTILINEO
                n = 4.5
                m = 0.8

            elif angle <= 25:
                # CURVA LEGGERA
                n = 4.2
                m = 1.5

            elif angle <= 40:
                # CURVA MEDIA
                n = 3.8
                m = 2.3

            elif angle <= 55:
                # CURVA FORTE
                n = 3.2
                m = 3.2

            elif angle <= 70:
                # CURVA MOLTO FORTE
                n = 2.6
                m = 4.0

            else:
                # CURVA ESTREMA
                n = 2.0
                m = 4.5

            rotation = m * w_rot

            # Limite di sicurezza della rotazione.
            rotation = float(
                np.clip(
                    rotation,
                    -4.5,
                    4.5
                )
            )

            print(
                f"PERCORSO TROVATO | "
                f"area={area:.0f} | "
                f"cx={cx} cy={cy} | "
                f"errore={angle:.1f} | "
                f"vel={n:.1f} | "
                f"rot={rotation:.2f}"
            )

            # ------------------------------------------------
            # MOVIMENTO
            # ------------------------------------------------

            set_bot_movement(
                sim,
                wheels,
                -n,
                0,
                rotation
            )

            cv2.putText(
                display,
                f"rosso={red_pixels}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

            cv2.putText(
                display,
                f"angolo={angle:.1f}",
                (10, 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

        # ----------------------------------------------------
        # OPENCV
        # ----------------------------------------------------

        try:

            cv2.imshow(
                "CAMERA",
                display
            )

            cv2.imshow(
                "PERCORSO ROSSO",
                mask
            )

            if (
                cv2.waitKey(1) & 0xFF
                == ord("q")
            ):
                break

        except cv2.error:
            # Se OpenCV non può creare finestre,
            # il robot continua comunque.
            pass

    # Stop.
    set_bot_movement(
        sim,
        wheels,
        0,
        0,
        0
    )


# ============================================================
# MAIN
# ============================================================

def main():

    client = None
    sim = None

    try:

        print()
        print("=" * 55)
        print(
            " ROBOT PATH FOLLOWING - COPPELIASIM 4.10"
        )
        print(
            " ZERO MQ REMOTE API"
        )
        print("=" * 55)
        print()

        print(
            "Connessione a CoppeliaSim..."
        )

        client, sim = connect()

        print(
            "Connessione OK."
        )

        print(
            "Avvio simulazione..."
        )

        sim.startSimulation()

        time.sleep(
            1
        )

        print(
            "Simulazione avviata."
        )

        run_robot(
            sim
        )

    except KeyboardInterrupt:

        print(
            "\nProgramma interrotto."
        )

    except Exception:

        print(
            "\nERRORE:"
        )
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

        print(
            "\nRobot fermato."
        )


if __name__ == "__main__":
    main()
