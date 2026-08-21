import argparse
import cv2
import numpy as np
import pandas as pd
import onnx
import onnxruntime as ort
from collections import deque
import os
import time
from tqdm import tqdm
import math
from scipy.interpolate import CubicSpline

def parse_args():
    parser = argparse.ArgumentParser(
        description="Volleyball ball detection and tracking"
    )
    parser.add_argument(
        "--video_path", type=str, required=True, help="Path to input video file"
    )
    parser.add_argument(
        "--track_length", type=int, default=25, help="Length of the ball track"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save output video and CSV",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to ONNX model file (e.g., models/vballNetV1.onnx)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        default=False,
        help="Enable visualization on display using cv2",
    )
    parser.add_argument(
        "--only_csv",
        action="store_true",
        default=False,
        help="Save only CSV, skip video output",
    )
    return parser.parse_args()


def load_model(model_path):
    if not os.path.exists(model_path):
        raise ValueError(f"Model file not found: {model_path}")
    if not model_path.endswith(".onnx"):
        raise ValueError("Model file must be in ONNX format (.onnx)")

    model = onnx.load(model_path)
    onnx.checker.check_model(model)
    session = ort.InferenceSession(model_path)
    return session


def initialize_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, frame_width, frame_height, fps, total_frames


def setup_output_writer(
    video_basename, output_dir, frame_width, frame_height, fps, only_csv
):
    if output_dir is None or only_csv:
        return None, None

    output_path = os.path.join(output_dir, f"{video_basename}_predict.mp4")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    out_writer = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height)
    )
    return out_writer, output_path


def setup_csv_file(video_basename, output_dir):
    if output_dir is None:
        return None
    csv_path = os.path.join(output_dir, f"{video_basename}_predict_ball.csv")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    pd.DataFrame(columns=["Frame", "Visibility", "X", "Y", "W", "H"]).to_csv(
        csv_path, index=False
    )
    return csv_path


def append_to_csv(result, csv_path):
    if csv_path is None:
        return
    pd.DataFrame([result]).to_csv(csv_path, mode="a", header=False, index=False)


def preprocess_frame(frame, input_height=288, input_width=512):
    frame = cv2.resize(frame, (input_width, input_height))
    frame = frame.astype(np.float32) / 255.0
    return frame


def postprocess_output(output, threshold=0.5, input_height=288, input_width=512):
    results = []
    for frame_idx in range(3):
        heatmap = output[0, frame_idx, :, :]
        _, binary = cv2.threshold(heatmap, threshold, 1.0, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            (binary * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                x, y, w, h = cv2.boundingRect(largest_contour)
                results.append((1, cx, cy, w, h))
            else:
                results.append((0, 0, 0, 0, 0))
        else:
            results.append((0, 0, 0, 0, 0))
    return results


def draw_track(frame, track_points, eventos_vuelo = None, vel_evento = None, start_idx=0, line_color=(0, 255, 255), thickness=3):
    if eventos_vuelo is None:
        eventos_vuelo = {}
    else:
        eventos_dict = dict(eventos_vuelo)

    if vel_evento is None:
        vel_evento = {}
    else:
        vel_dict = dict(vel_evento)

    for i in range(1, len(track_points)):
        pt1 = track_points[i - 1]
        pt2 = track_points[i]
        
        if pt1 is None or pt2 is None:
            continue
        cv2.line(frame, pt1, pt2, line_color, thickness)

        # Como el graficador funciona levantando de a 25 frames, se le pasa el frame_global a eventos dict para graficar el evento
        frame_global = start_idx + i

        if frame_global in eventos_dict:
            #print("Se grafico un evento")
            texto = f"{eventos_dict[frame_global]}s, {vel_dict[frame_global]}px/s"
            pos_texto = (pt2[0] + 10, pt2[1] - 10)
            cv2.putText(frame, texto, pos_texto, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
    if track_points and track_points[-1] is not None:
        cv2.circle(frame, track_points[-1], 6, (0, 0, 255), -1)
        
    return frame


def procesar_y_reparar_trayectoria(pred_item, track_points, frame_width, frame_height, input_width, input_height, max_dist=100):
    if not hasattr(procesar_y_reparar_trayectoria, "racha"):
        procesar_y_reparar_trayectoria.racha = 0
        procesar_y_reparar_trayectoria.buffer = []

    visibility, x, y, w, h = pred_item

    # 1. No visible
    if visibility == 0:
        procesar_y_reparar_trayectoria.racha = 0
        procesar_y_reparar_trayectoria.buffer.clear()
        track_points.append(None)
        return pred_item

    # 2. Coordenadas reales
    x_orig = x * frame_width / input_width
    y_orig = y * frame_height / input_height
    punto_actual = (int(x_orig), int(y_orig))

    # 3. Evaluación de racha y distancias
    if procesar_y_reparar_trayectoria.racha >= 3:
        ultimo_confirmado = track_points[-1]
        dist = math.hypot(punto_actual[0] - ultimo_confirmado[0], punto_actual[1] - ultimo_confirmado[1])
        
        if dist <= max_dist:
            procesar_y_reparar_trayectoria.racha += 1
            track_points.append(punto_actual)
            return pred_item
        else:
            procesar_y_reparar_trayectoria.racha = 1
            procesar_y_reparar_trayectoria.buffer = [punto_actual]
            track_points.append(None)
            return pred_item

    elif procesar_y_reparar_trayectoria.racha == 0:
        procesar_y_reparar_trayectoria.racha = 1
        procesar_y_reparar_trayectoria.buffer = [punto_actual]
        track_points.append(None)
        return pred_item

    else:  # racha == 1 o 2
        ultimo_candidato = procesar_y_reparar_trayectoria.buffer[-1]
        dist = math.hypot(punto_actual[0] - ultimo_candidato[0], punto_actual[1] - ultimo_candidato[1])

        if dist <= max_dist:
            procesar_y_reparar_trayectoria.racha += 1
            procesar_y_reparar_trayectoria.buffer.append(punto_actual)
            track_points.append(None)
        else:
            procesar_y_reparar_trayectoria.racha = 1
            procesar_y_reparar_trayectoria.buffer = [punto_actual]
            track_points.append(None)
            return pred_item

    # 4. Confirmación e interpolación lineal entre huecos
    if procesar_y_reparar_trayectoria.racha == 3:
        p1, p2, p3 = procesar_y_reparar_trayectoria.buffer
        
        track_points[-3] = p1
        track_points[-2] = p2
        track_points[-1] = p3
        procesar_y_reparar_trayectoria.buffer.clear()

        indices_validos = [i for i, p in enumerate(track_points) if p is not None]

        for idx in range(len(indices_validos) - 1):
            i_A = indices_validos[idx]
            i_B = indices_validos[idx + 1]

            if i_B > i_A + 1:
                p_A = track_points[i_A]
                p_B = track_points[i_B]
                num_pasos = i_B - i_A

                for k in range(i_A + 1, i_B):
                    t = (k - i_A) / num_pasos
                    x_interp = int(round(p_A[0] + t * (p_B[0] - p_A[0])))
                    y_interp = int(round(p_A[1] + t * (p_B[1] - p_A[1])))
                    track_points[k] = (x_interp, y_interp)

    return pred_item

def calcular_tiempos_de_vuelo(
        track_points, fps, umbral_acel=8000, min_frames_entre_golpes=10
):
    """
    Calcula aceleraciones y tiempos de vuelo directamente sobre track_points ya reparado,
    omitiendo cualquier re-interpolación redundante.
    """
    indices_validos = [i for i, p in enumerate(track_points) if p is not None and p != (-1, -1)]
    
    # 1. Si no hay suficientes puntos, retornar 5 elementos vacíos
    if len(indices_validos) < 4:
        return [], np.array([]), np.array([]), np.array([]), np.array([])

    inicio = indices_validos[0]

    # 2. Rellenar huecos (None o (-1, -1)) usando los vecinos válidos más cercanos
    n_puntos = len(track_points)
    segmento = []

    # Normalizar reemplazando None por (-1, -1) para procesar
    pts_temp = [(-1, -1) if p is None else p for p in track_points]

    # Reemplazo por interpolación lineal/vecinos para todo el arreglo
    for p in range(n_puntos):
        if pts_temp[p] == (-1, -1):
            if p < n_puntos - 1 and pts_temp[p + 1] != (-1, -1):
                segmento.append(pts_temp[p + 1])
            elif p > 0 and pts_temp[p - 1] != (-1, -1):
                segmento.append(pts_temp[p - 1])
            else:
                # Si está rodeado de invalidos, toma el punto válido más cercano
                idx_cercano = min(indices_validos, key=lambda x: abs(x - p))
                segmento.append(pts_temp[idx_cercano])
        else:
            segmento.append(pts_temp[p])

    pts = np.array(segmento, dtype=float)
    x = pts[:, 0]
    y = pts[:, 1]
    dt = 1.0 / fps

    # 3. Derivadas directas sobre la trayectoria ya suavizada
    vx = np.gradient(x, dt)
    vy = np.gradient(y, dt)
    velocidad = np.hypot(vx,vy)
    
    ax = np.gradient(vx, dt)
    ay = np.gradient(vy, dt)
    aceleracion = np.hypot(ax, ay)

    # 4. Detección de picos de impacto
    eventos_vuelo = []
    vel_evento = []
    ultimo_frame_golpe = None
    n_puntos = len(aceleracion)

    for i in range(1, n_puntos - 1):
        es_pico = (
            abs(ay[i] - ay[i-1]) > umbral_acel
        )

        if es_pico:
            frame_real = inicio + i  # Mapear al índice global del video
            
            if ultimo_frame_golpe is None:
                ultimo_frame_golpe = frame_real
                eventos_vuelo.append((frame_real, 0.0))
                vel_evento.append((frame_real,0.0))
                #print("Se detectó un evento")
            elif (frame_real - ultimo_frame_golpe) >= min_frames_entre_golpes:
                t_vuelo = (frame_real - ultimo_frame_golpe) * dt
                eventos_vuelo.append((frame_real, round(t_vuelo, 3)))
                vel_evento.append((frame_real, round(np.mean(velocidad[ultimo_frame_golpe : frame_real + 1]),1)))
                ultimo_frame_golpe = frame_real
                #print("Se detectó un evento")
        #else:
           #print("No se detectó un evento")

    return eventos_vuelo,vel_evento, vx, vy, ax, ay

def main():
    args = parse_args()
    input_width, input_height = 512, 288

    model = load_model(args.model_path)
    cap, frame_width, frame_height, fps, total_frames = initialize_video(
        args.video_path
    )

    video_basename = os.path.splitext(os.path.basename(args.video_path))[0]
    out_writer, _ = setup_output_writer(
        video_basename, args.output_dir, frame_width, frame_height, fps, args.only_csv
    )
    csv_path = setup_csv_file(video_basename, args.output_dir)

    # Buffer de entrada para el modelo (3 frames para inferencia)
    frame_buffer = deque(maxlen=3)
    
    # Historial completo de la trayectoria
    track_points = []

    # =========================================================================
    # PASADA 1: Inferencia e interpolación completa de la trayectoria
    # =========================================================================
    pbar1 = tqdm(total=total_frames, desc="Pasada 1: Detección e Interpolación", unit="frame")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed_frame = preprocess_frame(frame_rgb, input_height, input_width)
        frame_buffer.append(processed_frame)

        if len(frame_buffer) < 3:
            for _ in range(2):
                frame_buffer.append(processed_frame)

        if len(frame_buffer) == 3:
            input_tensor = np.concatenate(frame_buffer, axis=2)
            input_tensor = np.expand_dims(input_tensor, axis=0)
            input_tensor = np.transpose(input_tensor, (0, 3, 1, 2)).astype(np.float32)

            input_name = model.get_inputs()[0].name
            output = model.run(None, {input_name: input_tensor})[0]

            predictions = postprocess_output(
                output, input_height=input_height, input_width=input_width
            )
            
            # Gestiona la racha y rellena track_points
            procesar_y_reparar_trayectoria(
                predictions[2], 
                track_points, 
                frame_width, 
                frame_height, 
                input_width, 
                input_height,
                max_dist=100
            )

        pbar1.update(1)

    pbar1.close()

    eventos_vuelo,vel_evento,vx,vy,ax,ay = calcular_tiempos_de_vuelo(track_points,fps)

    # Guardar resultados en el archivo CSV
    if csv_path is not None:
        rows = []
        for idx, pt in enumerate(track_points):
            vis = 1 if pt is not None else 0
            px, py = pt if pt is not None else (-1, -1)
            vx_val = round(float(vx[idx]), 2) if vis != 0 and idx < len(vx) else 0.0
            vy_val = round(float(vy[idx]), 2) if vis != 0 and idx < len(vx) else 0.0
            ax_val = round(float(ax[idx]), 2) if vis != 0 and idx < len(ax) else 0.0
            ay_val = round(float(ay[idx]), 2) if vis != 0 and idx < len(ax) else 0.0
            rows.append({
                "Frame": idx,
                "Visibility": vis,
                "X": px,
                "Y": py,
                "vX": vx_val,
                "vY": vy_val,
                "aX": ax_val,
                "aY": ay_val
            })
        pd.DataFrame(rows).to_csv(csv_path, index=False)

    # Si se solicitó solo CSV, finalizamos aquí sin renderizar video
    if args.only_csv:
        cap.release()
        return

    # =========================================================================
    # PASADA 2: Renderizado visual con los datos ya corregidos
    # =========================================================================
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    pbar2 = tqdm(total=total_frames, desc="Pasada 2: Renderizado", unit="frame")
    frame_index = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_index >= len(track_points):
            break

        # Extraer el tramo de la estela hasta el frame actual
        start_idx = max(0, frame_index - args.track_length + 1)
        pts_slice = track_points[start_idx : frame_index + 1]

        if args.visualize or out_writer is not None:
            vis_frame = draw_track(frame, pts_slice, eventos_vuelo = eventos_vuelo, vel_evento = vel_evento, start_idx = start_idx)
            
            if args.visualize:
                cv2.namedWindow("Ball Tracking", cv2.WINDOW_NORMAL)
                cv2.imshow("Ball Tracking", vis_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                    
            if out_writer is not None:
                out_writer.write(vis_frame)

        pbar2.update(1)
        frame_index += 1

    pbar2.close()
    cap.release()
    if out_writer is not None:
        out_writer.release()
    if args.visualize:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()