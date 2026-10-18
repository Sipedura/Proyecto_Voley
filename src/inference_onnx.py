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
# Filtro de suavizado utilizado en calculo_tiempos_de_vuelo para suavizar las parábolas sin eliminar los cambios bruscos
from scipy.signal import savgol_filter

def parse_args():
    parser = argparse.ArgumentParser(
        description="Volleyball ball detection and tracking"
    )
    parser.add_argument(
        "--video_path", type=str, required=False, help="Path to input video file", default = "videos/voley8ed1.mp4"
        #"--video_path", type=str, required=True, help="Path to input video file"
    )
    parser.add_argument(
        "--track_length", type=int, default=25, help="Length of the ball track"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        #default=None, #Este es el que va
        help="Directory to save output video and CSV",
        default = "./predicts/"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=False,
        help="Path to ONNX model file (e.g., models/vballNetV1.onnx)",
        default = "vb-models/VballNetV1_150_h288_w512.onnx"
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

"""
    Realiza el escalado del frame para su posterior procesamiento en postprocess_output
"""
def preprocess_frame(frame, input_height=288, input_width=512):
    frame = cv2.resize(frame, (input_width, input_height))
    frame = frame.astype(np.float32) / 255.0
    return frame

"""
    Función que realiza el traqueo de la pelota a partir del modelo cargado
"""
def postprocess_output(
    output, 
    min_confidence=0.55, 
    rel_ratio=0.75, 
    min_area=3, 
    max_area=800,
    search_center=None,          
    kernel_size=(80, 80),
    snap_threshold=8.0           # Umbral en píxeles para "pegar" a la posición anterior si está muy cerca
):
    """
    Postprocesa los heatmaps aplicando el kernel de búsqueda y un umbral 
    de cercanía para estabilizar detecciones estáticas o con micro-ruido.
    """
    # Atributo estático para recordar la última coordenada (cx, cy) válida entre llamadas
    if not hasattr(postprocess_output, "last_valid_pos"):
        postprocess_output.last_valid_pos = None

    results = []
    for frame_idx in range(3):
        heatmap = output[0, frame_idx, :, :].copy()
        h, w = heatmap.shape

        # Aplicar el kernel / ROI centrado si existe una posición previa
        if search_center is not None:
            cx_c, cy_c = search_center
            kw, kh = kernel_size
            
            x_min = max(0, cx_c - kw // 2)
            x_max = min(w, cx_c + kw // 2)
            y_min = max(0, cy_c - kh // 2)
            y_max = min(h, cy_c + kh // 2)

            mask = np.zeros((h, w), dtype=bool)
            mask[y_min:y_max, x_min:x_max] = True
            heatmap[~mask] = 0.0

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(heatmap)

        if max_val < min_confidence:
            results.append((0, 0, 0, 0, 0))
            # Opcional: si se pierde la confianza, podemos limpiar o mantener el estado
            continue

        adaptive_thresh = max_val * rel_ratio
        _, binary = cv2.threshold(heatmap, adaptive_thresh, 1.0, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(
            (binary * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        valid_contours = [c for c in contours if min_area <= cv2.contourArea(c) <= max_area]

        if valid_contours:
            best_contour = max(valid_contours, key=cv2.contourArea)
            M = cv2.moments(best_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # --- APLICAR ESTABILIZACIÓN POR CERCANÍA ---
                if postprocess_output.last_valid_pos is not None:
                    last_cx, last_cy = postprocess_output.last_valid_pos
                    dist = math.hypot(cx - last_cx, cy - last_cy)
                    
                    # Si el nuevo punto está muy cerca del anterior, lo fijamos exactamente al anterior
                    if dist <= snap_threshold:
                        cx, cy = last_cx, last_cy
                
                # Actualizar la última posición válida
                postprocess_output.last_valid_pos = (cx, cy)
                # -------------------------------------------

                x, y, w_box, h_box = cv2.boundingRect(best_contour)
                results.append((1, cx, cy, w_box, h_box))
            else:
                results.append((0, 0, 0, 0, 0))
        else:
            results.append((0, 0, 0, 0, 0))

    return results
"""
Función que levanta la trayectoria de la pelota de voley e informa los tiempos de vuelo
"""
"""
def draw_track(
    frame, track_points, eventos_vuelo = None, vel_evento = None, start_idx=0, line_color=(0, 255, 255), thickness=3
):
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
            cv2.putText(frame, texto, pos_texto, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
        
    if track_points and track_points[-1] is not None:
        cv2.circle(frame, track_points[-1], 6, (0, 0, 255), -1)
        
    return frame
"""
def draw_track(frame, track_points, eventos_vuelo=None, vel_evento=None, start_idx=0, line_color=(0, 255, 255), thickness=3):
    if eventos_vuelo is None:
        eventos_dict = {}
    else:
        eventos_dict = dict(eventos_vuelo)

    if vel_evento is None:
        vel_dict = {}
    else:
        vel_dict = dict(vel_evento)

    # --- ATRIBUTOS PERSISTENTES PARA LA ESQUINA SUPERIOR DERECHA ---
    if not hasattr(draw_track, "ultimo_texto"):
        draw_track.ultimo_texto = None

    # Recorremos el segmento actual para dibujar la línea de la trayectoria
    for i in range(1, len(track_points)):
        pt1 = track_points[i - 1]
        pt2 = track_points[i]
        
        if pt1 is None or pt2 is None:
            continue
        cv2.line(frame, pt1, pt2, line_color, thickness)

        # Frame global actual analizado en el renderizado
        frame_global = start_idx + i

        # Si ocurre un evento en este preciso frame, actualizamos el texto persistente
        if frame_global in eventos_dict:
            draw_track.ultimo_texto = f"Vuelo: {eventos_dict[frame_global]}s | Vel: {vel_dict[frame_global]}px/s"

    # Si hay un texto registrado, lo dibujamos siempre en la esquina superior derecha
    if draw_track.ultimo_texto is not None:
        h_frame, w_frame = frame.shape[:2]
        
        # Configuraciones de fuente y tamaño
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        font_thickness = 2
        
        # Obtenemos el tamaño del texto para calcular la posición X alineada a la derecha
        (text_w, text_h), baseline = cv2.getTextSize(draw_track.ultimo_texto, font, font_scale, font_thickness)
        
        # Margen desde el borde superior derecho (ej: 20 píxeles de margen)
        margin_x = 20
        margin_y = 30
        pos_x = w_frame - text_w - margin_x
        pos_y = margin_y + text_h

        # Opcional: Dibujar un rectángulo semi-transparentes o fondo oscuro detrás para mejorar la legibilidad
        cv2.rectangle(frame, (pos_x - 10, pos_y - text_h - 10), (pos_x + text_w + 10, pos_y + baseline + 5), (0, 0, 0), -1)
        
        # Dibujar el texto en la esquina superior derecha (Color verde o blanco para destacar)
        cv2.putText(frame, draw_track.ultimo_texto, (pos_x, pos_y), font, font_scale, (0, 255, 0), font_thickness)

    if track_points and track_points[-1] is not None:
        cv2.circle(frame, track_points[-1], 6, (0, 0, 255), -1)
        
    return frame

def procesar_y_reparar_trayectoria(
    pred_item, track_points, frame_width, frame_height, input_width, input_height, max_dist=100
):
    """
    Procesado y reparación de la trayectoria, este análsis se realiza en la primera pasada
    Se planteo así debido a que con un análsis de una sola pasada no se podía realizar el cálculo de los tiempos de vuelo a partir
    de las variaciones en la aceleración
    """
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
    track_points, fps, frame_height, min_frames_entre_golpes=20, k_sigma=8.0
):
    """
    Calcula aceleraciones y eventos de impacto utilizando:
      1. Normalización por altura del video (independiente de si es 720p, 1080p o 4K).
      2. Umbral adaptativo estadístico en 2D basado en la media y desviación estándar de los picos.
    """

    indices_validos = [i for i, p in enumerate(track_points) if p is not None and p != (-1, -1)]
    
    if len(indices_validos) < 4:
        return [], [], np.array([]), np.array([]), np.array([]), np.array([])

    inicio = indices_validos[0]
    n_puntos = len(track_points)
    segmento = []
    pts_temp = [(-1, -1) if p is None else p for p in track_points]

    for p in range(n_puntos):
        if pts_temp[p] == (-1, -1):
            if p < n_puntos - 1 and pts_temp[p + 1] != (-1, -1):
                segmento.append(pts_temp[p + 1])
            elif p > 0 and pts_temp[p - 1] != (-1, -1):
                segmento.append(pts_temp[p - 1])
            else:
                idx_cercano = min(indices_validos, key=lambda x: abs(x - p))
                segmento.append(pts_temp[idx_cercano])
        else:
            segmento.append(pts_temp[p])

    pts = np.array(segmento, dtype=float)
    x = pts[:, 0]
    y = pts[:, 1]
    dt = 1.0 / fps

    # Filtro de suavizado utilizado para suavizar las parábolas sin eliminar los cambios bruscos
    # --- INICIO SUAVIZADO SAVITZKY-GOLAY ---
    # window_length: debe ser impar. Un valor entre 5 y 9 es ideal para video a 30/60fps.
    # polyorder: 2 o 3. El orden 2 representa perfectamente una parábola física (gravedad).

    window_length = 7  
    polyorder = 2      
    
    if len(x) >= window_length:
        x_smooth = savgol_filter(x, window_length, polyorder)
        y_smooth = savgol_filter(y, window_length, polyorder)
    else:
        # Fallback si el segmento es muy corto
        x_smooth = x
        y_smooth = y
    # --- FIN SUAVIZADO ---
    
    # Derivadas en píxeles 
    vx = np.gradient(x_smooth, dt)
    #vx = np.gradient(x, dt)
    vy = np.gradient(y_smooth, dt)
    #vy = np.gradient(y, dt)
    velocidad = np.hypot(vx, vy)

    # Aceleración en píxeles
    ax = np.gradient(vx, dt)
    ay = np.gradient(vy, dt)
    acel_total = np.hypot(ax,ay)

    # 1. Variación de la aceleración en 2D (Jerk vectorial)
    diff_ax = np.diff(ax)
    diff_ay = np.diff(ay)
    diff_a2d = np.hypot(diff_ax, diff_ay)

    # Normalización por altura de cuadro: unidades en (alturas_de_pantalla / s^2)
    #diff_a2d_norm = diff_a2d / frame_height
    # Normalización por altura de cuadro: unidades en (alturas_de_pantalla / s^2)
    acel_norm = acel_total / frame_height

    # 2. Umbral Adaptativo Estadístico:
    mediana = np.median(acel_norm)
    std_robusta = np.median(np.abs(acel_norm - mediana)) * 1.4826  # MAD (más robusto a outliers)
    
    umbral_dinamico = mediana + (k_sigma * std_robusta)
    
    # Cota mínima de seguridad para evitar falsos positivos
    # Puede que necesites ajustar este piso (ej. a 2.0) si notas que el módulo 2D capta más ruido
    umbral_dinamico = max(umbral_dinamico, 6.0)

    # 3. Detección de eventos con supresión de picos contiguos
    eventos_vuelo = []
    vel_evento = []
    ultimo_frame_golpe = None

    for i in range(len(acel_norm)):
        if acel_norm[i] > umbral_dinamico:
            frame_real = inicio + i + 1
            
            if ultimo_frame_golpe is None:
                ultimo_frame_golpe = frame_real
                eventos_vuelo.append((frame_real, 0.0))
                vel_evento.append((frame_real, 0.0))
            elif (frame_real - ultimo_frame_golpe) >= min_frames_entre_golpes:
                t_vuelo = (frame_real - ultimo_frame_golpe) * dt
                vel_media = np.mean(velocidad[ultimo_frame_golpe : frame_real + 1])
                eventos_vuelo.append((frame_real, round(t_vuelo, 3)))
                vel_evento.append((frame_real, round(float(vel_media), 1)))
                ultimo_frame_golpe = frame_real

    return eventos_vuelo, vel_evento, vx, vy, ax, ay, acel_total

def main():
    args = parse_args()
    #args.video_path = "videos/voley8ed1.mp4"
    #args.model_path = "vb-models/VballNetV1_150_h288_w512.onnx"
    #args.output_dir = "./predicts/"
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

    # --- VARIABLES PARA EL KERNEL DE BÚSQUEDA DINÁMICO ---
    kernel_center = None        # (x, y) en coordenadas de entrada (512x288)
    kernel_size = (90, 90)      # Tamaño del kernel en pixeles. Ajustar según la velocidad esperada.
    lost_count = 0              # Contador de frames sin detección
    max_lost_frames = 5         # Si se pierde 5 frames, reabrir a búsqueda global
    # ----------------------------------------------------

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

            # --- NUEVA LLAMADA CON KERNEL DINÁMICO ---
            predictions = postprocess_output(
                output,
                search_center=kernel_center,
                kernel_size=kernel_size
            )
            
            # Extraemos el resultado del frame actual (índice 2)
            pred_actual = predictions[2]
            vis, cx, cy, w_box, h_box = pred_actual

            # --- ACTUALIZACIÓN DEL KERNEL ---
            if vis == 1:
                kernel_center = (cx, cy)  # Mover el kernel a la nueva posición
                lost_count = 0            # Reiniciar contador de pérdidas
            else:
                lost_count += 1
                # Si pasa varios frames sin verse, se elimina el kernel para búsqueda global
                if lost_count >= max_lost_frames:
                    kernel_center = None
            
            # Gestiona la racha y rellena track_points
            procesar_y_reparar_trayectoria(
                pred_actual, 
                track_points, 
                frame_width, 
                frame_height, 
                input_width, 
                input_height,
                max_dist=100
            )

        pbar1.update(1)

    pbar1.close()

    eventos_vuelo, vel_evento, vx, vy, ax, ay, acel = calcular_tiempos_de_vuelo(
        track_points, 
        fps, 
        frame_height=frame_height, 
        min_frames_entre_golpes=10, 
        k_sigma=2.5
    )

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
            acel_val = round(float(acel[idx]), 2) if vis != 0 and idx < len(acel) else 0.0 # agrego el valor de aceleración para analizar
            rows.append({
                "Frame": idx,
                "Visibility": vis,
                "X": px,
                "Y": py,
                "vX": vx_val,
                "vY": vy_val,
                "aX": ax_val,
                "aY": ay_val,
                "acel": acel_val
            })
        pd.DataFrame(rows).to_csv(csv_path, index=False)

    # Si se solicitó solo CSV, finalizamos aquí sin renderizar video
    if args.only_csv:
        cap.release()
        return
    """
    # =========================================================================
    # PASADA 2: Renderizado visual con los datos ya corregidos (Sin congelar)
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
    """
    # =========================================================================
    # PASADA 2: Renderizado visual con los datos ya corregidos (Con congelar)
    # =========================================================================
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    pbar2 = tqdm(total=total_frames, desc="Pasada 2: Renderizado", unit="frame")
    frame_index = 0

    # Creamos un conjunto (set) con los números de frame donde ocurren eventos para búsqueda rápida O(1)
    frames_eventos = {ev[0] for ev in eventos_vuelo} if eventos_vuelo else set()
    
    # Cantidad de frames extras para medio segundo de pausa (ej: a 30 fps son 15 frames)
    frames_congelado = int(fps / 2)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_index >= len(track_points):
            break

        # Extraer el tramo de la estela hasta el frame actual
        start_idx = max(0, frame_index - args.track_length + 1)
        pts_slice = track_points[start_idx : frame_index + 1]

        if args.visualize or out_writer is not None:
            vis_frame = draw_track(frame, pts_slice, eventos_vuelo = eventos_vuelo, vel_evento = vel_evento, start_idx = start_idx)
            
            # Validar si el frame actual corresponde a un evento de vuelo
            es_evento = frame_index in frames_eventos
            
            if args.visualize:
                cv2.namedWindow("Ball Tracking", cv2.WINDOW_NORMAL)
                cv2.imshow("Ball Tracking", vis_frame)
                # Si es evento, pausamos medio segundo (500 ms) en pantalla; sino, velocidad normal del video
                delay = 500 if es_evento else max(1, int(1000 / fps))
                if cv2.waitKey(delay) & 0xFF == ord("q"):
                    break
                    
            if out_writer is not None:
                # Escribimos el frame del evento normalmente
                out_writer.write(vis_frame)
                
                # Si es un evento, duplicamos el frame en el archivo de video para "congelarlo"
                if es_evento:
                    for _ in range(frames_congelado):
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