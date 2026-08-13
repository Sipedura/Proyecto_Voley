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
# import math # Para la función de interpolación original, pero no se usa en la versión actual
from scipy.interpolate import CubicSpline
import math


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


"""
Definición original de draw_track, debería de modificarlo para mantenerlo
def draw_track(
    frame, track_points, current_color=(0, 0, 255), history_color=(255, 0, 0)
):
    for point in list(track_points)[:-1]:
        if point is not None:
            cv2.circle(frame, point, 5, history_color, -1)
    if track_points and track_points[-1] is not None:
        cv2.circle(frame, track_points[-1], 5, current_color, -1)
    return frame
"""
## Devolvio Gemini, hay q modificarlo
def draw_track(
    frame, track_points, line_color=(0, 255, 255), thickness=3
):
    # Dibuja segmentos continuos uniendo puntos consecutivos
    for i in range(1, len(track_points)):
        pt1 = track_points[i - 1]
        pt2 = track_points[i]
            
        # Omitir si alguno de los dos puntos no fue detectado en ese fotograma
        if pt1 is None or pt2 is None: # Umbral para ignorar puntos muy lejanos
            continue
        else: #borrar estas 2 lineas y descomentar abajo
            cv2.line(frame, pt1, pt2, line_color, thickness)
        """
        dist = np.sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2)

        if dist <= 100:  # Umbral de distancia para dibujar la línea
            cv2.line(frame, pt1, pt2, line_color, thickness)
        else:
            continue
        """
    
        
    # Dibujar un punto destacado en la posición actual de la pelota
    if track_points and track_points[-1] is not None:
        cv2.circle(frame, track_points[-1], 6, (0, 0, 255), -1)
        
    return frame
"""
def interpolar_prediccion(pred_item, track_points, frame_width, frame_height, input_width, input_height, ):
    
    #Aplica interpolación cúbica si:
    #- Hay al menos 4 puntos en 'track_points'
    #- La visibilidad es 0 O la distancia euclidiana supera 'max_dist' (100 px).
    
    #Si hay menos de 4 puntos o no se cumplen las condiciones, devuelve la predicción original sin modificar.
    
    # Si hay menos de 4 puntos en el historial, se ignora la interpolación
    if len(track_points) < 4:
        return pred_item

    visibility, x, y, w, h = pred_item
    
    # Proyección a la resolución del frame para medir la distancia euclidiana real
    x_orig = x * frame_width / input_width
    y_orig = y * frame_height / input_height

    requiere_interpolacion = False

    # 1. Comprobar condiciones
    if visibility != 0:
        return(visibility, x, y, w, h)  # No se requiere interpolación si es visible
    else:
        requiere_interpolacion = True

    # 2. Aplicar interpolación cúbica con CubicSpline
    if requiere_interpolacion:
        # Extraer exactamente los últimos 4 puntos registrados
        puntos = list(track_points)[-4:]
        x_vals = np.array([p[0] for p in puntos])
        y_vals = np.array([p[1] for p in puntos])
        
        # Puntos temporales t = [0, 1, 2, 3]
        t_pasado = np.arange(4)
        
        # Ajuste de trazador cúbico para X e Y
        cs_x = CubicSpline(t_pasado, x_vals, bc_type='natural')
        cs_y = CubicSpline(t_pasado, y_vals, bc_type='natural')
        
        # Extrapolación al frame actual (t = 4)
        x_interp_orig = float(cs_x(4))
        y_interp_orig = float(cs_y(4))

        # Reescalar las coordenadas de vuelta al tamaño 'input'
        x_interp = x_interp_orig * input_width / frame_width
        y_interp = y_interp_orig * input_height / frame_height

        # Retorna el punto corregido forzando visibilidad = 1
        return [1, x_interp, y_interp, w, h]

    # Si no requirió interpolación, devuelve el elemento intacto
    return pred_item
"""

import math
import numpy as np
from scipy.interpolate import CubicSpline

def procesar_y_reparar_trayectoria(pred_item, track_points, frame_width, frame_height, input_width, input_height, max_dist=100):
    """
    Filtra falsos positivos requiriendo 3 frames consecutivos visibles Y con distancia 
    consecutiva <= max_dist (100 px). 
    
    Una vez confirmada la racha (dist(P1,P2) <= 100 y dist(P2,P3) <= 100), repara 
    retroactivamente los huecos intermedios usando CubicSpline.
    """
    # Variables de estado persistentes entre llamadas
    if not hasattr(procesar_y_reparar_trayectoria, "racha"):
        procesar_y_reparar_trayectoria.racha = 0
        procesar_y_reparar_trayectoria.buffer = []

    visibility, x, y, w, h = pred_item

    # 1. Si no es visible, reiniciar racha y guardar None
    if visibility == 0:
        procesar_y_reparar_trayectoria.racha = 0
        procesar_y_reparar_trayectoria.buffer.clear()
        track_points.append(None)
        return pred_item

    # 2. Convertir coordenadas a píxeles del frame
    x_orig = x * frame_width / input_width
    y_orig = y * frame_height / input_height
    punto_actual = (int(x_orig), int(y_orig))

    # 3. Control de racha y validación de distancias
    if procesar_y_reparar_trayectoria.racha >= 3:
        # Trayectoria ya confirmada: validar respecto al último punto confirmado
        ultimo_confirmado = track_points[-1]
        dist = math.hypot(punto_actual[0] - ultimo_confirmado[0], punto_actual[1] - ultimo_confirmado[1])
        
        if dist <= max_dist:
            procesar_y_reparar_trayectoria.racha += 1
            track_points.append(punto_actual)
            return pred_item
        else:
            # Salto anómalo: rompe la racha y el punto pasa a ser candidato P1
            procesar_y_reparar_trayectoria.racha = 1
            procesar_y_reparar_trayectoria.buffer = [punto_actual]
            track_points.append(None)
            return pred_item

    elif procesar_y_reparar_trayectoria.racha == 0:
        # Primer candidato de la racha (P1)
        procesar_y_reparar_trayectoria.racha = 1
        procesar_y_reparar_trayectoria.buffer = [punto_actual]
        track_points.append(None)
        return pred_item

    else: # racha == 1 (buscando P2) o racha == 2 (buscando P3)
        # Validar distancia <= 100 px contra el último candidato del buffer
        ultimo_candidato = procesar_y_reparar_trayectoria.buffer[-1]
        dist = math.hypot(punto_actual[0] - ultimo_candidato[0], punto_actual[1] - ultimo_candidato[1])

        if dist <= max_dist:
            procesar_y_reparar_trayectoria.racha += 1
            procesar_y_reparar_trayectoria.buffer.append(punto_actual)
            track_points.append(None)  # Provisional hasta validar los 3 puntos
        else:
            # Distancia > 100 px: el punto actual pasa a ser el nuevo candidato P1
            procesar_y_reparar_trayectoria.racha = 1
            procesar_y_reparar_trayectoria.buffer = [punto_actual]
            track_points.append(None)
            return pred_item

    # 4. Confirmación e interpolación al alcanzar exactamente racha == 3
    if procesar_y_reparar_trayectoria.racha == 3:
        p1, p2, p3 = procesar_y_reparar_trayectoria.buffer
        
        # Confirmar P1, P2 y P3 en el deque
        track_points[-3] = p1
        track_points[-2] = p2
        track_points[-1] = p3
        procesar_y_reparar_trayectoria.buffer.clear()

        # Interpolación por puente en los huecos 'None' entre tramos confirmados
        puntos_conocidos = [(i, p) for i, p in enumerate(track_points) if p is not None]
        indices_none = [i for i, p in enumerate(track_points) if p is None]

        if indices_none and len(puntos_conocidos) >= 2:
            t_conocidos = [p[0] for p in puntos_conocidos]
            x_conocidos = [p[1][0] for p in puntos_conocidos]
            y_conocidos = [p[1][1] for p in puntos_conocidos]

            if len(puntos_conocidos) >= 4:
                # CubicSpline cuando hay al menos 4 puntos ancla en total
                cs_x = CubicSpline(t_conocidos, x_conocidos, bc_type='natural')
                cs_y = CubicSpline(t_conocidos, y_conocidos, bc_type='natural')
                
                for i in indices_none:
                    if t_conocidos[0] < i < t_conocidos[-1]:
                        track_points[i] = (int(cs_x(i)), int(cs_y(i)))
            else:
                # Interpolación lineal fallback si hay entre 2 y 3 puntos ancla
                for i in indices_none:
                    if t_conocidos[0] < i < t_conocidos[-1]:
                        x_interp = int(np.interp(i, t_conocidos, x_conocidos))
                        y_interp = int(np.interp(i, t_conocidos, y_conocidos))
                        track_points[i] = (x_interp, y_interp)

    return pred_item
"""
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

    frame_buffer = deque(maxlen=3)
    track_points = deque(maxlen=args.track_length)
    # Cambio Gemini
    #track_points = []
    frame_index = 0

    pbar = tqdm(total=total_frames, desc="Processing video", unit="frame")

    while cap.isOpened():
        start_time = time.time()

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
            
            predictions[2] = procesar_y_reparar_trayectoria(
                predictions[2], 
                track_points, 
                frame_width, 
                frame_height, 
                input_width, 
                input_height,
            )

            visibility, x, y, w, h = predictions[2]
            if visibility == 0:
                x_orig, y_orig, w_orig, h_orig = -1, -1, -1, -1
                # Comentado por el cambio implementado por gemini
                if len(track_points) > 0:
                    track_points.popleft()
                #track_points.append(None)
            else:
                x_orig = x * frame_width / input_width
                y_orig = y * frame_height / input_height
                w_orig = w * frame_width / input_width
                h_orig = h * frame_height / input_height
                track_points.append((int(x_orig), int(y_orig)))

            result = {
                "Frame": frame_index,
                "Visibility": visibility,
                "X": int(x_orig),
                "Y": int(y_orig),
                "W": int(w_orig),
                "H": int(h_orig),
            }
            append_to_csv(result, csv_path)

            if args.visualize or out_writer is not None:
                vis_frame = frame.copy()
                vis_frame = draw_track(vis_frame, track_points)
                if args.visualize:
                    cv2.namedWindow("Ball Tracking", cv2.WINDOW_NORMAL)
                    cv2.imshow("Ball Tracking", vis_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                if out_writer is not None:
                    out_writer.write(vis_frame)

        end_time = time.time()
        batch_time = end_time - start_time
        batch_fps = 1 / batch_time if batch_time > 0 else 0

        pbar.update(1)
        frame_index += 1

    pbar.close()
    cap.release()
    if out_writer is not None:
        out_writer.release()
    if args.visualize:
        cv2.destroyAllWindows()
"""
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

    # Buffer de entrada para la inferencia del modelo ONNX (3 frames)
    frame_buffer = deque(maxlen=3)
    
    # 1. RETARDO DE RENDERIZADO: Cola para guardar los frames crudos antes de dibujar
    DELAY_FRAMES = 15  
    render_queue = deque()

    # 2. Historial global continuo de la trayectoria
    track_points = []
    
    frame_index = 0
    pbar = tqdm(total=total_frames, desc="Processing video", unit="frame")

    while cap.isOpened():
        start_time = time.time()

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
            
            # La función gestiona la racha, los None y la interpolación en 'track_points'
            predictions[2] = procesar_y_reparar_trayectoria(
                predictions[2], 
                track_points, 
                frame_width, 
                frame_height, 
                input_width, 
                input_height,
                max_dist=100
            )

            # Se guarda el frame original con su número de frame en el buffer de retardo
            render_queue.append((frame.copy(), frame_index))

            # 3. Dibujar y guardar solo cuando la cola supere el tiempo de retardo
            if len(render_queue) > DELAY_FRAMES:
                delayed_frame, delayed_idx = render_queue.popleft()
                
                # Extraer la estela de puntos (track_length) correspondiente a este frame diferido
                start_idx = max(0, delayed_idx - args.track_length + 1)
                pts_slice = track_points[start_idx : delayed_idx + 1]
                
                # Guardar datos en CSV para el frame diferido
                p_curr = track_points[delayed_idx]
                vis_curr = 1 if p_curr is not None else 0
                x_curr, y_curr = p_curr if p_curr is not None else (-1, -1)
                
                result = {
                    "Frame": delayed_idx,
                    "Visibility": vis_curr,
                    "X": x_curr,
                    "Y": y_curr,
                    "W": 0,
                    "H": 0,
                }
                append_to_csv(result, csv_path)

                # Renderizar la trayectoria diferida
                if args.visualize or out_writer is not None:
                    vis_frame = draw_track(delayed_frame, pts_slice)
                    if args.visualize:
                        cv2.namedWindow("Ball Tracking", cv2.WINDOW_NORMAL)
                        cv2.imshow("Ball Tracking", vis_frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                    if out_writer is not None:
                        out_writer.write(vis_frame)

        end_time = time.time()
        pbar.update(1)
        frame_index += 1

    # 4. VACIAR EL BUFFER AL FINALIZAR EL VIDEO (para procesar los últimos 15 frames)
    while render_queue:
        delayed_frame, delayed_idx = render_queue.popleft()
        start_idx = max(0, delayed_idx - args.track_length + 1)
        pts_slice = track_points[start_idx : delayed_idx + 1]
        
        p_curr = track_points[delayed_idx]
        vis_curr = 1 if p_curr is not None else 0
        x_curr, y_curr = p_curr if p_curr is not None else (-1, -1)
        
        result = {
            "Frame": delayed_idx,
            "Visibility": vis_curr,
            "X": x_curr,
            "Y": y_curr,
            "W": 0,
            "H": 0,
        }
        append_to_csv(result, csv_path)

        if args.visualize or out_writer is not None:
            vis_frame = draw_track(delayed_frame, pts_slice)
            if args.visualize:
                cv2.imshow("Ball Tracking", vis_frame)
                cv2.waitKey(1)
            if out_writer is not None:
                out_writer.write(vis_frame)

    pbar.close()
    cap.release()
    if out_writer is not None:
        out_writer.release()
    if args.visualize:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
