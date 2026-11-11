"""
Скрипт для интерактивного захвата датасета с веб-камеры в реальном времени.
Позволяет студенту за 2-3 минуты собрать 100% честный датасет со своего стола.
Управление:
- Пробел (SPACE): начать/приостановить захват кадров (сохраняется ~3 кадра в секунду).
- 'N': перейти к следующему классу.
- 'Q': выйти из программы.
"""
import os
import time
import cv2

CLASSES = ["cup", "keys", "headphones", "pen"]
TARGET_COUNT = 150
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset", "raw")

def capture_dataset():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ОШИБКА] Не удалось открыть веб-камеру.")
        return

    print("=== ИНТЕРАКТИВНЫЙ ЗАХВАТ ДАТАСЕТА ===")
    print("Инструкция:")
    print("  [ПРОБЕЛ] - Включить / Выключить запись кадров")
    print("  [N]      - Переключиться на следующий класс")
    print("  [Q]      - Выход\n")

    current_idx = 0
    recording = False
    last_save_time = 0
    save_interval = 0.25 # каждые 250мс сохраняем кадр

    while True:
        cls_name = CLASSES[current_idx]
        cls_dir = os.path.join(OUTPUT_DIR, cls_name)
        os.makedirs(cls_dir, exist_ok=True)
        current_count = len([f for f in os.listdir(cls_dir) if f.endswith(('.jpg', '.png'))])

        ret, frame = cap.read()
        if not ret:
            print("Ошибка получения кадра.")
            break

        now = time.time()
        if recording and (now - last_save_time >= save_interval):
            if current_count < TARGET_COUNT:
                filename = f"{cls_name}_webcam_{current_count+1:04d}.jpg"
                filepath = os.path.join(cls_dir, filename)
                cv2.imwrite(filepath, frame)
                last_save_time = now
                current_count += 1
            else:
                recording = False

        # Отрисовка статуса на экране
        display = frame.copy()
        color = (0, 255, 0) if recording else (0, 165, 255)
        status_text = f"REC [{cls_name.upper()}] ({current_count}/{TARGET_COUNT})" if recording else f"PAUSED [{cls_name.upper()}] ({current_count}/{TARGET_COUNT})"
        cv2.putText(display, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(display, "SPACE: Rec/Pause | N: Next class | Q: Quit", (20, display.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Dataset Collector - Desktop Objects", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            recording = not recording
            print(f"[{cls_name}] Запись: {'ВКЛ' if recording else 'ВЫКЛ'}")
        elif key == ord('n') or key == ord('N'):
            current_idx = (current_idx + 1) % len(CLASSES)
            recording = False
            print(f"Переключено на класс: {CLASSES[current_idx]}")
        elif key == ord('q') or key == ord('Q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Захват завершён.")

if __name__ == "__main__":
    capture_dataset()
