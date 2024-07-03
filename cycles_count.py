import torch
import torchvision.transforms as T
import torchvision
import numpy as np
from PIL import Image
import cv2

# Загрузка предобученной модели с использованием параметра 'weights'
weights = torchvision.models.segmentation.DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1
model = torchvision.models.segmentation.deeplabv3_resnet101(weights=weights).eval().cuda()


# Функция для предобработки изображения
def preprocess(image):
    transform = T.Compose([
        T.Resize(520),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).cuda()


# Функция для постобработки и получения сегментированной маски
def decode_segmap(image, nc=21):
    label_colors = np.array([
        (0, 0, 0),  # 0=background
        # Цвета для остальных классов
        (128, 0, 0),  # 1=aeroplane
        (0, 128, 0),  # 2=bicycle
        (128, 128, 0),  # 3=bird
        (0, 0, 128),  # 4=boat
        (128, 0, 128),  # 5=bottle
        (0, 128, 128),  # 6=bus
        (128, 128, 128),  # 7=car
        (64, 0, 0),  # 8=cat
        (192, 0, 0),  # 9=chair
        (64, 128, 0),  # 10=cow
        (192, 128, 0),  # 11=dining table
        (64, 0, 128),  # 12=dog
        (192, 0, 128),  # 13=horse
        (64, 128, 128),  # 14=motorbike
        (192, 128, 128),  # 15=person
        (0, 64, 0),  # 16=potted plant
        (128, 64, 0),  # 17=sheep
        (0, 192, 0),  # 18=sofa
        (128, 192, 0),  # 19=train
        (0, 64, 128)  # 20=tv/monitor
    ])

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]

    rgb = np.stack([r, g, b], axis=2)
    return rgb


# Функция для сегментации одного кадра
def segment_frame(frame):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    input_image = preprocess(image)

    with torch.no_grad():
        output = model(input_image)['out'][0]
    output_predictions = output.argmax(0).cpu().numpy()

    return output_predictions


# Функция для нахождения пятен
def find_spots(segmented_frame):
    spots = {}

    # Конвертируем изображение в 8-битное
    segmented_frame_8u = segmented_frame.astype(np.uint8)

    num_labels, labels_im = cv2.connectedComponents(segmented_frame_8u)

    for label in range(1, num_labels):
        mask = labels_im == label
        spots[label] = mask

    return spots


# Класс для отслеживания пятен
class Spot:
    def __init__(self, id, mask):
        self.id = id
        self.max_bounds = self.calculate_bounds(mask)
        self.bounds_history = [self.max_bounds]
        self.cycle_count = 0
        self.is_stable = False
        self.reference_frame = mask
        self.stable_bounds = None
        self.pixel_change_count = None  # Массив для подсчета изменений пикселей

    def calculate_bounds(self, mask):
        y_indices, x_indices = np.where(mask)
        if len(y_indices) == 0 or len(x_indices) == 0:
            return None
        return (x_indices.min(), y_indices.min(), x_indices.max(), y_indices.max())

    def update(self, mask, frame_number, stability_threshold, similarity_threshold, frame_history_length):
        new_bounds = self.calculate_bounds(mask)

        if new_bounds:
            self.max_bounds = (
                min(self.max_bounds[0], new_bounds[0]),
                min(self.max_bounds[1], new_bounds[1]),
                max(self.max_bounds[2], new_bounds[2]),
                max(self.max_bounds[3], new_bounds[3])
            )
            self.bounds_history.append(self.max_bounds)

        # Проверка на колебания рамки за последние frame_history_length кадров
        if len(self.bounds_history) >= frame_history_length:
            recent_bounds = self.bounds_history[-frame_history_length:]
            min_x1 = min([b[0] for b in recent_bounds])
            min_y1 = min([b[1] for b in recent_bounds])
            max_x2 = max([b[2] for b in recent_bounds])
            max_y2 = max([b[3] for b in recent_bounds])
            stable_bounds = (min_x1, min_y1, max_x2, max_y2)

            width_changes = [abs((b[2] - b[0]) - (max_x2 - min_x1)) for b in recent_bounds]
            height_changes = [abs((b[3] - b[1]) - (max_y2 - min_y1)) for b in recent_bounds]

            if max(width_changes) <= stability_threshold and max(height_changes) <= stability_threshold:
                if not self.is_stable:
                    self.is_stable = True
                    self.reference_frame = mask
                    self.stable_bounds = stable_bounds
                    self.pixel_change_count = np.zeros((max_y2 - min_y1 + 1, max_x2 - min_x1 + 1), dtype=np.float32)
                    print(f"Stable bounds found for spot {self.id} at frame {frame_number}: {stable_bounds}")

                    # Проход по сегментированным кадрам до нахождения стабильной рамки для подсчета изменений
                    for past_frame in segmented[:-frame_history_length]:
                        self.update_pixel_change_count(past_frame, mask)
            else:
                self.is_stable = False
                self.cycle_count = 0
                self.stable_bounds = None
                self.pixel_change_count = None

        if self.is_stable:
            self.update_pixel_change_count(segmented[-2], mask)
            if any([
                (new_bounds[2] - new_bounds[0] - (self.stable_bounds[2] - self.stable_bounds[0])) > stability_threshold,
                (new_bounds[3] - new_bounds[1] - (self.stable_bounds[3] - self.stable_bounds[1])) > stability_threshold
            ]):
                self.pixel_change_count.fill(0)
                self.pixel_change_count = np.zeros(
                    (new_bounds[3] - new_bounds[1] + 1, new_bounds[2] - new_bounds[0] + 1), dtype=np.float32)

    def update_pixel_change_count(self, prev_frame, curr_frame):
        if self.stable_bounds is None:
            return

        min_x, min_y, max_x, max_y = self.stable_bounds
        prev_cropped = prev_frame[min_y:max_y + 1, min_x:max_x + 1]
        curr_cropped = curr_frame[min_y:max_y + 1, min_x:max_x + 1]

        if prev_cropped.shape != curr_cropped.shape:
            print(
                f"Shape mismatch in cropped frames: prev_cropped {prev_cropped.shape}, curr_cropped {curr_cropped.shape}")
            cv2.imwrite('prev_cropped.png', prev_cropped.astype(np.uint8))
            cv2.imwrite('curr_cropped.png', curr_cropped.astype(np.uint8))
            exit()

        # Инициализация массива changes нулями с shape curr_cropped
        changes = np.zeros((curr_cropped.shape[0], curr_cropped.shape[1]), dtype=np.float32)

        #self.pixel_change_count = np.random.rand(self.pixel_change_count.shape[0],
        #                                         self.pixel_change_count.shape[1]) * 255
        # Использование numpy для проверки изменений в пределах порога
        threshold = 14
        # Использование numpy для проверки изменений в каждом компоненте r, g, b с допустимым отклонением в 20
        diff = np.abs(prev_cropped - curr_cropped)

        # Проверка изменений в каждом компоненте r, g, b с допустимым отклонением в threshold
        for i in range(curr_cropped.shape[0]):
            for j in range(curr_cropped.shape[1]):
                if (diff[i, j] > threshold).any():
                    changes[i, j] = 1.0

        # cv2.imwrite('diff.png', diff.astype(np.uint8))
        if self.pixel_change_count is None:
            self.pixel_change_count = changes
        else:
            self.pixel_change_count += changes


    def save_pixel_change_count_image(self, filename):
        if self.pixel_change_count is None:
            print("No pixel change count data available.")
            return

        # Нормализация значений в pixel_change_count
        max_value = np.max(self.pixel_change_count)
        if max_value == 0:
            normalized_pixel_change_count = self.pixel_change_count
        else:
            normalized_pixel_change_count = (self.pixel_change_count / max_value) * 255

        # Создание изображения
        pixel_change_count_image = normalized_pixel_change_count.astype(np.uint8)

        # Сохранение изображения в файл
        cv2.imwrite(filename, pixel_change_count_image)
        print(f"Pixel change count image saved to {filename}")

    def get_pixel_change_count_image(self):
        if self.pixel_change_count is None:
            return None


        # Нормализация значений в pixel_change_count
        max_value = np.max(self.pixel_change_count)
        if max_value == 0:
            normalized_pixel_change_count = self.pixel_change_count
        else:
            normalized_pixel_change_count = (self.pixel_change_count / max_value) * 255

        # Создание изображения
        pixel_change_count_image = normalized_pixel_change_count.astype(np.uint8)

        return pixel_change_count_image

    def calculate_cycle_count(self):
        if self.pixel_change_count is None:
            return 0

        # Найти уникальные значения и количество их повторений
        unique_values, counts = np.unique(self.pixel_change_count, return_counts=True)

        # Исключить нулевые значения
        non_zero_indices = unique_values > 0
        unique_values = unique_values[non_zero_indices]
        counts = counts[non_zero_indices]

        if len(unique_values) == 0:
            return 0

        # Найти значение, которое повторяется чаще всего
        most_common_value = unique_values[np.argmax(counts)]

        # Разделить это значение на 2.0
        cycle_count = most_common_value
                       #/ 2.0)

        return cycle_count

        #unique, counts = np.unique(self.pixel_change_count, return_counts=True)
        #max_count_value = unique[np.argmax(counts)]

        #return max_count_value / 2

    def get_max_bounds(self):
        return self.max_bounds

    def get_stable_bounds(self):
        return self.stable_bounds if self.is_stable else None


# Переменная для хранения всех сегментированных кадров
segmented = []


# Основная функция для сегментации видео и анализа пятен
def ss_video_and_count_cycles(video_path, output_path, pixel_change_output_path, stability_threshold,
                              similarity_threshold, frame_history_length):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Создание видео с сегментированными кадрами и рамками
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MJPG'), fps, (frame_width, frame_height))
    pixel_change_out = cv2.VideoWriter(pixel_change_output_path, cv2.VideoWriter_fourcc(*'MJPG'), fps,
                                       (frame_width, frame_height))

    if not out.isOpened():
        print(f"Error opening video writer: {output_path}")
        return

    if not pixel_change_out.isOpened():
        print(f"Error opening pixel change video writer: {pixel_change_output_path}")
        return

    frame_number = 0
    spots = {}
    next_spot_id = 1

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1
        segmented_frame = segment_frame(frame)
        colored_segmented_frame = decode_segmap(segmented_frame)
        segmented.append(segmented_frame)

        # Найти пятна на сегментированном кадре
        current_spots = find_spots(segmented_frame)

        for label, mask in current_spots.items():
            found_spot = None
            for spot in spots.values():
                if spot.id == label:
                    found_spot = spot
                    break

            if found_spot:
                found_spot.update(mask, frame_number, stability_threshold, similarity_threshold, frame_history_length)
            else:
                new_spot = Spot(next_spot_id, mask)
                spots[next_spot_id] = new_spot
                next_spot_id += 1

        # Добавление текущих желтых рамок для пятен
        for spot in spots.values():
            max_bounds = spot.get_max_bounds()
            if max_bounds:
                x1, y1, x2, y2 = max_bounds
                cv2.rectangle(colored_segmented_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)  # Желтый цвет (BGR)

            stable_bounds = spot.get_stable_bounds()
            if stable_bounds:
                x1, y1, x2, y2 = stable_bounds
                cv2.rectangle(colored_segmented_frame, (x1, y1), (x2, y2), (255, 255, 255), 2)  # Белый цвет (BGR)

                # Создание и добавление изображения pixel_change_count в кадр
                pixel_change_image = spot.get_pixel_change_count_image()
                # cv2.imwrite('diff.png', pixel_change_image.astype(np.uint8))
                if pixel_change_image is not None:
                    colored_segmented_frame[y1:y2 + 1, x1:x2 + 1] = cv2.cvtColor(pixel_change_image, cv2.COLOR_GRAY2BGR)

        resized_frame = cv2.resize(colored_segmented_frame, (frame_width, frame_height),
                                   interpolation=cv2.INTER_NEAREST)
        out.write(cv2.cvtColor(resized_frame, cv2.COLOR_RGB2BGR))
        pixel_change_out.write(frame)

    cap.release()
    out.release()
    pixel_change_out.release()
    cv2.destroyAllWindows()

    for spot_id, spot in spots.items():
        cycle_count = spot.calculate_cycle_count()
        print(f"Spot {spot_id}: {cycle_count} cycles")


# Путь к видео
video_path = 'D:\VideosTest\Man_kach.mp4'
output_path = 'D:\VideosTest\segmented_video_with_bounds.avi'
pixel_change_output_path = 'D:\VideosTest\pixel_change_video_with_bounds.avi'

# Сегментация и анализ видео с параметрами стабильности и сходства
stability_threshold = 20  # Колебания изменений рамки в пикселях
similarity_threshold = 0.9  # Необходимое сходство
frame_history_length = 15  # Количество кадров, соответствующее времени 0.5 секунд

ss_video_and_count_cycles(video_path, output_path, pixel_change_output_path, stability_threshold, similarity_threshold,
                          frame_history_length)


















