import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import glob
import cv2
from concurrent.futures import ThreadPoolExecutor

import onnx
import onnxruntime
import numpy as np
from onnxruntime.datasets import get_example
from PIL import Image
from typing import (
    List,
    Optional,
)
from matplotlib import pyplot as plt
from matplotlib import patches
from matplotlib.axes import Subplot
import cv2
from tqdm import tqdm
import time

print(os.getcwd())
MAX_NUM_CLASSES = 200
rds = np.random.RandomState(seed=526)
COLORS = [((rds.random((3, )) * 0.6 + 0.3)) for _ in range(MAX_NUM_CLASSES)]


def _plot_bboxes(ax: Subplot,
                 image: np.ndarray,
                 bboxes: np.ndarray,
                 labels: np.ndarray,
                 scores: Optional[float],
                 score_threshold: Optional[float],
                 depth: np.ndarray,
                 category_id_to_name: dict,
                 ) -> Subplot:
    ax.axis('off')
    ax.imshow(image)
    h_size, w_size, _ = image.shape
    include_scores = scores is not None
    include_depth = depth is not None

    if labels.ndim == 2:
        labels = np.squeeze(labels, axis=-1)
    if include_scores:
        if scores.ndim == 2:
            scores = np.squeeze(scores, axis=-1)
        if score_threshold is None:
            score_threshold = 0.1
    else:
        scores = [None for _ in range(len(bboxes))]
    
    if include_depth:
        if depth.ndim == 2:
            depth = np.squeeze(depth, axis=-1)
    else:
        depth = [None for _ in range(len(bboxes))]

    for index_object in range(len(bboxes)):
        x_min, y_min, x_max, y_max = bboxes[index_object]
        label = labels[index_object]
        score = scores[index_object]
        dep = depth[index_object]

        bbox_h_size = y_max - y_min
        bbox_w_size = x_max - x_min

        is_bbox = bbox_h_size > 0 and bbox_w_size > 0
        cond_score = (include_scores and score >= score_threshold) or (not include_scores)

        if is_bbox and cond_score:
            x_min = int(x_min * w_size)
            x_max = int(x_max * w_size)
            y_min = int(y_min * h_size)
            y_max = int(y_max * h_size)

            # bbox
            rect = patches.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min,
                                     linewidth=2,
                                     edgecolor=COLORS[label],
                                     facecolor='none')
            ax.add_patch(rect)

            # label
            bbox_props = dict(boxstyle="square,pad=0",
                              linewidth=2,
                              facecolor=COLORS[label],
                              edgecolor=COLORS[label])

            class_name = category_id_to_name[label]
            if score is None and dep is None:
                label_info = f'{class_name}'
            elif score is not None and dep is None:
                label_info = f'{class_name}: {score*100:.1f}%'
            elif score is None and dep is not None:
                label_info = f'{class_name}: {dep:.1f}m'
            else:
                label_info = f'{class_name}: {score*100:.1f}%: {dep:.1f}m'

            ax.text(x_min, y_min, label_info,
                    ha="left", va="bottom", rotation=0,
                    size=10, color='w', fontweight='bold', bbox=bbox_props)

    return ax


def plot_images_with_bboxes(images: List[np.ndarray],
                            bboxes: List[np.ndarray],
                            labels: List[np.ndarray],
                            category_id_to_name: dict,
                            scores: List[float] = None,
                            depth: List[np.ndarray] = None,
                            score_threshold: Optional[float] = None,
                            savefig_path: str = None,
                            close_plt: bool = False,
                            ) -> None:
    assert isinstance(category_id_to_name, (dict, type(None)))
    assert isinstance(savefig_path, (str, type(None)))

    set_bboxes = bboxes is not None
    set_labels = labels is not None
    set_scores = scores is not None
    set_depth = depth is not None

    images = [images] if not isinstance(images, list) else images
    if set_bboxes:
        bboxes = [bboxes] if not isinstance(bboxes, list) else bboxes
    if set_labels:
        labels = [labels] if not isinstance(labels, list) else labels
    if set_scores:
        scores = [scores] if not isinstance(scores, list) else scores
    else:
        scores = [None for _ in range(len(images))]
    
    if set_depth:
        depth = [depth] if not isinstance(depth, list) else depth
    else:
        depth = [None for _ in range(len(images))]

    if set_bboxes and set_labels:
        assert (len(images) == len(bboxes)) and (len(labels) == len(bboxes)), (
            '[ERROR] length is not same: '
            f'images={len(images)}, bboxes={len(bboxes)}, labels={len(labels)}'
        )

    n_axes = len(images)
    if n_axes < 2:
        fig, ax = plt.subplots()
        if not set_bboxes:
            ax.axis('off')
            ax.imshow(images[0])
        else:
            ax = _plot_bboxes(
                ax,
                image=images[0],
                bboxes=bboxes[0],
                labels=labels[0],
                scores=scores[0],
                depth=depth[0],
                score_threshold=score_threshold,
                category_id_to_name=category_id_to_name,
            )
    else:
        fig, axes = plt.subplots(figsize=(12.0, 9.0), ncols=n_axes)
        if not set_bboxes:
            for ax, image in zip(axes, images):
                ax.axis('off')
                ax.imshow(image)
        else:
            for ax, image, bbox, label, score, dep in zip(axes, images, bboxes, labels, scores, depth):
                ax = _plot_bboxes(
                    ax,
                    image=image,
                    bboxes=bbox,
                    labels=label,
                    scores=score,
                    depth=dep,
                    score_threshold=score_threshold,
                    category_id_to_name=category_id_to_name,
                )
    fig.tight_layout()

    if savefig_path:
        fig.savefig(savefig_path, bbox_inches="tight")

    if close_plt:
        # https://qiita.com/Masahiro_T/items/bdd0482a8efd84cdd270
        plt.clf()
        plt.close()




def fast_grid_crop_custom_np(image, crop_params, target_size):
    """
    高速なクロップ＆リサイズ処理。
    OpenCVのリサイズを並列処理化し、クロップを最適化。

    Args:
        image (np.ndarray): 入力画像 (H, W, C)
        crop_params (np.ndarray): クロップパラメータ (N, 4) -> [y_min, x_min, y_max, x_max]
        target_size (tuple): 出力サイズ (height, width)
    
    Returns:
        np.ndarray: クロップ後リサイズされた画像 (N, target_size[0], target_size[1], C)
    """
    height, width = image.shape[:2]
    num_crops = crop_params.shape[0]
    
    # OpenCVの高速リサイズを並列処理
    def crop_and_resize(box):
        x_min, y_min, x_max, y_max = map(int, box)
        cropped = image[y_min:y_max, x_min:x_max]
        return cv2.resize(cropped, target_size, interpolation=cv2.INTER_LINEAR)
    
    # 並列処理
    with ThreadPoolExecutor() as executor:
        crops = list(executor.map(crop_and_resize, crop_params))

    return np.array(crops)

def fast_process_image_np(image, input_size, crop_params):
    """
    高速な画像処理を行い、クロップとリサイズを一括実行。

    Args:
        image (np.ndarray): 入力画像 (H, W, C)
        input_size (tuple): 出力サイズ (height, width, channels)
        crop_params (np.ndarray): クロップパラメータ (N, 4) -> [y_min, x_min, y_max, x_max]
    
    Returns:
        np.ndarray: 処理後の画像 (N+1, input_size[0], input_size[1], C)
    """
    target_size = (input_size[1], input_size[0])

    # 並列処理でクロップ＆リサイズ
    crops = fast_grid_crop_custom_np(image, crop_params, target_size)

    # OpenCVでオリジナル画像をリサイズ
    original_resized = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

    # 結果を結合
    resized_crops = np.concatenate([crops, original_resized[None, ...]], axis=0)

    return resized_crops


def _get_scale_and_shift(box, base_image_size):
    crop_height = box[3] - box[1]
    crop_width = box[2] - box[0]
    y_scale = crop_height / base_image_size[0]
    x_scale = crop_width / base_image_size[1]
    y_shift = box[1] / base_image_size[0]
    x_shift = box[0] / base_image_size[1]

    scale = [x_scale, y_scale, x_scale, y_scale, 1.0, 1.0]
    shift = [x_shift, y_shift, x_shift, y_shift, 0.0, 0.0]
    return scale, shift

def iou_np(a, b, a_area, b_area):
    # aは1つの矩形を表すshape=(4,)のnumpy配列
    # array([xmin, ymin, xmax, ymax])
    # bは任意のN個の矩形を表すshape=(N, 4)のnumpy配列
    # 2次元目の4は、array([xmin, ymin, xmax, ymax])

    # a_areaは矩形aの面積
    # b_areaはbに含まれる矩形のそれぞれの面積
    # shape=(N,)のnumpy配列。Nは矩形の数

    # aとbの矩形の共通部分(intersection)の面積を計算するために、
    # N個のbについて、aとの共通部分のxmin, ymin, xmax, ymaxを一気に計算
    abx_mn = np.maximum(a[0], b[:, 0])  # xmin
    aby_mn = np.maximum(a[1], b[:, 1])  # ymin
    abx_mx = np.minimum(a[2], b[:, 2])  # xmax
    aby_mx = np.minimum(a[3], b[:, 3])  # ymax
    # 共通部分の幅を計算。共通部分が無ければ0
    w = np.maximum(0, abx_mx - abx_mn + 1)
    # 共通部分の高さを計算。共通部分が無ければ0
    h = np.maximum(0, aby_mx - aby_mn + 1)
    # 共通部分の面積を計算。共通部分が無ければ0
    intersect = w*h

    # N個のbについて、aとのIoUを一気に計算
    iou_np = intersect / (a_area + b_area - intersect)
    return iou_np

def NMS(pred, th_iou=0.5, delta=2000, ):
    bboxes = pred[..., :4]
    scores = pred[..., 4]
    labels = pred[..., 5]
    # boxの面積のみ抽出
    areas = (bboxes[:, 2] - bboxes[:, 0] + 1) \
        * (bboxes[:, 3] - bboxes[:, 1] + 1)

    # NMSを適用
    # boxの座標シフト
    shift = delta * labels.astype(np.float32)
    before_nms = bboxes + shift[:, np.newaxis]
    sort_index = np.argsort(-scores)
    final_index = []
    while (len(sort_index) > 0):
        # confが最も高いboxを取得
        best_idx = sort_index[0]
        final_index.append(best_idx)
        if len(sort_index) == 1:
            break
        # IOU計算
        ind_list = sort_index[1:]
        iou = iou_np(before_nms[best_idx], before_nms[ind_list],
                     areas[best_idx], areas[ind_list])

        # IoUが閾値iou_threshold以上の矩形を計算
        sort_index = sort_index[1:][iou < th_iou]

    final_bboxes = bboxes[final_index]

    # 検出結果を作成 (バウンディングボックス、スコア、クラスID)
    detections = np.concatenate(
        [
            final_bboxes.astype(np.float32),
            scores[final_index, np.newaxis].astype(np.float32),  # NMS後のスコア
            labels[final_index, np.newaxis].astype(np.float32),  # NMS後のクラスID
        ],
        axis=-1,
    )

    return detections

if __name__ == "__main__":
    crop_params = [
        # [ x_min, y_min, x_max, y_max]
        # [0, 0, 600, 450],
        # [0, 375, 600, 825],
        # [0, 252, 768, 828],
        [200, 200, 800, 650],
        # [0, 750, 600, 1200],

        # [500, 0, 1100, 450],
        # [500, 375, 1100, 825],
        # [576, 252, 1344, 828],
        [660, 200, 1260, 650],
        # [500, 750, 1100, 1200],

        # [1000, 0, 1600, 450],
        # [1000, 375, 1600, 825],
        # [1152, 252, 1920, 828],
        [1120, 200, 1720, 650],
        # [1000, 750, 1600, 1200],
    ]


    class_id_to_name = {
        0: "car",
        1: "person",
        2: "biker",
        3: "traffic_light_blue",
        4: "traffic_light_yellow",
        5: "traffic_light_red",
        6: "traffic_light_unknown",
        7: "crosswalk",
        8: "stop_line",
        9: "stop_sign",
        10: "stop_mark",
    }
    # ignore_labels = [0, 1, 2, 7, 8, 10]
    ignore_labels = [7, 8]

    max_det = 300
    conf_thres = 0.05
    im_list = glob.glob(os.path.join(os.getcwd(), "images/*.jpg"), recursive=True)
    index = 0
    example_model = get_example(os.path.join(os.getcwd(), "models/best.onnx"))
    sess = onnxruntime.InferenceSession(example_model, 
                                        providers=['CUDAExecutionProvider'])
                                        # providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    model = onnx.load(example_model)
    onnx.checker.check_model(model)

    input_name = sess.get_inputs()[0].name
    print("Input name  :", input_name)
    input_shape = sess.get_inputs()[0].shape
    print("Input shape :", input_shape)
    input_type = sess.get_inputs()[0].type
    print("Input type  :", input_type)
    output_name = sess.get_outputs()[0].name
    print("Output name  :", output_name)  
    output_shape = sess.get_outputs()[0].shape
    print("Output shape :", output_shape)
    output_type = sess.get_outputs()[0].type
    print("Output type  :", output_type)

    example_model_depth = get_example(os.path.join(os.getcwd(), "models/depthmodel_full_v2.onnx"))
    sess2 = onnxruntime.InferenceSession(example_model_depth, 
                                        providers=['CUDAExecutionProvider'])
                                        # providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    depthmodel = onnx.load(example_model_depth)
    onnx.checker.check_model(depthmodel)
    # print(onnx.helper.printable_graph(depthmodel.graph))
    input_name2 = sess2.get_inputs()[0].name
    print("Input name  :", input_name2)
    input_shape2 = sess2.get_inputs()[0].shape
    print("Input shape :", input_shape2)
    input_type2 = sess2.get_inputs()[0].type
    print("Input type  :", input_type2)
    output_name2 = sess2.get_outputs()[0].name
    print("Output name  :", output_name2)  
    output_shape2 = sess2.get_outputs()[0].shape
    print("Output shape :", output_shape2)
    output_type2 = sess2.get_outputs()[0].type
    print("Output type  :", output_type2)

    image_file = '/home/share/embedded_ai/embedded_ai/yolo_datasets/valid/images/020191219125611511.jpg'
    flags: int = cv2.IMREAD_COLOR


    # path = "/mnt/processed_data/processed_data/1_20241119-011750/ECO-U0001/ECO-0001/undistorted_movie/undistorted_front_movie/1716234422-1716234446_turn_right.MP4"
    path = "/mnt/processed_data/video_confidential/20240807_labeling/input_data/movie/1716244831-1716244880_turn_right.MP4"
    # path = "/mnt/processed_data/processed_data/1_20241119-011750/ECO-U0001/ECO-0001/undistorted_movie/undistorted_front_movie/1716235037-1716235059_turn_left.MP4"
    # path = "/mnt/processed_data/processed_data/1_20241119-011750/ECO-U0001/ECO-0001/undistorted_movie/undistorted_front_movie/1716234976-1716234988_turn_left.MP4"
    
    cap = cv2.VideoCapture(path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    image_ids, _ = os.path.splitext(os.path.basename(path))
    frame_count_index = 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    BASE_IMAGE_SIZE = [height, width]
    scale_list = []
    shift_list = []
    for i in range(len(crop_params)):
        sca, shif = _get_scale_and_shift(box=crop_params[i], base_image_size=BASE_IMAGE_SIZE)
        scale_list.append(sca)
        shift_list.append(shif)
    scale_list.append([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    shift_list.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    scale_list = np.array(scale_list)
    shift_list = np.array(shift_list)
    index = 0
    with tqdm(total=frame_count) as pbar:
        while True:
            c0 = time.perf_counter()
            ret, frame = cap.read()
            check_point = time.perf_counter()
            if not ret:
                break
            c1 = time.perf_counter()
            # process for detection
            x = fast_process_image_np(frame, input_size=[576, 768, 3], crop_params=np.array(crop_params))
            x = x[..., ::-1].transpose((0, 3, 1, 2))
            x = x.astype(np.float32)
            x /= 255

            # preprocess for depth
            depth_input = np.stack([cv2.resize(frame, (640, 192), interpolation=cv2.INTER_LINEAR)])
            depth_input = depth_input[..., ::-1].transpose((0, 3, 1, 2))
            depth_input = depth_input.astype(np.float32)
            depth_input /= 255
            c2 = time.perf_counter()

            output_onnx = sess.run(None, {input_name: x})[0]
            c3 = time.perf_counter()

            output_depth = sess2.run(None, {input_name2: depth_input})
            c31 = time.perf_counter()

            # for detection output
            output = []
            for batch in range(3):
                detection = output_onnx[batch]
                mask = np.isin(detection[..., 5], ignore_labels)
                modified_detections = np.copy(detection)
                modified_detections[..., 4] = np.where(mask, 0., modified_detections[..., 4])
                detection[mask] = modified_detections[mask]
                detection = detection[detection[:, 4] > conf_thres] / [768, 576, 768, 576, 1, 1]
                detection *= scale_list[batch]
                detection += shift_list[batch]
                output.extend(detection)
            detection = output_onnx[3]
            detection = detection[detection[:, 4] > conf_thres] / [768, 576, 768, 576, 1, 1]
            output.extend(detection)

            output = NMS(np.reshape(np.array(output), (-1, 6)))
            output = np.reshape(output, (1, -1, 6))
            bbox = output[..., :4]
            score = output[..., 4]
            label = output[..., 5]

            # for depth output
            depth_image = np.squeeze(output_depth[3][0].transpose(1,2,0)*255).astype(np.uint8)


            c4 = time.perf_counter()


            pbar.set_postfix(
                FileRead=c1-c0,
                preprocess=c2-c1,
                detect=c3-c2,
                depth=c31-c3,
                postprocess=c4-c31
            )

            plot_images_with_bboxes(
                images=frame[..., ::-1],
                bboxes=bbox[0],
                category_id_to_name=class_id_to_name,
                scores=score[0],
                labels=label[0].astype(np.int32),
                score_threshold=0.1,
                savefig_path=f'/home/nemoto/yolo_output/{index:06}.png'
            )
            depth_image = np.squeeze(output_depth[3][0].transpose(1,2,0)*255).astype(np.uint8)
            depth_image = Image.fromarray(depth_image)
            depth_image.save(f'/home/nemoto/dnadepth_output/{i:06}.png')
            index += 1
            pbar.update(1)

    print("detection finish")
