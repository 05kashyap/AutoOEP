# patch_deepface.py
import os
import site
import sys
import inspect

def get_deepface_path():
    """Find the path to the installed deepface package."""
    try:
        import deepface
        return os.path.dirname(inspect.getfile(deepface))
    except (ImportError, ModuleNotFoundError):
        print("Error: deepface is not installed. Please install it first using 'pip install deepface'")
        sys.exit(1)
    except Exception as e:
        # Fallback for environments where inspect might fail
        try:
            for packages_path in site.getsitepackages():
                deepface_path = os.path.join(packages_path, "deepface")
                if os.path.isdir(deepface_path):
                    print(f"Found deepface at: {deepface_path}")
                    return deepface_path
        except Exception:
            print(f"An unexpected error occurred while finding deepface path: {e}")
            sys.exit(1)
    
    print("Could not find the deepface installation path.")
    sys.exit(1)


def patch_deepface_py(deepface_path):
    """Patches the main DeepFace.py file."""
    file_path = os.path.join(deepface_path, "DeepFace.py")
    print(f"Patching {file_path}...")

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Add verify_with_landmarks function
    verify_with_landmarks_code = '''
def verify_with_landmarks(
    img1_path: Union[str, np.ndarray],
    face_landmarker_result1: Any,
    img2_path: Union[str, np.ndarray],
    face_landmarker_result2: Any,
    model_name: str = "VGG-Face",
    distance_metric: str = "cosine",
    enforce_detection: bool = True,
    align: bool = True,
    expand_percentage: int = 0,
    normalization: str = "base",
    silent: bool = False,
    threshold: Optional[float] = None,
    anti_spoofing: bool = False,
) -> Dict[str, Any]:
    """
    Verify if an image pair represents the same person or different persons using mediapipe face landmarks.
    Args:
        img1_path (str or np.ndarray): Path to the first image.
            Accepts exact image path as a string, numpy array (BGR).

        face_landmarker_result1 (mediapipe.tasks.python.vision.face_landmarker.FaceLandmarkerResult):
            Mediapipe face landmarker result for the first image.

        img2_path (str or np.ndarray): Path to the second image.
            Accepts exact image path as a string, numpy array (BGR).

        face_landmarker_result2 (mediapipe.tasks.python.vision.face_landmarker.FaceLandmarkerResult):
            Mediapipe face landmarker result for the second image.

        model_name (str): Model for face recognition. Options: VGG-Face, Facenet, Facenet512,
            OpenFace, DeepFace, DeepID, Dlib, ArcFace, SFace and GhostFaceNet (default is VGG-Face).

        distance_metric (string): Metric for measuring similarity. Options: 'cosine',
            'euclidean', 'euclidean_l2', 'angular' (default is cosine).

        enforce_detection (boolean): If no face is detected in an image, raise an exception.
            Set to False to avoid the exception for low-resolution images (default is True).

        align (bool): Flag to enable face alignment (default is True).

        expand_percentage (int): expand detected facial area with a percentage (default is 0).

        normalization (string): Normalize the input image before feeding it to the model.
            Options: base, raw, Facenet, Facenet2018, VGGFace, VGGFace2, ArcFace (default is base)

        silent (boolean): Suppress or allow some log messages for a quieter analysis process
            (default is False).

        threshold (float): Specify a threshold to determine whether a pair represents the same
            person or different individuals. This threshold is used for comparing distances.
            If left unset, default pre-tuned threshold values will be applied based on the specified
            model name and distance metric (default is None).

        anti_spoofing (boolean): Flag to enable anti spoofing (default is False).

    Returns:
        result (dict): A dictionary containing verification results.
    """
    return verification.verify_with_landmarks(
        img1_path=img1_path,
        face_landmarker_result1=face_landmarker_result1,
        img2_path=img2_path,
        face_landmarker_result2=face_landmarker_result2,
        model_name=model_name,
        distance_metric=distance_metric,
        enforce_detection=enforce_detection,
        align=align,
        expand_percentage=expand_percentage,
        normalization=normalization,
        silent=silent,
        threshold=threshold,
        anti_spoofing=anti_spoofing,
    )

'''
    if "def verify_with_landmarks(" not in content:
        content = content.replace(
            "def analyze(", verify_with_landmarks_code + "\n\ndef analyze("
        )
        print("  + Added verify_with_landmarks function.")

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    print("  -> Done.")


def patch_verification_py(deepface_path):
    """Patches the verification.py file."""
    file_path = os.path.join(deepface_path, "modules", "verification.py")
    print(f"Patching {file_path}...")

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Add imports
    if "from deepface.commons import image_utils" not in content:
        content = content.replace(
            "from deepface.commons.logger import Logger",
            "from deepface.commons.logger import Logger\nfrom deepface.commons import image_utils",
        )
        print("  + Added image_utils import.")

    # Add verify_with_landmarks and its helper
    verify_with_landmarks_logic = '''
def verify_with_landmarks(
    img1_path: Union[str, np.ndarray],
    face_landmarker_result1: Any,
    img2_path: Union[str, np.ndarray],
    face_landmarker_result2: Any,
    model_name: str = "VGG-Face",
    distance_metric: str = "cosine",
    enforce_detection: bool = True,
    align: bool = True,
    expand_percentage: int = 0,
    normalization: str = "base",
    silent: bool = False,
    threshold: Optional[float] = None,
    anti_spoofing: bool = False,
) -> Dict[str, Any]:
    """
    Verify if an image pair represents the same person or different persons using mediapipe face landmarks.
    """

    tic = time.time()

    model: FacialRecognition = modeling.build_model(
        task="facial_recognition", model_name=model_name
    )

    img1_embedding, img1_facial_area = __extract_face_from_landmarks_and_embedding(
        img_path=img1_path,
        face_landmarker_result=face_landmarker_result1,
        model=model,
        enforce_detection=enforce_detection,
        align=align,
        normalization=normalization,
        anti_spoofing=anti_spoofing,
        expand_percentage=expand_percentage,
    )

    img2_embedding, img2_facial_area = __extract_face_from_landmarks_and_embedding(
        img_path=img2_path,
        face_landmarker_result=face_landmarker_result2,
        model=model,
        enforce_detection=enforce_detection,
        align=align,
        normalization=normalization,
        anti_spoofing=anti_spoofing,
        expand_percentage=expand_percentage,
    )

    distance = find_distance(img1_embedding, img2_embedding, distance_metric)

    pretuned_threshold = find_threshold(model_name, distance_metric)
    threshold = threshold or pretuned_threshold
    distance = float(distance)
    verified = distance <= pretuned_threshold
    confidence = find_confidence(
        distance=distance,
        model_name=model_name,
        distance_metric=distance_metric,
        verified=bool(verified),
    )

    toc = time.time()

    resp_obj = {
        "verified": verified,
        "distance": distance,
        "threshold": threshold,
        "confidence": confidence,
        "model": model_name,
        "detector_backend": "mediapipe",
        "similarity_metric": distance_metric,
        "facial_areas": {"img1": img1_facial_area, "img2": img2_facial_area},
        "time": round(toc - tic, 2),
    }

    return resp_obj


def __extract_face_from_landmarks_and_embedding(
    img_path: Union[str, np.ndarray],
    face_landmarker_result: Any,
    model: FacialRecognition,
    enforce_detection: bool,
    align: bool,
    normalization: str,
    anti_spoofing: bool,
    expand_percentage: int,
) -> Tuple[List[float], dict]:

    img, _ = image_utils.load_image(img_path)
    height, width, _ = img.shape

    if not face_landmarker_result.face_landmarks:
        if enforce_detection:
            raise ValueError("No face landmarks found in the provided result.")
        target_size = model.input_shape
        face_img = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
        embedding_obj = representation.represent(
            img_path=face_img,
            model_name=model.model_name,
            enforce_detection=False,
            detector_backend="skip",
            align=False,
            normalization=normalization,
        )
        embedding = embedding_obj[0]["embedding"]
        return embedding, {"x": 0, "y": 0, "w": width, "h": height}

    largest_face_landmarks = None
    max_area = -1
    facial_area = {}

    for landmarks in face_landmarker_result.face_landmarks:
        x_coords = [landmark.x * width for landmark in landmarks]
        y_coords = [landmark.y * height for landmark in landmarks]
        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))
        area = (x_max - x_min) * (y_max - y_min)
        if area > max_area:
            max_area = area
            largest_face_landmarks = landmarks
            facial_area = {"x": x_min, "y": y_min, "w": x_max - x_min, "h": y_max - y_min}

    face_img = img
    if largest_face_landmarks:
        if align:
            left_eye_indices = [33, 133]
            right_eye_indices = [362, 263]

            left_eye = (
                int(largest_face_landmarks[left_eye_indices[0]].x * width),
                int(largest_face_landmarks[left_eye_indices[0]].y * height),
            )
            right_eye = (
                int(largest_face_landmarks[right_eye_indices[0]].x * width),
                int(largest_face_landmarks[right_eye_indices[0]].y * height),
            )
            
            aligned_img, angle = detection.align_img_wrt_eyes(img=img, left_eye=left_eye, right_eye=right_eye)
            x, y, w, h = facial_area["x"], facial_area["y"], facial_area["w"], facial_area["h"]
            rx1, ry1, rx2, ry2 = detection.project_facial_area(
                facial_area=(x, y, x + w, y + h), angle=angle, size=(img.shape[0], img.shape[1])
            )
            face_img = aligned_img[int(ry1):int(ry2), int(rx1):int(rx2)]
        else:
            x, y, w, h = facial_area["x"], facial_area["y"], facial_area["w"], facial_area["h"]
            if expand_percentage > 0:
                expanded_w = w + int(w * expand_percentage / 100)
                expanded_h = h + int(h * expand_percentage / 100)
                x = max(0, x - int((expanded_w - w) / 2))
                y = max(0, y - int((expanded_h - h) / 2))
                w = min(width - x, expanded_w)
                h = min(height - y, expanded_h)
            face_img = img[y : y + h, x : x + w]

    if anti_spoofing:
        antispoof_model = modeling.build_model(task="spoofing", model_name="Fasnet")
        x, y, w, h = facial_area["x"], facial_area["y"], facial_area["w"], facial_area["h"]
        is_real, _ = antispoof_model.analyze(img=img, facial_area=(x, y, w, h))
        if not is_real:
            raise ValueError("Spoof detected in the provided image.")

    embedding_obj = representation.represent(
        img_path=face_img,
        model_name=model.model_name,
        enforce_detection=False,
        detector_backend="skip",
        align=False,
        normalization=normalization,
    )

    return embedding_obj[0]["embedding"], facial_area

'''
    if "def verify_with_landmarks(" not in content:
        content = content.replace(
            "def find_cosine_distance(", verify_with_landmarks_logic + "\n\ndef find_cosine_distance("
        )
        print("  + Added verify_with_landmarks logic.")

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    print("  -> Done.")


if __name__ == "__main__":
    deepface_install_path = get_deepface_path()
    if deepface_install_path:
        patch_deepface_py(deepface_install_path)
        patch_verification_py(deepface_install_path)
        print("\nDeepFace has been patched successfully!")