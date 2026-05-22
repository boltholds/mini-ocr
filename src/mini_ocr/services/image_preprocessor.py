from pathlib import Path
import cv2


class ImagePreprocessor:
    def preprocess(self, image_path: Path) -> Path:
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return image_path
        img = cv2.fastNlMeansDenoising(img, None, 10, 7, 21)
        img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        out = image_path.with_name(image_path.stem + "_prep.png")
        cv2.imwrite(str(out), img)
        return out
