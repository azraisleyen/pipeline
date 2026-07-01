
import cv2
import torch

try:
    from kornia.feature import LoFTR
except Exception:
    LoFTR = None

from ..core.utils import resize_keep_aspect


class GeometricVerifier:
    def __init__(
        self,
        device="cpu",
        enabled=True,
        pretrained="outdoor",
        run_on_ambiguous_only=True,
        min_matches=8,
        min_inlier_ratio=0.25,
        max_reproj_error=4.0,
        max_image_size=480,
    ):
        self.device = device
        self.enabled = bool(enabled)
        self.pretrained = pretrained
        self.run_on_ambiguous_only = bool(run_on_ambiguous_only)
        self.min_matches = int(min_matches)
        self.min_inlier_ratio = float(min_inlier_ratio)
        self.max_reproj_error = float(max_reproj_error)
        self.max_image_size = int(max_image_size)

        self.matcher = None
        if self.enabled and LoFTR is not None:
            self.matcher = LoFTR(pretrained=pretrained).to(device).eval()

    def is_available(self):
        return self.enabled and self.matcher is not None

    def _to_gray_tensor(self, image_bgr):
        image_bgr = resize_keep_aspect(image_bgr, self.max_image_size)
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        tensor = torch.from_numpy(gray).float() / 255.0
        tensor = tensor.unsqueeze(0).unsqueeze(0).to(self.device)
        return tensor

    def verify(self, ref_img_bgr, det_img_bgr):
        result = {
            "used": False,
            "num_matches": 0,
            "num_inliers": 0,
            "inlier_ratio": 0.0,
            "geom_score": 0.0,
            "geom_pass": False,
            "error": None,
        }

        if not self.is_available():
            return result

        if ref_img_bgr is None or det_img_bgr is None:
            return result

        if ref_img_bgr.size == 0 or det_img_bgr.size == 0:
            return result

        try:
            result["used"] = True

            ref_t = self._to_gray_tensor(ref_img_bgr)
            det_t = self._to_gray_tensor(det_img_bgr)

            with torch.no_grad():
                out = self.matcher({"image0": ref_t, "image1": det_t})

            mkpts0 = out["keypoints0"].detach().cpu().numpy()
            mkpts1 = out["keypoints1"].detach().cpu().numpy()

            num_matches = len(mkpts0)
            result["num_matches"] = int(num_matches)

            if num_matches < 4:
                return result

            _, inliers = cv2.findHomography(
                mkpts0,
                mkpts1,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.max_reproj_error,
            )

            if inliers is None:
                return result

            num_inliers = int(inliers.sum())
            inlier_ratio = float(num_inliers) / float(max(num_matches, 1))

            result["num_inliers"] = num_inliers
            result["inlier_ratio"] = inlier_ratio
            result["geom_score"] = inlier_ratio
            result["geom_pass"] = (
                num_matches >= self.min_matches
                and inlier_ratio >= self.min_inlier_ratio
            )

            return result

        except Exception as exc:
            result["error"] = str(exc)
            return result
