"""Rank stock-video search results by how well their preview images match the query
(CLIP image-text similarity), so a story scene gets footage that actually shows it."""
from io import BytesIO

import requests
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from clip_labels import MODEL_ID

MAX_PICTURES = 3        # preview frames per candidate (Pexels gives several)


class ClipRanker:
    def __init__(self) -> None:
        self.model = CLIPModel.from_pretrained(MODEL_ID).eval()
        self.proc = CLIPProcessor.from_pretrained(MODEL_ID)

    @staticmethod
    def _pictures(video: dict) -> list[Image.Image]:
        urls = [p["picture"] for p in video.get("video_pictures", [])[:MAX_PICTURES]] or [video["image"]]
        images = []
        for url in urls:
            try:
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
                images.append(Image.open(BytesIO(resp.content)).convert("RGB"))
            except (requests.RequestException, OSError):
                continue  # a missing thumbnail just lowers this candidate's evidence
        return images

    @torch.no_grad()
    def score(self, query: str, videos: list[dict]) -> list[float]:
        """Mean cosine similarity of each video's preview frames to 'a photo of <query>'."""
        text = self.proc(text=[f"a photo of {query}"], return_tensors="pt", padding=True)
        t = self.model.get_text_features(**text)
        t = t / t.norm(dim=-1, keepdim=True)
        scores = []
        for v in videos:
            images = self._pictures(v)
            if not images:
                scores.append(-1.0)
                continue
            emb = self.model.get_image_features(**self.proc(images=images, return_tensors="pt"))
            emb = emb / emb.norm(dim=-1, keepdim=True)
            scores.append(float((emb @ t.T).mean()))
        return scores
