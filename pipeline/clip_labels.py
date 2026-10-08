"""Zero-shot action and mood labelling with CLIP (runs on CPU, free)."""
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

MODEL_ID = "openai/clip-vit-base-patch32"
BATCH = 16

ACTIONS = {
    "resting": "a photo of a cat resting calmly",
    "looking": "a photo of a cat staring at a person's hand",
    "petted": "a photo of a person's hand gently petting a cat",
    "biting": "a photo of a cat biting a person's hand",
    "fighting": "a photo of a cat grabbing and kicking a hand with its paws",
}

MOODS = {
    "calm": "a photo of a calm relaxed cat",
    "curious": "a photo of a curious alert cat",
    "playful": "a photo of a playful excited kitten",
    "annoyed": "a photo of an annoyed cat with ears pulled back",
    "angry": "a photo of an angry aggressive cat",
}


class ClipLabeller:
    def __init__(self) -> None:
        self.model = CLIPModel.from_pretrained(MODEL_ID).eval()
        self.proc = CLIPProcessor.from_pretrained(MODEL_ID)
        self._text = {
            "action": self._embed_text(list(ACTIONS.values())),
            "mood": self._embed_text(list(MOODS.values())),
        }

    @torch.no_grad()
    def _embed_text(self, prompts: list[str]) -> torch.Tensor:
        inputs = self.proc(text=prompts, return_tensors="pt", padding=True)
        emb = self.model.get_text_features(**inputs)
        return emb / emb.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def _embed_images(self, paths: list[Path]) -> torch.Tensor:
        chunks = []
        for i in range(0, len(paths), BATCH):
            imgs = [Image.open(p).convert("RGB") for p in paths[i:i + BATCH]]
            emb = self.model.get_image_features(**self.proc(images=imgs, return_tensors="pt"))
            chunks.append(emb / emb.norm(dim=-1, keepdim=True))
        return torch.cat(chunks)

    @torch.no_grad()
    def label(self, paths: list[Path]) -> list[dict]:
        img = self._embed_images(paths)
        scale = self.model.logit_scale.exp()
        results = []
        probs = {
            kind: (scale * img @ txt.T).softmax(dim=-1)
            for kind, txt in self._text.items()
        }
        names = {"action": list(ACTIONS), "mood": list(MOODS)}
        for row in range(len(paths)):
            entry = {}
            for kind, p in probs.items():
                best = int(p[row].argmax())
                entry[kind] = names[kind][best]
                entry[f"{kind}_conf"] = round(float(p[row][best]), 3)
            results.append(entry)
        return results
