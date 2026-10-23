"""Bound DeepSeek image history by its context, honoring explicit count policy.

Pinned TensorFold 07aa025 has a separate DsVision path with a hardcoded
eight-image default. The generic App's --vision-max-images never reaches it.
Keep all images, unchanged and in order; stop decoding once image tokens alone
exceed the request window. The existing App still checks text + images + reply.
"""

from pathlib import Path


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError("Pinned DeepSeek image source changed: " + old[:100])
    return source.replace(old, new)


def patch(root):
    path = root / "vision.py"
    source = path.read_text()
    source = replace_once(source, "max_images: int = 8,", "max_images: int | None = None,")
    source = replace_once(source,
        "    def split(self, messages: list) -> tuple[list, list[Picture]]:",
        "    def split(self, messages: list, *, max_images: int | None = None,\n"
        "              max_tokens: int | None = None) -> tuple[list, list[Picture]]:")
    source = replace_once(source, "        out, pictures = [], []\n",
        "        out, pictures = [], []\n"
        "        image_tokens = 0\n"
        "        count_limit = self.max_images if max_images is None else max_images\n")
    source = replace_once(source,
        "                    if len(pictures) >= self.max_images:\n"
        '                        raise ValueError(f"a request may carry at most {self.max_images} images")',
        "                    if count_limit is not None and len(pictures) >= count_limit:\n"
        '                        raise ValueError(f"a request may carry at most {count_limit} images")')
    source = replace_once(source,
        "                    pictures.append(decode(load_image_bytes(url, self.allow_urls, self.max_bytes), self.cfg))",
        "                    picture = decode(load_image_bytes(url, self.allow_urls, self.max_bytes), self.cfg)\n"
        "                    image_tokens += picture.tokens\n"
        "                    if max_tokens is not None and image_tokens >= max_tokens:\n"
        "                        from tensorfold.server.errors import CONTEXT_LIMIT\n"
        '                        raise ValueError(f"{CONTEXT_LIMIT} {max_tokens} tokens; "\n'
        '                                         f"images require {image_tokens} tokens before text and reply")\n'
        "                    pictures.append(picture)")
    compile(source, str(path), "exec")
    path.write_text(source)

    path = root / "app.py"
    source = path.read_text()
    source = replace_once(source,
        "from tensorfold.server.errors import RequestError",
        "from tensorfold.server.errors import CONTEXT_LIMIT, RequestError, refusal")
    source = replace_once(source, "        super().__init__(*args, **kwargs)\n",
        "        super().__init__(*args, **kwargs)\n"
        "        self.vision_max_images = kwargs.get(\"vision_max_images\")\n")
    source = replace_once(source,
        "            text_messages, pictures = vision.split(messages)\n"
        "        except (ValueError, OSError) as exc:\n"
        '            raise RequestError(f"image input: {exc}") from exc',
        "            text_messages, pictures = vision.split(\n"
        "                messages, max_images=getattr(self, \"vision_max_images\", None),\n"
        "                max_tokens=self._context_limit())\n"
        "        except (ValueError, OSError) as exc:\n"
        "            if str(exc).startswith(CONTEXT_LIMIT):\n"
        "                raise refusal(str(exc)) from exc\n"
        '            raise RequestError(f"image input: {exc}") from exc')
    compile(source, str(path), "exec")
    path.write_text(source)


if __name__ == "__main__":
    patch(Path("/opt/tensorfold/src/tensorfold/families/deepseek_v41/cuda"))
