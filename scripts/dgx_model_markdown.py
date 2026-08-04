#!/usr/bin/env python3
"""Render untrusted Hugging Face Markdown safely in a terminal."""

from __future__ import annotations

import shutil
import sys


def compact_markdown(markup: str):
    """Create Rich Markdown with restrained terminal-friendly block styles."""
    from rich.markdown import CodeBlock, Heading, ImageItem, Markdown
    from rich.syntax import Syntax
    from rich.text import Text

    class CompactHeading(Heading):
        def __rich_console__(self, console, options):
            self.text.justify = None
            yield self.text

    class CompactImage(ImageItem):
        def __rich_console__(self, console, options):
            title = self.text or Text("image")
            yield title

    class CompactCodeBlock(CodeBlock):
        def __rich_console__(self, console, options):
            code = str(self.text).rstrip()
            yield Syntax(
                code,
                self.lexer_name,
                theme=self.theme,
                word_wrap=True,
                background_color="default",
                padding=(0, 1),
            )

    class CompactMarkdown(Markdown):
        elements = {
            **Markdown.elements,
            "heading_open": CompactHeading,
            "image": CompactImage,
            "fence": CompactCodeBlock,
            "code_block": CompactCodeBlock,
        }

    return CompactMarkdown(markup, code_theme="monokai")


def sanitize_terminal_text(value: str) -> str:
    """Remove terminal control characters while preserving Markdown layout."""
    return "".join(
        character
        for character in value
        if character in "\n\t" or (ord(character) >= 32 and ord(character) != 127)
    )


def main() -> None:
    try:
        from rich.console import Console
    except ImportError:
        raise SystemExit(
            "Markdown preview requires the Python package 'rich'. "
            "Use --raw to print the original Markdown."
        )

    source = sanitize_terminal_text(sys.stdin.read())
    width = shutil.get_terminal_size(fallback=(100, 24)).columns
    console = Console(
        width=max(20, width),
        force_terminal=sys.stdout.isatty(),
        highlight=False,
        soft_wrap=False,
    )
    console.print(compact_markdown(source))


if __name__ == "__main__":
    main()
